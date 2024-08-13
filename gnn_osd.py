"""
You need a virtual environment (clean one)
pip uninstall panqec
pip uninstall ldpc

Afterwards

git clone -b public_osd https://github.com/alexandrupaler/ldpc.git
pip install -e ldpc

pip install panqec
"""

import ldpc

from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder, BeliefPropagationOSDDecoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import sympy
from codes_q import *

from panq_functions import GNNDecoder, collate, fraction_of_solved_puzzles,compute_accuracy, \
    surface_code_edges, generate_syndrome_error_volume, adapt_trainset,ler_loss,load_model


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    use_amp = True  #to use automatic mixed precision
    amp_data_type = torch.float16
else:
    device = torch.device('cpu')
    use_amp = False
    '''float16 is not supported for cpu use bfloat16 instead'''
    amp_data_type = torch.bfloat16


def init_log_probs_of_decoder(decoder, my_log_probs):
    #print("old ", decoder.log_prob_ratios)

    for i in range(len(decoder.log_prob_ratios)):
        decoder.set_log_prob(i, my_log_probs[i])

    #print("new ", decoder.log_prob_ratios)

def logical_error_rate_osd(gnn, testloader, code, osd_decoder=None, enable_osd=False):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    gnn.eval()
    device = gnn.device
    with torch.no_grad():
        n_test = 0
        n_l_error = 0
        n_codespace_error = 0
        n_total_ler = 0
        hx = torch.tensor(code.Hx.A, dtype=torch.float16, device=device)
        hz = torch.tensor(code.Hz.A, dtype=torch.float16, device=device)
        lx = torch.tensor(code.logicals_x[:, :code.n], dtype=torch.float16, device=device)
        lz = torch.tensor(code.logicals_z[:, code.n:], dtype=torch.float16, device=device)

        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
            # batch_size = inputs.size(0) // size
            outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
            encoding = outputs.shape[-1]

            if enable_osd:
                # final_solution = osd(outputs,targets,code,osd_decoder)
                n_l, n_c, n_t, batch_size = osd(outputs, targets, code, hx, hz, osd_decoder)
                # solution = outputs.view(gnn.n_iters, -1, size, encoding)
                # final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()

                n_l_error += n_l
                n_codespace_error += n_c
                n_total_ler += n_t
                n_test += batch_size

            else:
                """new for gpu"""
                solution = outputs.view(gnn.n_iters, -1, size, encoding)
                final_solution = solution[:, :, error_index:].argmax(dim=-1)
                batch_size = final_solution.shape[1]

                final_targets = targets.view(batch_size, size)[:, error_index:]
                final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                                 final_targets, 0) // 3
                final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(
                    final_targets == 3,
                    final_targets, 0) // 3

                final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                                    final_solution,
                                                                                                    0) // 3
                final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(
                    final_solution == 3, final_solution, 0) // 3

                # final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
                # final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)
                n_iters = final_solution.shape[0]
                final_targetsx = final_targetsx.unsqueeze(0).repeat(n_iters, 1, 1)
                final_targetsz = final_targetsz.unsqueeze(0).repeat(n_iters, 1, 1)

                rfx = ((final_targetsx + final_solutionx) % 2).type(torch.float16)
                rfz = ((final_targetsz + final_solutionz) % 2).type(torch.float16)

                ms = torch.cat(((rfx @ hz.T) % 2, (rfz @ hx.T) % 2), dim=-1).type(torch.int)
                mseitr = torch.any(ms, axis=2)
                minitr = torch.argmin(mseitr.sum(axis=-1))
                mse = mseitr[minitr]
                n_codespace_error += mse.sum().item()

                l = torch.cat(((rfx[minitr] @ lz.T) % 2, (rfz[minitr] @ lx.T) % 2), dim=-1).type(torch.int)
                l = torch.any(l, axis=1)
                n_l_error += l.sum().item()

                n_total_ler += torch.logical_or(l, mse).sum().item()
                # n_total_ler += l.sum() + mse.sum()
                n_test += batch_size
        # n_total_ler = (n_l_error + n_codespace_error)
        return (n_l_error / n_test), (n_codespace_error / n_test), n_total_ler / n_test

def osd(outputs, targets, code,hx,hz, osd_decoder=None):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    # n_iters=out.shape[0]
    encoding = outputs.shape[-1]
    solution = outputs.view(gnn.n_iters, -1, size, encoding)
    final_solution = solution[:, :, error_index:].argmax(dim=-1)
    batch_size = final_solution.shape[1]

    final_targets = targets.view(batch_size, size)[:, error_index:]
    final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                     final_targets, 0) // 3
    final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3,
                                                                                          final_targets, 0) // 3

    final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                        final_solution, 0) // 3
    final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(
        final_solution == 3, final_solution, 0) // 3

    # final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
    # final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)
    n_iters = final_solution.shape[0]
    final_targetsx = final_targetsx.unsqueeze(0).repeat(n_iters, 1, 1)
    final_targetsz = final_targetsz.unsqueeze(0).repeat(n_iters, 1, 1)

    # gpu part
    rfx = ((final_targetsx + final_solutionx) % 2).type(torch.float16)
    rfz = ((final_targetsz + final_solutionz) % 2).type(torch.float16)

    ms = torch.cat(((rfx @ hz.T) % 2, (rfz @ hx.T) % 2), dim=-1).type(torch.int)
    mseitr = torch.any(ms, axis=2)
    minitr = torch.argmin(mseitr.sum(axis=-1))
    mse = np.array(mseitr[minitr].type(torch.int).cpu())
    nonzero_syn_id = np.nonzero(mse.astype("uint8"))[0]
    # cpu part

    # rfx = np.array(((final_targetsx + final_solutionx) % 2).type(torch.int).cpu())

    # cpu part
    final_solution = np.array(nn.functional.softmax(solution[minitr, :, error_index:], dim=2).cpu())

    fllrx = -np.log((final_solution[:, :, 1] + final_solution[:, :, 3]) / final_solution[:, :, 0])  # works better
    fllrz = -np.log((final_solution[:, :, 2] + final_solution[:, :, 3]) / final_solution[:, :, 0])

    final_syn = np.array((targets.view(batch_size, size)[:, :error_index]).cpu())
    final_syn = np.append(final_syn[:, :error_index // 2], final_syn[:, error_index // 2:] // 2, axis=1)

    # final_solution = solution[minitr, :, error_index:].argmax(dim=2)
    # osd_out = np.array(final_solution.cpu())
    osd_err_x = np.array(final_solutionx[minitr].cpu())
    osd_err_z = np.array(final_solutionz[minitr].cpu())

    for i in nonzero_syn_id:
        init_log_probs_of_decoder(osd_decoder.x_decoder, fllrx[i])  #x with fllrx originally
        init_log_probs_of_decoder(osd_decoder.z_decoder, fllrz[i])
        # init_log_probs_of_decoder(osd_decoder.x_decoder, fnllr_sig[i])

        osd_err=osd_decoder.decode(final_syn[i])
        osd_err_x[i] = osd_err[:code.n]
        osd_err_z[i] = osd_err[code.n:]
        # osd_out[i] = osd_err[:error_index+1]
        # osd_out[i] = torch.LongTensor((osd_err[:code.d ** 2] + 2 * osd_err[code.d ** 2:]))

    rfx = ((np.array(final_targetsx[0].cpu()) + osd_err_x) % 2)
    rfz = ((np.array(final_targetsz[0].cpu()) + osd_err_z) % 2)

    ms = np.append((rfx @ code.Hz.T) % 2, (rfz @ code.Hx.T) % 2, axis=-1)
    mse = np.any(ms, axis=1)
    n_codespace_error = mse.sum()

    l = np.append((rfx @ code.logicals_z[:, code.n:].T) % 2, (rfz @ code.logicals_x[:, :code.n].T) % 2, axis=-1)
    l = np.any(l, axis=1)
    n_l_error = l.sum()

    n_total_ler = np.logical_or(l, mse).sum()
    # n_total_ler = n_l_error
    # n_codespace_error = 0
    # n_total_ler += l.sum() + mse.sum()
    n_test = batch_size

    return n_l_error, n_codespace_error, n_total_ler, n_test
    # return osd_out


d=11
#plot_code(code)
#surface_code_edges(code)
error_model_name = "DP"
if(error_model_name == "X"):
    error_model = PauliErrorModel(1, 0.0, 0)
elif (error_model_name == "Z"):
    error_model = PauliErrorModel(0, 0.0, 1)
elif (error_model_name == "XZ"):
    error_model = PauliErrorModel(0.5, 0.0, 0.5)
elif (error_model_name == "DP"):
    error_model = PauliErrorModel(0.34, 0.32, 0.34)

size = 2 * d ** 2 - 1
n_node_inputs = 4
n_node_outputs = 4
n_iters=100
n_node_features=500
n_edge_features=500

msg_net_size = 512
msg_net_dropout_p = 0.05
gru_dropout_p = 0.05

enable_osd=False
print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,"n_edge_features: ", n_edge_features ,"enable osd",enable_osd)

fname = f"trained_models/d11_DP_30_500_500_500000_0.15_20000_0.05_0.0001_0.0001_512_0.05_0.05_"
# fname = f"trained_models/d13_from_d11_DP_60_500_500_500000_0.15_20000_0.1_1e-05_0.0001_512_0.05_0.05_"
# fname = f"trained_models/d11_DP_30_500_500_400000_0.15_10000_0.08_0.0001_0.0001_512_0.05_0.05_"
# fname = f"trained_models/d{d}_X_45_500_500_best_"


dist=13
print('trained',d,'\t test',dist,'\t test error',error_model_name )
code = surface_2d.RotatedPlanar2DCode(dist)
# code = create_rotated_surface_codes(dist)
gnn = GNNDecoder(dist=dist, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs, n_iters=n_iters,
                 n_node_features=n_node_features, n_edge_features=n_edge_features,
                 msg_net_size=msg_net_size, msg_net_dropout_p=msg_net_dropout_p, gru_dropout_p=gru_dropout_p)
gnn.to(device)

src, tgt = surface_code_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)
GNNDecoder.device = device
# 0.0006_0.0014 vs 0.0006_0.001
# tools.load_model(gnn, fname + 'gnn.pth 0.0235_0.0002 156', device)
# tools.load_model(gnn, fname + 'gnn.pth 0.0184_0.0026 502', device)
load_model(gnn, fname + 'gnn.pth 0.00135_0.00215 77', device)
# tools.load_model(gnn, fname + 'gnn.pth 0.01035_0.00155 230', device)
perr = []
frac = []

# err_rates = np.array([0.1,0.06,0.05,0.04,0.03])
# err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08])
err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06])
# err_rates = np.array([0.2,0.18,0.16,0.14])
# err_rates = np.array([0.12,0.1,0.08,0.06])
# err_rates = np.array([0.2,0.1,0.06])

nruns=len(err_rates)
le_rates = np.zeros((nruns,5),dtype='float')
len_test_set_org = 10**3
print(f"test set size = {len_test_set_org}/p")
print("i \tp \tLER_X \tLER_Z \tL_tot \ttime_taken")
gnn.eval()
with torch.no_grad():
    for i in range(nruns):

        st = time.time()
        err_rate = err_rates[i]
        # len_test_set = int(len_test_set_org)
        len_test_set = int(len_test_set_org / err_rate)
        testset = adapt_trainset(generate_syndrome_error_volume(code, error_model, p=err_rate, batch_size=len_test_set,for_training=False), code,
                                 num_classes=n_node_inputs, for_training=False)
        testloader = DataLoader(testset, batch_size=128, collate_fn=collate, shuffle=False)
        t1 = time.time()
        # print("data generation",t1 - st)
        osd_decoder = BeliefPropagationOSDDecoder(code, error_model, error_rate=err_rate, osd_order=0, max_bp_iter=0)
        osd_decoder.initialize_decoders()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled = use_amp):
            lerx, lerz, ler_tot = logical_error_rate_osd(gnn, testloader, code, osd_decoder=osd_decoder, enable_osd=enable_osd)
        t2 = time.time()
        # print("ler calculation", t2 - t1)
        fraction_solved = 1  # fraction_of_solved_puzzles(gnn, testloader, code)
        # t3= time.time()ß
        # print("frac",t3-t2)
        test_loss = 0  # compute_accuracy(gnn, testloader, code)
        # t4= time.time()
        # print("test loss",t4-t3)
        le_rates[i, 0] = err_rate
        le_rates[i, 1] = lerx
        le_rates[i, 2] = lerz
        le_rates[i, 3] = t2 - t1
        frac.append(fraction_solved)
        print(i, err_rate, lerx, lerz, ler_tot, test_loss,np.round(t2-t1,2))
        perr.append(err_rate)
        # err_rate = err_rate / 2

# np.save(fname+f'd{dist}_{len_test_set}_{n_iters}itr_osd',le_rates)
