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

from bb_panq_functions import GNNDecoder, collate, fraction_of_solved_puzzles,compute_accuracy, logical_error_rate, \
    surface_code_edges, generate_syndrome_error_volume, adapt_trainset,ler_loss,bb_code,load_model

from codes_q import *
from ldpc import bposd_decoder

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


d=18
#plot_code(code)
#surface_code_edges(code)
error_model_name = "DP"
if(error_model_name == "X"):
    error_model = [1, 0, 0]
elif (error_model_name == "Z"):
    error_model = [0, 0, 1]
elif (error_model_name == "XZ"):
    error_model = [0.5, 0, 0.5]
elif (error_model_name == "DP"):
    error_model = [0.34,0.32,0.34]
    # error_model = PauliErrorModel(0.34, 0.32, 0.34)

# size = 2 * d ** 2 - 1
n_node_inputs = 4
n_node_outputs = 4
n_iters=30
n_node_features=50
n_edge_features=50

msg_net_size = 128
msg_net_dropout_p = 0.05
gru_dropout_p = 0.05

enable_osd=False
print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,"n_edge_features: ", n_edge_features ,"enable osd",enable_osd)

# fname = f"trained_models/BB_n288_k12_d18_from_d18_DP_100_50_50_200000_0.18_5000_0.1_1e-05_0.0001_128_0.05_0.05_"
# fname = f"trained_models/BB_n72_k12_d6_from_d6_DP_30_50_50_20000_0.15_1000_0.06_0.0001_0.001_128_0.05_0.05_"
# fname = f"trained_models/BB_n144_k12_d12_from_d12_DP_30_50_50_40000_0.18_2000_0.08_0.0001_0.0001_128_0.05_0.05_"
# fname = f"trained_models/BB_n144_k12_d12_from_d12_DP_45_50_50_40000_0.2_2000_0.1_0.0001_0.0001_128_0.05_0.05_"
fname = f"trained_models/BB_n288_k12_d18_from_d18_DP_45_50_50_40000_0.2_2000_0.1_0.0001_0.0001_128_0.05_0.05_"
# fname = f"trained_models/BB_n360_k12_d24_from_d24_DP_100_50_50_200000_0.15_5000_0.1_1e-05_0.0001_128_0.05_0.05_"
# fname = f"trained_models/BB_n756_k16_d34_DP_60_50_50_100000_0.15_5000_0.1_0.0001_0.0001_256_0.05_0.05_"
# fname = f"trained_models/d{d}_X_45_500_500_best_"


dist = 24

code = bb_code(dist)
print('trained', d, '\t test', dist, "\tcode name :",code.name)
# code = surface_2d.RotatedPlanar2DCode(dist)
gnn = GNNDecoder(dist=dist, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs, n_iters=n_iters,
                 n_node_features=n_node_features, n_edge_features=n_edge_features,
                 msg_net_size=msg_net_size, msg_net_dropout_p=msg_net_dropout_p, gru_dropout_p=gru_dropout_p)
gnn.to(device)

src, tgt = surface_code_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)

hxperp = torch.FloatTensor(code.hx_perp).to(device)
hzperp = torch.FloatTensor(code.hz_perp).to(device)
GNNDecoder.hxperp = hxperp
GNNDecoder.hzperp = hzperp

GNNDecoder.device = device
# 0.0006_0.0014 vs 0.0006_0.001
# tools.load_model(gnn, fname + 'gnn.pth 0.041_0.015_0.041 27', device)
# tools.load_model(gnn, fname + 'gnn.pth 0.0204_0.0208_0.021 0', device)
# tools.load_model(gnn, fname + 'gnn.pth 0.0295_0.0285_0.0305 256', device)
# tools.load_model(gnn, fname + 'gnn.pth 0.0288_0.0294_0.0294 98', device)
load_model(gnn, fname + 'gnn.pth 0.041_0.043_0.0435 100', device)
# tools.load_model(gnn, fname + 'gnn.pth 0.0362_0.041_0.041 141', device)
perr = []
frac = []

# err_rates = np.array([0.14,0.06,0.05,0.04,0.03])
err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06])
# err_rates = np.array([0.2,0.18,0.16,0.14])
# err_rates = np.array([0.12,0.1,0.08,0.06])
# err_rates = np.array([0.06])
# err_rates = np.array([0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2])
# err_rates = np.array([0.06,0.08,0.1])
# err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06,0.05,0.04])
nruns=len(err_rates)
le_rates = np.zeros((nruns,5),dtype='float')
len_test_set_org = 10**5
print(f"test set size = {len_test_set_org}")
print("i \tp \tLER_X \tLER_Z \tL_tot \ttime_taken")
gnn.eval()
with torch.no_grad():
    for i in range(nruns):
        st = time.time()
        err_rate = err_rates[i]
        len_test_set = int(len_test_set_org)
        # len_test_set = int(len_test_set_org / err_rate)
        testset = adapt_trainset(generate_syndrome_error_volume(code, error_model, p=err_rate, batch_size=len_test_set,for_training=False), code,
                                 num_classes=n_node_inputs, for_training=False)
        testloader = DataLoader(testset, batch_size=128, collate_fn=collate, shuffle=False)
        t1 = time.time()
        # print("data generation",t1 - st)
        x_decoder = bposd_decoder(code.hz,error_rate=err_rate,max_iter=0,bp_method="msl",ms_scaling_factor=0.625,osd_method="osd0",osd_order=0)
        z_decoder = bposd_decoder(code.hx,error_rate=err_rate,max_iter=0,bp_method="msl",ms_scaling_factor=0.625,osd_method="osd0",osd_order=0)
        osd_decoder = (x_decoder,z_decoder)
        # mwpm_decoder = MatchingDecoder(code, error_model, error_rate=err_rate)
        # osd_decoder.initialize_decoders()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            lerx, lerz, ler_tot = logical_error_rate(gnn, testloader, code, osd_decoder, enable_osd=enable_osd)
            # lerx, lerz, ler_tot = 0, 0, 0
            # gnn.eval()
            # device = gnn.device
            # with torch.no_grad():
            #     for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            #         inputs, targets = inputs.to(device), targets.to(device)
            #         src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
            #         # batch_size = inputs.size(0) // size
            #         outputs = gnn(inputs, src_ids, dst_ids)
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
        le_rates[i, 3] = ler_tot
        le_rates[i, 4] = t2 - t1
        frac.append(fraction_solved)
        print(i, err_rate, lerx, lerz, ler_tot, test_loss,np.round(t2-t1,2))
        perr.append(err_rate)
        # err_rate = err_rate / 2

# np.save(fname+f'd{dist}_lerates_{n_iters}itr_osd',le_rates)