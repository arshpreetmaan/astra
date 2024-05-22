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

from panq_functions import GNNDecoder, collate, fraction_of_solved_puzzles,compute_accuracy, logical_error_rate, \
    surface_code_edges, generate_syndrome_error_volume, adapt_trainset,ler_loss, load_model


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    use_amp = True  #to use automatic mixed precision
    amp_data_type = torch.float16
else:
    device = torch.device('cpu')
    use_amp = False
    '''float16 is not supported for cpu use bfloat16 instead'''
    amp_data_type = torch.bfloat16

d=11  # trained distance
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
n_iters=30    # flexibility to change this
n_node_features=500
n_edge_features=500

msg_net_size = 512
msg_net_dropout_p = 0.05
gru_dropout_p = 0.05

""" to use the second stage decoder initialised with gnn's output llrs """
enable_osd=False
print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,"n_edge_features: ", n_edge_features ,"enable mwpm similar to osd",enable_osd)

fname = f"trained_models/d11_DP_30_500_500_500000_0.15_20000_0.05_0.0001_0.0001_512_0.05_0.05_"

dist=3   # distance to test on
print('trained',d,'\t test',dist,'\t test error',error_model_name )
code = surface_2d.RotatedPlanar2DCode(dist)
gnn = GNNDecoder(dist=dist, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs, n_iters=n_iters,
                 n_node_features=n_node_features, n_edge_features=n_edge_features,
                 msg_net_size=msg_net_size, msg_net_dropout_p=msg_net_dropout_p, gru_dropout_p=gru_dropout_p)
gnn.to(device)

src, tgt = surface_code_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)
GNNDecoder.device = device


#load_model(gnn, fname + 'gnn.pth 0.0115_0.0107_0.0164 4', device)
load_model(gnn, fname + 'gnn.pth 0.00135_0.00215 77', device)
#load_model(gnn, fname + 'gnn.pth 0.0249_0.0267_0.0372 30', device)
#load_model(gnn, fname + 'gnn.pth 0.02665_0.0154_0.03435 36', device)

# err_rates = np.array([0.1,0.06,0.05,0.04,0.03])
err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06])
# err_rates = np.array([0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2])
nruns=len(err_rates)
le_rates = np.zeros((nruns,5),dtype='float')
len_test_set = 10**2
gnn.eval()
with torch.no_grad():
    for i in range(nruns):
        st = time.time()
        err_rate = err_rates[i]
        testset = adapt_trainset(generate_syndrome_error_volume(code, error_model, p=err_rate, batch_size=len_test_set,for_training=False), code,
                                 num_classes=n_node_inputs, for_training=False)
        testloader = DataLoader(testset, batch_size=256, collate_fn=collate, shuffle=False)
        t1 = time.time()
        print("data generation",t1 - st)
        # osd_decoder = BeliefPropagationOSDDecoder(code, error_model, error_rate=err_rate, osd_order=0, max_bp_iter=0)
        mwpm_decoder = MatchingDecoder(code, error_model, error_rate=err_rate)
        # osd_decoder.initialize_decoders()
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled = use_amp):
            lerx, lerz, ler_tot = logical_error_rate(gnn, testloader, code, enable_osd=enable_osd, osd_decoder=mwpm_decoder)
        t2 = time.time()
        print("ler calculation", t2 - t1)
        fraction_solved = 1  # fraction_of_solved_puzzles(gnn, testloader, code)
        # t3= time.time()
        # print("frac",t3-t2)
        test_loss = 0  # compute_accuracy(gnn, testloader, code)
        # t4= time.time()
        # print("test loss",t4-t3)
        le_rates[i, 0] = err_rate
        le_rates[i, 1] = lerx
        le_rates[i, 2] = lerz
        le_rates[i, 3] = ler_tot
        le_rates[i, 4] = t2 - t1
        print(i, err_rate, fraction_solved, lerx, lerz, ler_tot, test_loss)

# np.save(fname+f'd{dist}_lerates_210_mil_{n_iters}itr2',le_rates)
# np.save(fname+f'd{dist}_lerates_35_{len_test_set}_{n_iters}itr_mwpm',le_rates)
# np.save(fname+f'd{dist}_lerates_119_mil_{n_iters}itr_osd',le_rates)