from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder, BeliefPropagationOSDDecoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# import tools
from panq_functions import GNNDecoder, collate, fraction_of_solved_puzzles, compute_accuracy, logical_error_rate, \
    surface_code_edges, generate_syndrome_error_volume, adapt_trainset, ler_loss, save_model, load_model

from ldpc.mod2 import *

if torch.cuda.is_available():
    device = torch.device('cuda')
    use_amp = True # to use automatic mixed precision
    amp_data_type = torch.float16
else:
    device = torch.device('cpu')
    use_amp = True
    '''float16 is not supported for cpu use bfloat16 instead'''
    amp_data_type = torch.bfloat16


"""
    Parameters
"""

d = 3
error_model_name = "DP"

if (error_model_name == "X"):
    error_model = PauliErrorModel(1, 0.0, 0)
elif (error_model_name == "Z"):
    error_model = PauliErrorModel(0, 0.0, 1)
elif (error_model_name == "XZ"):
    error_model = PauliErrorModel(0.5, 0.0, 0.5)
elif (error_model_name == "DP"):
    error_model = PauliErrorModel(0.34, 0.32, 0.34)

# list of hyperparameters
n_node_inputs = 4
n_node_outputs = 4
n_iters = 3
n_node_features = 50
n_edge_features = 50
len_test_set = 10
test_err_rate = 0.05

len_train_set = len_test_set * 10
max_train_err_rate = 0.15

lr = 0.0001
weight_decay = 0.0001

msg_net_size = 512
msg_net_dropout_p = 0.05
gru_dropout_p = 0.05

# filename format = f"trained_models/d{d}_X_{n_iters}_{n_node_features}_{n_edge_features}_{len_train_set}_{max_train_err_rate}_{len_test_set}_{test_err_rate}_{lr}_{weight_decay}_{msg_net_size}_{msg_net_dropout_p}_{gru_dropout_p}_"

# print("Distance: ", d, "  Error Model: ",error_model_name, " Err_rate: ", test_err_rate)
print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,
      "n_edge_features: ", n_edge_features)
print("msg_net_size: ", msg_net_size, "msg_net_dropout_p: ", msg_net_dropout_p, "gru_dropout_p: ", gru_dropout_p)
print("learning rate: ", lr, "weight decay: ", weight_decay, "len train set: ", len_train_set, 'max train error rate: ',
      max_train_err_rate, "len test set: ", len_test_set, "test error rate: ", test_err_rate)
# print("loss = msx hxperp + sloss + eloss")

"""
    Create the Surface code
"""
dist = d  # can be different in case of using lower distance trained decoder for larger distance
print('trained', d, '\t retrain', dist)

code = surface_2d.RotatedPlanar2DCode(dist)
gnn = GNNDecoder(dist=dist, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs, n_iters=n_iters,
                 n_node_features=n_node_features, n_edge_features=n_edge_features,
                 msg_net_size=msg_net_size, msg_net_dropout_p=msg_net_dropout_p, gru_dropout_p=gru_dropout_p)
gnn.to(device)

src, tgt = surface_code_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)

hxperp = torch.FloatTensor(nullspace(code.Hx.toarray())).to(device)
hzperp = torch.FloatTensor(nullspace(code.Hz.toarray())).to(device)
GNNDecoder.hxperp = hxperp
GNNDecoder.hzperp = hzperp

GNNDecoder.device = device

total_params = sum(param.numel() for param in gnn.parameters())

fnameload = f"trained_models/d{d}_{error_model_name}_30_500_500_200000_0.15_10000_0.05_gnn.pth 0.0144_0.0044 37"
model_loaded = False
if os.path.isfile(fnameload):
    load_model(gnn, fnameload, device)
    model_loaded = True

# Generate the test data
testset = adapt_trainset(
    generate_syndrome_error_volume(code, error_model=error_model, p=test_err_rate, batch_size=len_test_set,
                                   for_training=False), code,
    num_classes=n_node_inputs, for_training=False)
testloader = DataLoader(testset, batch_size=512, collate_fn=collate, shuffle=False)

"""
    Train
"""
fnamenew = f"trained_models/d{dist}_{error_model_name}_{n_iters}_{n_node_features}_{n_edge_features}_" \
           f"{len_train_set}_{max_train_err_rate}_{len_test_set}_{test_err_rate}_{lr}_" \
           f"{weight_decay}_{msg_net_size}_{msg_net_dropout_p}_{gru_dropout_p}_"
if model_loaded:
    fnamenew = f"trained_models/d{dist}_from_d{d}_{error_model_name}_{n_iters}_{n_node_features}_{n_edge_features}_" \
               f"{len_train_set}_{max_train_err_rate}_{len_test_set}_{test_err_rate}_{lr}_" \
               f"{weight_decay}_{msg_net_size}_{msg_net_dropout_p}_{gru_dropout_p}_"

# def training():
""" optimizer and schedular """
optimizer = optim.AdamW(gnn.parameters(), lr=lr, weight_decay=weight_decay)

exploration_samples = 10 ** 7
lr_reduce_epoch_step = exploration_samples // len_train_set
max_training_data = 10 ** 8
end_training_epoch = max_training_data // len_train_set

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_reduce_epoch_step, gamma=0.1)

""" automatic mixed precision """
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

epochs = 10000
batch_size = 64
criterion = nn.CrossEntropyLoss()

le_rates = np.zeros((epochs, 5), dtype='float')
start_time = time.time()

print("epoch, lr, wd, fract. corr. synd, LER_X, LER_Z, LER_tot, test loss, train loss, train time")
size = 2 * GNNDecoder.dist ** 2 - 1
error_index = GNNDecoder.dist ** 2 - 1

min_test_err_rate = test_err_rate
min_lerz = test_err_rate

""" load training data from files"""

fname_traning_data = ""
# if max_training_data:# <= 10**8:
#     fname_traning_data = f"training_data/d{dist}_X_{10**8}_{max_train_err_rate}.npy"
#     if os.path.isfile(fname_traning_data):
#         print("Training data loaded from file")
#     else:
#         raise Exception("Training data file doesn't exists!")
# else:
#     raise Exception("Required training data too large!!!")

# training_data = np.load(fname_traning_data, mmap_mode="r")

""" generate training data """
trainset = adapt_trainset(
    generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=len_train_set),
    code, num_classes=n_node_inputs)
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=False)

for epoch in range(epochs):
    gnn.train()
    if epoch == end_training_epoch:
        break
    # print(epoch)
    # trainset = np.copy(training_data[len_train_set*epoch:len_train_set*(epoch+1),:])
    # trainset = adapt_trainset(trainset,code,num_classes=n_node_inputs)

    """ generate new training data for each epoch """
    # trainset = adapt_trainset(generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=len_train_set),
    #                           code, num_classes=n_node_inputs)
    # trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    epoch_loss = []
    for i, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
        # optimizer.zero_grad()
        loss = 0

        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            outputs = gnn(inputs, src_ids, dst_ids)
            for j, out in enumerate(outputs):
                eloss = criterion(out.view(-1, size, n_node_inputs)[:, error_index:].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, error_index:].flatten())
                sloss = criterion(out.view(-1, size, n_node_inputs)[:, :error_index].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, :error_index].flatten())

                loss += ler_loss(out, targets, code) + sloss + eloss
            loss /= outputs.shape[0]
            # print(loss)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # print(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss.append(loss.detach())
    epoch_loss = torch.mean(torch.tensor(epoch_loss)).item()
    with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
        fraction_solved = fraction_of_solved_puzzles(gnn, testloader, code)
        test_loss = compute_accuracy(gnn, testloader, code)
        lerx, lerz, ler_tot = logical_error_rate(gnn, testloader, code)

    scheduler.step()  # update lr
    # print(optimizer.param_groups[0]['lr'],optimizer.param_groups[0]["weight_decay"])

    le_rates[epoch, 0] = lerx
    le_rates[epoch, 1] = lerz
    le_rates[epoch, 2] = ler_tot
    le_rates[epoch, 3] = test_loss
    le_rates[epoch, 4] = epoch_loss
    curr_time = time.time() - start_time

    print(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]["weight_decay"], fraction_solved, lerx,
          lerz, ler_tot, test_loss, epoch_loss, curr_time)

    if ler_tot < min_test_err_rate:
        min_test_err_rate = ler_tot
        min_lerz = lerz
        save_model(gnn, fnamenew + f'gnn.pth {lerx}_{lerz} {epoch}', confirm=False)

    if lerz < min_lerz:
        min_lerz = lerz
        save_model(gnn, fnamenew + f'gnn.pth {lerx}_{lerz} {epoch}', confirm=False)

    if epoch % 10 == 0:
        np.save(fnamenew + f'training_lers_and_losses', le_rates)
        # tools.save_model(gnn, fnamenew + f'gnn.pth {lerx}_{lerz} {epoch}', confirm=False)

    if lerz == 0:
        min_lerz = lerz
        save_model(gnn, fnamenew + f'gnn.pth {lerx}_{lerz} {epoch}', confirm=False)
        break
