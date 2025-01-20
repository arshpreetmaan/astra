from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import BeliefPropagationOSDDecoder, MatchingDecoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import sympy
import os


class GNNDecoder(nn.Module):
    dist = None
    surface_code_edges = None
    device = None
    hxperp = None
    hzperp = None
    def __init__(self, dist=3, n_iters=7, n_node_features=10, n_node_inputs=9, n_edge_features=11, n_node_outputs=9, msg_net_size=96, msg_net_dropout_p=0.0, gru_dropout_p=0.0):
        """
        Args:
          n_iters: Number of graph iterations.
          n_node_features: Number of features in the states of each node.
          n_node_inputs: Number of inputs to each graph node (on each graph iteration).
          n_edge_features: Number of features in the messages sent along the edges of the graph (produced
              by the message network).
          n_node_outputs: Number of outputs produced by at each node of the graph.
        """
        super(GNNDecoder, self).__init__()

        GNNDecoder.dist = dist

        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_edge_features = n_edge_features
        self.n_node_outputs = n_node_outputs
        # The states of each graph node are linearly combined to compute the output
        # of the corresponding node at the current iteration.
        self.final_digits = nn.Linear(self.n_node_features, self.n_node_outputs)

        self.msg_net = nn.Sequential(
            nn.Linear(2 * n_node_features, msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            nn.Linear(msg_net_size, msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            nn.Linear(msg_net_size, msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            nn.Linear(msg_net_size, n_edge_features)
        )

        self.gru = nn.GRU(input_size=n_edge_features + n_node_inputs, hidden_size=n_node_features)
        self.gru_drop = nn.Dropout(gru_dropout_p)
        return None

    def forward(self, node_inputs, src_ids, dst_ids):
        """
        Args:
          node_inputs of shape (n_nodes, n_node_inputs): Tensor of inputs to every node of the graph.
          src_ids of shape (n_edges): Indices of source nodes of every edge.
          dst_ids of shape (n_edges): Indices of destination nodes of every edge.

        Returns:
          outputs of shape (n_iters, n_nodes, n_node_outputs): Outputs of all the nodes at every iteration of the
              graph neural network.
        """
        # YOUR CODE HERE
        device = node_inputs.device
        node_states = torch.zeros(node_inputs.shape[0], self.n_node_features,device=device)
        # node_states[:, :node_inputs.shape[1]] = node_inputs
        # node_states = torch.cat((node_states, node_inputs), dim=1)
        outputs_tensor = torch.zeros(self.n_iters, node_inputs.shape[0], self.n_node_outputs,device=device)

        for i in range(self.n_iters):
            msg_in = torch.cat((node_states[src_ids], node_states[dst_ids]), dim=1)
            messages = self.msg_net(msg_in)

            agg_msg = torch.zeros(node_inputs.shape[0], self.n_edge_features, device=device, dtype=messages.dtype)
            agg_msg.index_add_(dim=0, index=dst_ids, source=messages)
            gru_inputs = torch.cat((agg_msg, node_inputs), dim=1)

            output, node_states = self.gru(gru_inputs.view(1, node_inputs.shape[0], -1),
                                           node_states.view(1, node_inputs.shape[0], -1))
            node_states = node_states.squeeze(0)
            outputs_tensor[i] = self.final_digits(node_states)
            node_states = self.gru_drop(node_states)  #kind of stab inactivation

        return outputs_tensor

# ## Custom data loader
#
# We first create a custom data loader to process a mini-batch of graphs (in parallel) to compute the derivatives wrt
# the parameters of the graph neural network. To do that, **we transform a mini-batch of graphs to one large graph
# without interconnecting edges between the subgraphs corresponding to individual examples in the mini-batch.**
# We do this using a custom collate function.


def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (inputs, targets),
          inputs of shape (n_nodes, n_node_inputs): Inputs to each node in the graph. Inputs are one-hot coded digits.
          A missing digit is encoded with all zeros. n_nodes= nodes in the tanner graph
          targets of shape (n_nodes): A LongTensor of targets (correct digits of tanner graph).

    Returns:
      inputs of shape (batch_size*n_nodes, n_node_inputs): Inputs to each node in the graph. Inputs are one-hot coded digits
          for syndromes/errors similar to in the sudoku puzzle. A missing digit is encoded with all zeros.
      targets of shape (batch_size*n_nodes): A LongTensor of targets (correct digits of tanner graph).
      src_ids of shape (batch_size*nodes in the tanner graph): LongTensor of source node ids for each edge in the large graph.
      dst_ids of shape (batch_size*nodes in the tanner graph): LongTensor of destination node ids for each edge in the large graph.
    """
    # YOUR CODE HERE
    (inp, target) = list_of_samples[0]
    if GNNDecoder.surface_code_edges is None:
        raise Exception
    og_src_ids, og_tgt_ids = GNNDecoder.surface_code_edges

    all_inputs = inp.clone().detach()
    all_targets = target.clone().detach()
    all_src_ids = og_src_ids.clone().detach()
    all_dst_ids = og_tgt_ids.clone().detach()

    if GNNDecoder.dist is None:
        raise Exception
    add = 2 * (GNNDecoder.dist) ** 2 - 1

    for (inp, target) in list_of_samples[1:]:
        og_src_ids = torch.add(og_src_ids, add)
        og_tgt_ids = torch.add(og_tgt_ids, add)
        all_inputs = torch.cat((all_inputs, inp))
        all_targets = torch.cat((all_targets, target))
        all_src_ids = torch.cat((all_src_ids, og_src_ids))
        all_dst_ids = torch.cat((all_dst_ids, og_tgt_ids))

    return all_inputs, torch.LongTensor(all_targets), torch.LongTensor(all_src_ids), torch.LongTensor(all_dst_ids)


def plot_code(code):
    qcord = code.qubit_coordinates
    scord = code.stabilizer_coordinates
    x1, y1 = zip(*qcord)
    d = plt.scatter(x1, y1, color="k")
    ztype = []
    xtype = []
    for i in scord:
        if code.stabilizer_type(i) == "vertex":
            ztype.append(i)
        else:
            xtype.append(i)

    x2, y2 = zip(*ztype)
    z = plt.scatter(x2, y2, color="g")
    x3, y3 = zip(*xtype)
    x = plt.scatter(x3, y3, color="r")
    plt.legend((d, z, x), ("data", "Z type", "X type"))
    # plt.savefig("d5_panqec.png")
    # plt.show()


def surface_code_edges(code):
    # graph=[(i, j) for i, j in zip(*code.stabilizer_matrix.nonzero())]
    src_ids, dst_ids = code.stabilizer_matrix.nonzero()  # syndrome,data qubit
    l = int(len(dst_ids) / 2)
    dst_ids = dst_ids - 1
    dst_ids[l:] = dst_ids[l:] + code.d ** 2

    # for only Z stab = detect X
    # dst_ids = dst_ids[:l]
    # dst_ids = np.append(dst_ids, src_ids[l:])

    temp = src_ids
    src_ids = np.append(src_ids, dst_ids)
    dst_ids = np.append(dst_ids, temp)

    # G = nx.Graph()
    G = nx.DiGraph()
    for (s, t) in zip(src_ids, dst_ids):
        G.add_edge(s, t)

    # color_map = ['red' if node < code.d ** 2 - 1 else 'green' for node in G]
    # nx.draw(G, node_color=color_map, with_labels=True)
    # # plt.savefig("trained_models/d3_panqec.png")
    # # plt.show()
    return src_ids, dst_ids


def generate_syndrome_error_volume(code, error_model, p, batch_size, for_training=True):
    d = code.d
    size = 2 * d ** 2 - 1
    syndrome_error_volume = np.zeros((batch_size, size), dtype='uint8')
    starttime = time.time()
    # bpdec = decoder(code, error_model, error_rate=0.1, osd_order=0)
    # decoder = MatchingDecoder
    # mwpm = decoder(code, error_model, error_rate=p)
    if not for_training:
        error = np.zeros((batch_size, 2 * d ** 2), dtype='uint8')
        for i in range(batch_size):
            error[i] = error_model.generate(code, p)
        syndrome = code.measure_syndrome(error).T

        errorxz= (error[:,:d**2] + 2*error[:,d**2:]) #% 3
        # syndromexz = syndrome
        syndromexz = np.append(syndrome[:, :(d ** 2 - 1) // 2], 2*syndrome[:, (d ** 2 - 1) // 2:], axis=1)
        syndrome_error_volume = np.append(syndromexz, errorxz, axis=1)


    if for_training:
        # syndrome_error_volume = np.zeros((batch_size, size), dtype=int)
        rng = np.random.default_rng(1)
        error = np.zeros((batch_size, 2 * d ** 2), dtype='uint8')
        for i in range(batch_size):
            pr = p * rng.random()
            error[i] = error_model.generate(code, pr)
        syndrome = code.measure_syndrome(error).T
        #mwpm
        # pred_error = np.zeros((batch_size, 2 * d ** 2), dtype='uint8')
        pred_error = error
        ''' to use mwpm prediction as error data'''
        # for i in range(batch_size):
        #     pred_error[i]=mwpm.decode(syndrome[i])
        perror = (pred_error[:,:d**2] + 2*pred_error[:,d**2:]) #% 3
        # syndromexz = syndrome
        syndromexz = np.append(syndrome[:, :(d ** 2 - 1) // 2], 2*syndrome[:, (d ** 2 - 1) // 2:], axis=1)
        # syndrome_error_volume = np.append(syndrome, pred_error[:, :d ** 2], axis=1)
        syndrome_error_volume = np.append(syndromexz, perror, axis=1)

        # syndrome_error_volume = np.append(syndrome, error[:, :d ** 2], axis=1)


    #print(time.time() - starttime)
    return syndrome_error_volume


def adapt_trainset(batch, code, num_classes=2, for_training=True):
    # batch is now np array  [syndrome error]
    # if for_training:
    # batch = np.unique(batch, axis=0)
    #print(f"{len(batch)} unique in train set")
    # st = time.time()

    error_index = code.d ** 2 - 1
    batch_np = batch
    targets_all = torch.LongTensor(batch_np)
    inputs_all = targets_all[:, :error_index]
    inputs_all = nn.functional.one_hot(inputs_all, num_classes)
    zeros = torch.zeros((inputs_all.shape[0], code.d ** 2, num_classes)).long()
    # zeros = torch.zeros((inputs_all.shape[0]).long()
    inputs_all = torch.cat((inputs_all, zeros), dim=1)
    # inputs_all = nn.functional.one_hot(inputs_all, num_classes)
    trainset = list(zip(inputs_all, targets_all))
    #print("adapt dataset",time.time()-st)
    return trainset


def fraction_of_solved_puzzles(gnn, testloader, code):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    gnn.eval()
    device = gnn.device
    with torch.no_grad():
        n_test = 0
        n_test_solved = 0
        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
            # inputs is [n_nodes, 9*9, 9]
            # targets is [n_nodes]
            batch_size = inputs.size(0) // size
            # inputs, targets = inputs.to(device), targets.to(device)
            # src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
            encoding = outputs.shape[-1]

            solution = outputs.view(gnn.n_iters, batch_size, size, -1)
            final_solution = solution[-1].argmax(dim=2)
            # solved = (final_solution.view(-1, size) == targets.view(batch_size, size)).all(dim=1)
            solved = ((final_solution.view(-1, size))[:, :error_index] == (targets.view(batch_size, size))[:,
                                                                          :error_index]).all(dim=1)
            n_test += solved.size(0)
            n_test_solved += solved.sum().item()

    return n_test_solved / n_test

def init_log_probs_of_decoder(decoder, my_log_probs):
    #print("old ", decoder.log_prob_ratios)

    for i in range(len(decoder.log_prob_ratios)):
        decoder.set_log_prob(i, my_log_probs[i])

    #print("new ", decoder.log_prob_ratios)

def logical_error_rate(gnn, testloader, code, enable_osd=False, osd_decoder=None):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    gnn.eval()
    device = gnn.device
    with torch.no_grad():
        n_test = 0
        n_l_error = 0
        n_codespace_error = 0
        n_total_ler = 0

        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
            # batch_size = inputs.size(0) // size
            outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, node_outputs]
            encoding = outputs.shape[-1]
            if encoding == 1:
                solution = outputs.view(gnn.n_iters, -1, size)
                final_solution = torch.heaviside(solution[-1, :, error_index:], torch.tensor([1.0]))
            else:
                if enable_osd:
                    final_solution = osd(outputs,targets,code,osd_decoder)
                    # solution = outputs.view(gnn.n_iters, -1, size, encoding)
                    # final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()

                else:
                    solution = outputs.view(gnn.n_iters, -1, size, encoding)
                    final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()
            batch_size = final_solution.shape[0]

            final_targets = targets.view(batch_size, size)[:, error_index:].cpu()
            final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                             final_targets, 0) // 3
            final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3,
                                                                                                  final_targets, 0) // 3

            final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                                final_solution, 0) // 3
            final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(
                final_solution == 3, final_solution, 0) // 3

            final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
            final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)


            rf = (final_targets + final_solution) % 2
            # l = np.any(code.logical_errors(rf) != 0, axis=1)
            # n_l_error += l.sum()

            ''' codespace error '''
            ms = code.measure_syndrome(rf).T
            # n_codespace_error += batch_size - np.all(ms == 0, axis=1).sum()
            mse = np.any(ms, axis=1)
            n_codespace_error += mse.sum()

            ''' logical operator error '''
            # msi = np.where(mse == 0)
            # l = np.any(code.logical_errors(rf[msi]) != 0, axis=1)
            l = np.any(code.logical_errors(rf) != 0, axis=1)
            n_l_error += l.sum()

            ''' total logical error'''
            n_total_ler += np.logical_or(l, mse).sum()
            # n_total_ler += mse.sum() + l.sum()
            n_test += batch_size

        return (n_l_error / n_test), (n_codespace_error / n_test), n_total_ler/n_test

def osd(outputs, targets, code, osd_decoder=None):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    # n_iters=out.shape[0]
    encoding = outputs.shape[-1]

    solution = outputs.view(outputs.shape[0], -1, size, encoding)
    # final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()
    final_solution = nn.functional.softmax(solution[-1,:, error_index:], dim=2).cpu()

    fllrx = -torch.log((final_solution[:, :, 1]+final_solution[:,:,3]) / final_solution[:, :, 0])   #works better
    fllrz = -torch.log((final_solution[:, :, 2]+final_solution[:,:,3]) / final_solution[:, :, 0])
    # fllrx = -torch.log((final_solution[:, :, 1] + final_solution[:, :, 3]) /
    #             (1 - (final_solution[:, :, 1] + final_solution[:, :, 3])))
    # fllrz = -torch.log((final_solution[:, :, 1] + final_solution[:, :, 3]) /
    #                   (1 - (final_solution[:, :, 1] + final_solution[:, :, 3])))
    # final_solution = nn.functional.sigmoid(solution[-1, :, error_index:]).cpu()
    # fllr = torch.log(final_solution[:, :, 1] / final_solution[:, :, 0])  # negative llr
    # final_solution_sig = nn.functional.sigmoid(solution[-1, :, error_index:]).cpu()
    # fnllr_sig = torch.log(final_solution_sig[:, :, 1] / final_solution_sig[:, :, 0])

    final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()
    osd_out = final_solution
    final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                        final_solution, 0) // 3
    final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(
        final_solution == 3, final_solution, 0) // 3
    batch_size = final_solution.shape[0]
    # hx = code.Hx.toarray()
    # hz = code.Hz.toarray()

    # final_targets = targets.view(batch_size, size).cpu()
    final_targets = targets.view(batch_size, size)[:, error_index:].cpu()
    final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                     final_targets, 0) // 3
    final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3,
                                                                                          final_targets, 0) // 3

    final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
    final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)

    final_syn = np.array((targets.view(batch_size, size)[:, :error_index]).cpu())
    final_syn = np.append(final_syn[:,:error_index//2],final_syn[:,error_index//2:]//2 , axis = 1)
    # og = np.argsort(fllr[3])[::-1]
    rf = (final_targets + final_solution) % 2
    ms = code.measure_syndrome(rf).T
    nonzero_syn_id = np.nonzero(1 - np.all(ms == 0, axis=1))[0]

    #osd_out = np.array(final_solution)
    #init_log_probs_of_decoder(osd_decoder.x_decoder, list(range(25)))
    decoder = MatchingDecoder
    for i in nonzero_syn_id:
        dec = decoder(code,osd_decoder.error_model,osd_decoder.error_rate,weights=(fllrx[i],fllrz[i]))
        # init_log_probs_of_decoder(osd_decoder.x_decoder, fllrx[i])
        # init_log_probs_of_decoder(osd_decoder.z_decoder, fllrz[i])
        # init_log_probs_of_decoder(osd_decoder.x_decoder, fnllr_sig[i])

        osd_err=(dec.decode(final_syn[i])).astype("int64")
        # osd_out[i] = osd_err[:error_index+1]
        osd_out[i] = torch.LongTensor((osd_err[:code.d ** 2] + 2 * osd_err[code.d ** 2:]))

    return osd_out
def ler_loss(out, targets, code):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    device = out.device
    # n_iters=out.shape[0]
    encoding = out.shape[-1]

    # outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
    solution = (out.view(-1, size, encoding))
    zeros = torch.zeros(code.d ** 2,device=device)
    # final_solution = solution[:, : syndrome_index, :].argmax(dim=2)
    # final_solution = nn.functional.softmax(solution[:, : syndrome_index, :], dim=2)[:,:,1]
    final_solution = nn.functional.softmax(solution[:, error_index:, :], dim=2)
    # final_solution =  nn.functional.sigmoid(solution[:, error_index:, :])
    batch_size = final_solution.shape[0]
    msx = 0
    msz = 0
    # ms = 0
    hxperp = GNNDecoder.hxperp
    hzperp = GNNDecoder.hzperp

    final_targets = targets.view(batch_size, size)[:, error_index:]
    final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3, final_targets, 0) // 3
    final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3, final_targets, 0) // 3

    # rx = final_targets + final_solution[:,:,1]
    # rfx = (torch.abs(torch.sin(torch.pi * rx / 2)))
    # # zer = torch.zeros(batch_size, code.d ** 2)
    # # rf = torch.cat((r, zer), dim=1)
    # msx_batch = torch.mean(torch.abs(torch.sin(torch.pi * ((rfx @ hxperp.T)) / 2)),dim=1)
    # msx = msx_batch.sum()

    rx = final_targetsx + final_solution[:, :, 1] + final_solution[:, :, 3]
    rfx = (torch.abs(torch.sin(torch.pi * rx / 2)))
    msx_batch = torch.mean(torch.abs(torch.sin(torch.pi * ((rfx @ hxperp.T)) / 2)), dim=1)
    msx = msx_batch.sum()

    rz = final_targetsz + final_solution[:, :, 2] + final_solution[:, :, 3]
    rfz = (torch.abs(torch.sin(torch.pi * rz / 2)))
    msz_batch = torch.mean(torch.abs(torch.sin(torch.pi * ((rfz @ hzperp.T)) / 2)), dim=1)
    msz = msz_batch.sum()

    n_l_error = msx +msz

    return n_l_error / batch_size

def ler_loss_itr(outputs, targets, code):
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    device = outputs.device
    # n_iters=out.shape[0]
    encoding = outputs.shape[-1]
    n_iters = outputs.shape[0]

    # outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
    solution = outputs.view(n_iters, -1, size, encoding)
    final_solution = nn.functional.softmax(solution[:, :, error_index:,:],dim=-1)
    batch_size = final_solution.shape[1]

    final_targets = targets.view(batch_size, size)[:, error_index:]
    final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                     final_targets, 0) // 3
    final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(
        final_targets == 3,
        final_targets, 0) // 3


    final_targetsx = final_targetsx.unsqueeze(0).repeat(n_iters, 1, 1)
    final_targetsz = final_targetsz.unsqueeze(0).repeat(n_iters, 1, 1)

    hxperp = GNNDecoder.hxperp
    hzperp = GNNDecoder.hzperp

    rx = final_targetsx + final_solution[:,: , :, 1] + final_solution[:,:, :, 3]
    rfx = (torch.abs(torch.sin(torch.pi * rx / 2)))
    msx_itrs = torch.sum(torch.mean(torch.abs(torch.sin(torch.pi * ((rfx @ hxperp.T)) / 2)), dim=-1),dim=-1)
    # msx = msx_batch.sum()

    rz = final_targetsz + final_solution[:,:, :, 2] + final_solution[:,:, :, 3]
    rfz = (torch.abs(torch.sin(torch.pi * rz / 2)))
    msz_itrs = torch.sum(torch.mean(torch.abs(torch.sin(torch.pi * ((rfz @ hzperp.T)) / 2)), dim=-1),dim=-1)
    # msz = msz_batch.sum()

    n_l_error = (msx_itrs + msz_itrs).sum()
    return n_l_error / batch_size #+ loss

def compute_accuracy(gnn, testloader, code):
    gnn.eval()
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    # device = torch.device('cpu')
    device = gnn.device
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        losses = []
        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)
            encoding = outputs.shape[-1]
            loss = 0
            # loss_itr_l1=[]
            # loss_itr_l2 = []
            for out in outputs:
                # loss += criterion(out, targets)
                eloss = criterion(out.view(-1, size, encoding)[:, error_index:].reshape(-1, encoding),
                                  targets.view(-1, size)[:, error_index:].flatten())
                sloss = criterion(out.view(-1, size, encoding)[:, :error_index].reshape(-1, encoding),
                                  targets.view(-1, size)[:, :error_index].flatten())
                l1 = ler_loss(out, targets, code)
                l2 = sloss + eloss
                loss += l1+l2
                # loss_itr_l1.append(l1)
                # loss_itr_l2.append(l2)
            loss /= outputs.shape[0]

            losses.append(loss.detach())

        losses = torch.mean(torch.tensor(losses)).item()
    return losses

def calc_test_losses(gnn, testloader, code):
    gnn.eval()
    size = 2 * code.d ** 2 - 1
    error_index = code.d ** 2 - 1
    # device = torch.device('cpu')
    device = gnn.device
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        n_test_frac = 0
        n_test_solved = 0
        losses = []
        n_test = 0
        n_l_error = 0
        n_codespace_error = 0
        n_total_ler = 0
        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)
            encoding = outputs.shape[-1]
            batch_size = inputs.size(0) // size

            """ fraction solved"""
            if encoding == 1:
                solution = outputs.view(gnn.n_iters, -1, size)
                final_solution = torch.heaviside(solution[-1], torch.tensor([1.0]))

            else:
                solution = outputs.view(gnn.n_iters, batch_size, size, -1)
                final_solution = solution[-1].argmax(dim=2)
            # solved = (final_solution.view(-1, size) == targets.view(batch_size, size)).all(dim=1)
            solved = ((final_solution.view(-1, size))[:, :error_index] == (targets.view(batch_size, size))[:,
                                                                          :error_index]).all(dim=1)
            n_test_frac += solved.size(0)
            n_test_solved += solved.sum().item()

            """ compute accuracy"""
            loss = 0
            ler_loss_all_itr = ler_loss_itr(outputs, targets, code)

            for j, out in enumerate(outputs):
                celoss = criterion(out, targets)
                loss += celoss
                # torch.cuda.nvtx.range_pop()
            loss += ler_loss_all_itr
            loss /= outputs.shape[0]
            losses.append(loss.detach())

            """ logical error rate"""
            solution = outputs.view(gnn.n_iters, -1, size, encoding)
            final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()
            batch_size = final_solution.shape[0]

            final_targets = targets.view(batch_size, size)[:, error_index:].cpu()
            final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                             final_targets, 0) // 3
            final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3,
                                                                                                  final_targets, 0) // 3

            final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                                final_solution, 0) // 3
            final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(final_solution == 3,
                                                                                                     final_solution, 0) // 3

            final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
            final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)

            rf = (final_targets + final_solution) % 2
            l = np.any(code.logical_errors(rf) != 0, axis=1)
            n_l_error += l.sum()

            ms = code.measure_syndrome(rf).T
            # n_codespace_error += batch_size - np.all(ms == 0, axis=1).sum()
            mse = np.any(ms, axis=1)
            n_codespace_error += mse.sum()

            n_total_ler += np.logical_or(l, mse).sum()
            n_test += batch_size



    fraction_solved = n_test_solved / n_test_frac
    losses = torch.mean(torch.tensor(losses)).item()
    lerx = (n_l_error / n_test)
    lerz = (n_codespace_error / n_test)
    ler_tot = (n_total_ler / n_test)
    return fraction_solved,losses,lerx,lerz,ler_tot

def save_model(model, filename, confirm=True):
    if confirm:
        try:
            save = input('Do you want to save the model (type yes to confirm)? ').lower()
            if save != 'yes':
                print('Model not saved.')
                return
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')

    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    filesize = os.path.getsize(filename)
    if filesize > 30000000:
        raise 'The file size should be smaller than 30Mb. Please try to reduce the number of model parameters.'
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()