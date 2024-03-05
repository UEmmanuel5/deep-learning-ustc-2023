# %% [markdown]
# preprocessing citeseer

# %%
#preprocessing citeseer

content_path = "citeseer/citeseer.content"
cites_path = "citeseer/citeseer.cites"
new_content_path = "citeseer_new/citeseer.content.new"
new_cites_path = "citeseer_new/citeseer.cites.new"
fr_content = open(content_path, "r")
fr_cites = open(cites_path, "r")
fw_content = open(new_content_path, "w")
fw_cites = open(new_cites_path, "w")

name_dict = dict()
ban_set = set()
content_lines = fr_content.readlines()
cites_lines = fr_cites.readlines()

import random

for line in content_lines:
    line = line.strip().split('	')
    name = line[0]
    try:
        x = int(name)
        if x in ban_set:
            print(x)
        ban_set.add(x)
    except ValueError:
        pass

for line in content_lines:
    line = line.strip().split('	')
    name = line[0]
    try:
        int(name)
        continue
    except ValueError:
        pass
    
    encode = random.randint(0, 200000)
    while encode in ban_set:
        encode = random.randint(0, 200000)
    ban_set.add(encode)
    name_dict[name] = str(encode)

for line in content_lines:
    temp = line
    line = line.strip().split('	')
    name = line[0]
    try:
        int(name)
        fw_content.write(temp)
    except ValueError:
        temp = temp.replace(name, name_dict[name])
        fw_content.write(temp)

for line in cites_lines:
    line = line.strip().split('	')
    n1 = line[0]
    n2 = line[1]
    try:
        int(n1)
        if n1 not in ban_set:
            continue
        n1 = str(n1)
    except ValueError:
        if n1 not in name_dict:
            continue
        n1 = name_dict[n1]
    
    try:
        int(n2)
        if n2 not in ban_set:
            continue
        n2 = str(n2)
    except ValueError:
        if n2 not in name_dict:
            continue
        n2 = name_dict[n2]

    fw_cites.write(n1+'	'+n2+'\n')
 
    
    



# %% [markdown]
# utils

# %%
#utils

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import negative_sampling
import json

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset, task, self_loop):
    
    if dataset == 'cora':
        path = "cora/"
        dataset = "cora"
    elif dataset == 'citeseer':
        #path = "../datasets/citeseer_new/"
        path = "citeseer_new/"
        dataset = "citeseer"
    
    #print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    np.random.shuffle(idx_features_labels)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    
    temp1 = map(idx_map.get, edges_unordered.flatten())
    temp2 = list(temp1)
    x = list(edges_unordered.flatten())
    print(x[462])
    for i in range(len(temp2)):
        elem = temp2[i]
        try:
            elem = int(elem)
        except TypeError:
            print(i)

    edges = np.array(temp2, dtype=np.int32).reshape(edges_unordered.shape)
    '''
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    '''
    #print('You are currently running {} task on {} dataset...'.format(task, dataset))
    if task == 'linkpred':
        edge_num = edges.shape[0]
        shuffled_ids = np.random.permutation(edge_num)
        test_set_size = int(edge_num * 0.15)
        val_set_size = int(edge_num * 0.15)
        test_ids = shuffled_ids[ : test_set_size]
        val_ids = shuffled_ids[test_set_size : test_set_size + val_set_size]
        train_ids = shuffled_ids[test_set_size + val_set_size : ]

        train_pos_edges = torch.tensor(edges[train_ids], dtype=int)
        val_pos_edges = torch.tensor(edges[val_ids], dtype=int)
        test_pos_edges = torch.tensor(edges[test_ids], dtype=int)

        train_pos_edges = torch.transpose(train_pos_edges, 1, 0)
        # shape = [2, train_pos_edge_num]
        val_pos_edges = torch.transpose(val_pos_edges, 1, 0)
        test_pos_edges = torch.transpose(test_pos_edges, 1, 0)

        def negative_sample(pos_edges, nodes_num):
            '''
            pos_edges = [[src_1,...],
                        [dst_1,...]]
            '''
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=nodes_num,
                num_neg_samples=pos_edges.shape[1],
                method='sparse'
            )
            edges = torch.cat((pos_edges, neg_edges), dim=-1)
            '''
            edges = [[src_1,src_2,...,src_m],
                    [dst_1,dst_2,...,dst_m]]
            shape = [2, 2*train_edge_num]
            '''
            edges_label = torch.cat((
                torch.ones(pos_edges.shape[1]),
                torch.zeros(neg_edges.shape[1])
            ),dim=0)
            # size = [2*train_edge_num]
            return edges, edges_label
        
        train_edges, train_label = negative_sample(train_pos_edges, idx.shape[0])
        val_edges, val_label = negative_sample(val_pos_edges, idx.shape[0])
        test_edges, test_label = negative_sample(test_pos_edges, idx.shape[0])
        
        adj = sp.coo_matrix((np.ones(train_pos_edges.shape[1]), (train_pos_edges[0], train_pos_edges[1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
    
        if self_loop == True:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        train_edges = train_edges.tolist()
        val_edges = val_edges.tolist()
        test_edges = test_edges.tolist()
        train_label = train_label.type(torch.float)
        val_label = val_label.type(torch.float)
        test_label = test_label.type(torch.float)

        return adj, features, train_edges, val_edges, test_edges, \
                    train_label, val_label, test_label

    elif task == 'nodecls':
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
    
        if self_loop == True:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)
        
        # split train || val || test
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test
    
    else:
        raise Exception("hyper-parameter `task` belongs to \{'nodecls', 'linkpred'\}.")


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
        
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_ppi_data(task='nodecls', self_loop=True):
    path = "ppi/"
    #print('Loading PPI dataset...')
    feature_file = path + "ppi-feats.npy"
    label_file = path + "ppi-class_map.json"
    edge_file = path + "ppi-walks.txt"
    graph_file = path + "ppi-G.json"

    #print('Uploading features ...')
    features = np.load(feature_file) # shape = (56944, 50)
    features = sp.csr_matrix(features, dtype=np.float32)
    #print('Uploading labels...')
    fr_label = open(label_file, "r")
    label_dict = json.load(fr_label)
    proc_label_dict = dict()
    for key in label_dict:
        proc_label_dict[int(key)] = list(label_dict[key])
    _labels = sorted(proc_label_dict.items(), key=lambda d: d[0])
    labels = list()
    for item in _labels:
        _, x = item
        labels.append(x)
    labels = np.array(labels, dtype=np.int32)
    print('Uploading graph...')
    fr_graph = open(graph_file, "r")
    graph_dict = json.load(fr_graph)
    nodes = graph_dict["nodes"]
    links = graph_dict["links"]
    #print('Generating edges')
    edges = [[links[i]["source"], links[i]["target"]] for i in range(len(links))]
    edges = np.array(edges, dtype=np.int32)
    #print('Generating nodes')
    idx = list()
    idx_train = list()
    idx_val = list()
    idx_test = list()
    for i in range(len(nodes)):
        idx.append(nodes[i]["id"])
        if nodes[i]["test"] == True:
            idx_test.append(nodes[i]["id"])
        elif nodes[i]["val"] == True:
            idx_val.append(nodes[i]["id"])
        else:
            idx_train.append(nodes[i]["id"])
    idx = np.array(idx, dtype=np.int32)

    #print('You are currently running {} task on PPI dataset...'.format(task))
    if task == 'linkpred':
        edge_num = edges.shape[0]
        shuffled_ids = np.random.permutation(edge_num)
        test_set_size = int(edge_num * 0.15)
        val_set_size = int(edge_num * 0.15)
        test_ids = shuffled_ids[ : test_set_size]
        val_ids = shuffled_ids[test_set_size : test_set_size + val_set_size]
        train_ids = shuffled_ids[test_set_size + val_set_size : ]

        train_pos_edges = torch.tensor(edges[train_ids], dtype=int)
        val_pos_edges = torch.tensor(edges[val_ids], dtype=int)
        test_pos_edges = torch.tensor(edges[test_ids], dtype=int)

        train_pos_edges = torch.transpose(train_pos_edges, 1, 0)
        # shape = [2, train_pos_edge_num]
        val_pos_edges = torch.transpose(val_pos_edges, 1, 0)
        test_pos_edges = torch.transpose(test_pos_edges, 1, 0)

        def negative_sample(pos_edges, nodes_num):
            '''
            pos_edges = [[src_1,...],
                        [dst_1,...]]
            '''
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=nodes_num,
                num_neg_samples=pos_edges.shape[1],
                method='sparse'
            )
            edges = torch.cat((pos_edges, neg_edges), dim=-1)
            '''
            edges = [[src_1,src_2,...,src_m],
                    [dst_1,dst_2,...,dst_m]]
            shape = [2, 2*train_edge_num]
            '''
            edges_label = torch.cat((
                torch.ones(pos_edges.shape[1]),
                torch.zeros(neg_edges.shape[1])
            ),dim=0)
            # size = [2*train_edge_num]
            return edges, edges_label
        
        train_edges, train_label = negative_sample(train_pos_edges, idx.shape[0])
        val_edges, val_label = negative_sample(val_pos_edges, idx.shape[0])
        test_edges, test_label = negative_sample(test_pos_edges, idx.shape[0])
        
        adj = sp.coo_matrix((np.ones(train_pos_edges.shape[1]), (train_pos_edges[0], train_pos_edges[1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
    
        if self_loop == True:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)

        features = torch.FloatTensor(np.array(features.todense()))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        train_edges = train_edges.tolist()
        val_edges = val_edges.tolist()
        test_edges = test_edges.tolist()
        train_label = train_label.type(torch.float)
        val_label = val_label.type(torch.float)
        test_label = test_label.type(torch.float)

        return adj, features, train_edges, val_edges, test_edges, \
                    train_label, val_label, test_label

    elif task == 'nodecls':
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
    
        if self_loop == True:
            adj = normalize(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj)
        
        
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test
    
    else:
        raise Exception("hyper-parameter `task` belongs to \{'nodecls', 'linkpred'\}.")

# %% [markdown]
# layers

# %%
#layers

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# %% [markdown]
# models

# %%
#models

import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution
from torch_geometric.nn import PairNorm

class GCN(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, dropout, 
                 layer_num=2, activation='relu', drop_edge=False, pair_norm=False):
        super(GCN, self).__init__()

        self.gc_inp = GraphConvolution(in_channels, hid_channels)
        self.gc_hids = nn.ModuleList([GraphConvolution(hid_channels, hid_channels) for _ in range(layer_num-2)])
        self.gc_out = GraphConvolution(hid_channels, out_channels)
        if activation == 'relu':
            self.activate = F.relu
        elif activation == 'sigmoid':
            self.activate = torch.sigmoid
        elif activation == 'tanh':
            self.activate =  torch.tanh

        
        self.pair_norm = pair_norm
        if pair_norm == True:
            self.norm = PairNorm()
        

        self.dropout = nn.Dropout(dropout)
        
        # for ppi dataset
        self.linear_out = nn.Linear(out_channels, 121)

    def forward(self, x, adj, task='nodecls', edges=None, ppi=False):
        x = self.gc_inp(x, adj)
        x = self.activate(x)

        for gc_layer in self.gc_hids:
            x = self.dropout(x)
            x = gc_layer(x, adj)
            
            if self.pair_norm:
                x = self.norm(x)
            
            x = self.activate(x)

        x = self.dropout(x)
        x = self.gc_out(x, adj)

        if task == 'nodecls':
            if ppi == False:
                # x.shape = [node_num, label_class_num]
                return F.log_softmax(x, dim=1)
            else:
                x = self.linear_out(x)
                # x.shape = [node_num, label_dim]
                return x
        elif task == 'linkpred':
            # x.shape = [node_num, hid_channels]
            assert edges != None
            src = x[edges[0]] # shape = [src_num, hid_channels]
            dst = x[edges[1]] # shape = [node_num, hid_channels]
            inner_prods = (src * dst).sum(dim=-1) # shape =[src_num]
            return inner_prods

# %% [markdown]
# train

# %%
# train
import matplotlib
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score

# Replace the following imports with your actual implementations
# from utils import load_data, accuracy, load_ppi_data
# from models import GCN

# Set your desired values for the arguments
class Args:
    def __init__(self):
        self.no_cuda = False
        self.fastmode = False
        self.seed = 42
        self.epochs = 50
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.01
        self.drop_edge = 0.0   #Exp 2
        self.pair_norm = True  #Exp 2
        self.self_loop = True  #Exp 1
        self.layer_num = 3  #Exp 1
        self.activate = 'relu'  #Exp 3
        self.dataset = 'citeseer'  # Set your dataset
        self.task = 'nodecls'  # Set your task

args = Args()

print(args.self_loop)
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.task == 'nodecls':
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        adj, features, labels, idx_train, idx_val, idx_test = load_data(
            dataset=args.dataset,
            task=args.task,
            self_loop=args.self_loop
        )
        # Model and optimizer
        model = GCN(in_channels=features.shape[1],
                    hid_channels=args.hidden,
                    out_channels=labels.max().item() + 1,
                    dropout=args.dropout,
                    layer_num=args.layer_num,
                    activation=args.activate,
                    drop_edge=args.drop_edge,
                    pair_norm=args.pair_norm)
    elif args.dataset == 'ppi':
        adj, features, labels, idx_train, idx_val, idx_test = load_ppi_data(
            task=args.task,
            self_loop=args.self_loop
        )
        # Model and optimizer
        model = GCN(in_channels=features.shape[1],
                    hid_channels=args.hidden,
                    out_channels=args.hidden,
                    dropout=args.dropout,
                    layer_num=args.layer_num,
                    activation=args.activate,
                    drop_edge=args.drop_edge,
                    pair_norm=args.pair_norm)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

elif args.task == 'linkpred':
    if args.dataset == 'cora' or args.dataset == 'citeseer':
        adj, features, train_edges, val_edges, test_edges, \
        train_label, val_label, test_label = load_data(
            dataset=args.dataset,
            task=args.task,
            self_loop=args.self_loop
        )
    elif args.dataset == 'ppi':
        adj, features, train_edges, val_edges, test_edges, \
        train_label, val_label, test_label = load_ppi_data(
            task=args.task,
            self_loop=args.self_loop
        )
    model = GCN(in_channels=features.shape[1],
                hid_channels=args.hidden,
                out_channels=args.hidden,
                dropout=args.dropout,
                layer_num=args.layer_num,
                activation=args.activate,
                drop_edge=args.drop_edge,
                pair_norm=args.pair_norm)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        train_label = train_label.cuda()
        val_label = val_label.cuda()
        test_label = test_label.cuda()

else:
    raise Exception('task({}) is supposed to belong to {"nodecls", "linkpred"}.'.format(task))

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.task == 'nodecls':
    if args.dataset != 'ppi':
        criterion = F.nll_loss
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
elif args.task == 'linkpred':
    criterion = torch.nn.BCEWithLogitsLoss()

val_performances = list()
test_performances = list()
train_losses = []  # to store training loss for plotting

def train(epoch, task='nodecls'):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    if task == 'nodecls':
        if args.dataset != 'ppi':
            output = model(features, adj)
            loss_train = criterion(output[idx_train], labels[idx_train])
        else:
            output = model(x=features, adj=adj, ppi=True)
            loss_train = criterion(output[idx_train], labels[idx_train].float())

        if args.dataset != 'ppi':
            acc_train = accuracy(output[idx_train], labels[idx_train])
        else:
            preds = (output[idx_train] > 0).float().cpu()
            f1_train = f1_score(labels[idx_train].cpu(), preds, average='micro')

    elif task == 'linkpred':
        output = model(features, adj, 'linkpred', train_edges)
        loss_train = criterion(output, train_label)
        logits = torch.sigmoid(output)
        auc_train = roc_auc_score(train_label.cpu().numpy(), logits.detach().cpu().numpy())

    loss_train.backward()
    optimizer.step()

    train_losses.append(loss_train.item())  # append the current training loss

    model.eval()
    if task == 'nodecls':
        if args.dataset != 'ppi':
            output = model(features, adj)
            loss_val = criterion(output[idx_val], labels[idx_val])
        else:
            output = model(x=features, adj=adj, ppi=True)
            loss_val = criterion(output[idx_val], labels[idx_val].float())

        if args.dataset != 'ppi':
            acc_val = accuracy(output[idx_val], labels[idx_val])
        else:
            preds = (output[idx_val] > 0).float().cpu()
            f1_val = f1_score(labels[idx_val].cpu(), preds, average='micro')

        if args.dataset != 'ppi':
            loss_test = criterion(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        else:
            loss_test = criterion(output[idx_test], labels[idx_test].float())
            preds = (output[idx_test] > 0).float().cpu()
            f1_test = f1_score(labels[idx_test].cpu(), preds, average='micro')

        if args.dataset != 'ppi':
            val_performances.append(acc_val.item())
            test_performances.append(acc_test.item())
        else:
            val_performances.append(f1_val.item())
            test_performances.append(f1_test.item())

    elif task == 'linkpred':
        output = model(features, adj, 'linkpred', val_edges)
        loss_val = criterion(output, val_label)
        logits = torch.sigmoid(output)
        auc_val = roc_auc_score(val_label.cpu().numpy(), logits.detach().cpu().numpy())

        output = model(features, adj, 'linkpred', test_edges)
        loss_test = criterion(output, test_label)
        logits = torch.sigmoid(output)
        auc_test = roc_auc_score(test_label.cpu().numpy(), logits.detach().cpu().numpy())

        val_performances.append(auc_val)
        test_performances.append(auc_test)

def test(task='nodecls'):
    if task == 'nodecls':
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    elif task == 'linkpred':
        model.eval()
        with torch.no_grad():
            output = model(features, adj, 'linkpred', test_edges)
            loss_test = criterion(output, test_label)
            logits = torch.sigmoid(output)
        auc_test = roc_auc_score(test_label.cpu().numpy(), logits.detach().cpu().numpy())

def output_best(val_performances, test_performances, task='nodecls'):
    val_performances = np.array(val_performances)
    max_id = np.argmax(val_performances)
    if task == 'linkpred':
        print("Test set results (with best validation performance):",
              "auc score= {:.4f}".format(test_performances[max_id]))
        pass
    else:
        if args.dataset != 'ppi':
            print("Test set results (with best validation performance):",
                  "acc = {:.4f}".format(test_performances[max_id]))
        else:
            print("Test set results (with best validation performance):",
                  "f1_score = {:.4f}".format(test_performances[max_id]))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, args.task)

# Testing
# test(args.task)
print('dataset:', args.dataset, ' --- task:', args.task,
      ' --- self_loop:', args.self_loop,
      ' --- layer_num:', args.layer_num,
      ' --- pair_norm:', args.pair_norm,
      ' --- activate:', args.activate,
      ' --- hidden:', args.hidden)


output_best(val_performances, test_performances, args.task)

torch.save(model.state_dict(), 'model.pth')
print('-----------------------------------------')


# %% [markdown]
# test

# %%
# test

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
# from utils import load_data, load_ppi_data
# from models import GCN

# Set your desired values for the arguments
class TestArgs:
    def __init__(self):
        self.no_cuda = False
        self.fastmode = False
        self.seed = 42
        self.epochs = 50 # Set a smaller number of epochs for testing
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
        self.drop_edge = 0.0  #Exp 2
        self.pair_norm = True  #Exp 2  
        self.self_loop = True  #Exp 1
        self.layer_num = 3  #Exp 1
        self.activate = 'relu'  #Exp 3
        self.dataset = 'cora'  # Choose the dataset for testing
        self.task = 'linkpred'  # Choose the task for testing ('nodecls' or 'linkpred')

test_args = TestArgs()

# Load data for testing
if test_args.task == 'nodecls':
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        dataset=test_args.dataset,
        task=test_args.task,
        self_loop=test_args.self_loop
    )
elif test_args.task == 'linkpred':
    adj, features, train_edges, val_edges, test_edges, \
    train_label, val_label, test_label = load_data(
        dataset=test_args.dataset,
        task=test_args.task,
        self_loop=test_args.self_loop
    )

# Model and optimizer for testing
model = GCN(in_channels=features.shape[1],
            hid_channels=test_args.hidden,
            out_channels=labels.max().item() + 1,
            dropout=test_args.dropout,
            layer_num=test_args.layer_num,
            activation=test_args.activate,
            drop_edge=test_args.drop_edge,
            pair_norm=test_args.pair_norm)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=test_args.lr, weight_decay=test_args.weight_decay)

# Test the model
model.train()
optimizer.zero_grad()

if test_args.task == 'nodecls':
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
else:
    output = model(features, adj, 'linkpred', train_edges)
    loss_train = torch.nn.BCEWithLogitsLoss()(output, train_label)

loss_train.backward()
optimizer.step()

model.eval()

# Evaluate on validation set for parameter tuning
if test_args.task == 'nodecls':
    output_val = model(features, adj)
    loss_val = F.nll_loss(output_val[idx_val], labels[idx_val])
    acc_val = accuracy_score(labels[idx_val].cpu().numpy(), output_val[idx_val].argmax(dim=1).cpu().numpy())
    print(f'Validation Accuracy: {acc_val:.4f}')
else:
    output_val = model(features, adj, 'linkpred', val_edges)
    loss_val = torch.nn.BCEWithLogitsLoss()(output_val, val_label)
    auc_val = roc_auc_score(val_label.cpu().numpy(), torch.sigmoid(output_val).detach().cpu().numpy())
    print(f'Validation AUC: {auc_val:.4f}')

# Test the model on the test set
if test_args.task == 'nodecls':
    output_test = model(features, adj)
    loss_test = F.nll_loss(output_test[idx_test], labels[idx_test])
    acc_test = accuracy_score(labels[idx_test].cpu().numpy(), output_test[idx_test].argmax(dim=1).cpu().numpy())
    print(f'Test Accuracy: {acc_test:.4f}')
else:
    output_test = model(features, adj, 'linkpred', test_edges)
    loss_test = torch.nn.BCEWithLogitsLoss()(output_test, test_label)
    auc_test = roc_auc_score(test_label.cpu().numpy(), torch.sigmoid(output_test).detach().cpu().numpy())
    print(f'Test AUC: {auc_test:.4f}')


# %% [markdown]
# plots

# %%
#plots

import matplotlib.pyplot as plt

# Plot Top 1 Accuracy over epochs for validation and test sets
plt.figure(figsize=(12, 6))
plt.plot(val_performances, label='Validation', marker='o')
plt.plot(test_performances, label='Test', marker='o')
plt.title('Top 1 Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Top 1 Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot the training loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', marker='o', color='orange')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



