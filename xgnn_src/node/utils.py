import os
import pickle as pkl

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,\
                    FraudAmazonDataset, FraudYelpDataset, BAShapeDataset, BACommunityDataset,\
                    TreeGridDataset, TreeCycleDataset
import dgl.function as fn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from xgnn_src.node.student import EGNN

from xgnn_src.node.teacher import APPNP, GraphSAGE, GAT, GCN, GCN_MLP2
from xgnn_src.node.teacher_online import APPNP2, GCN2, GraphSAGE2
from xgnn_src.node.student_online import APPNP_MLP, GCN_LPA, GCN_MLP, GraphSAGE_MLP
import xgnn_src.node.student_online as student_online

class DistanceLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits, soft_logits):
        loss = torch.cdist(logits, soft_logits)
        return loss.mean()

def std_normalize(features):
    std = torch.std(features, 0)
    mn = torch.mean(features, 0)
    features = (features - mn) / std
    return features

def kl(p1, p2):
    p1 = F.softmax(p1, 1)
    p2 = F.softmax(p2, 1)
    dv = torch.log(p1 + 1e-7) - torch.log(p2 + 1e-7)
    return (p1 * dv).sum(1)

def eval_student(teacher_logits, student_logits):
    k = kl(teacher_logits, student_logits).mean()
    t_indices = torch.argmax(teacher_logits, dim=1)
    s_indices = torch.argmax(student_logits, dim=1)
    correct = torch.sum(t_indices == s_indices)
    agr = correct.item() * 1.0 / len(teacher_logits)
    return k, agr

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def calculate_score(logits, labels, evaltype='acc'):
    if evaltype == "acc":
        acc = accuracy(logits, labels)
    elif evaltype in ["rec", "f1", "auc"]:
        _, indices = torch.max(logits, dim=1)
        if evaltype == "rec":
            acc = recall_score(labels.cpu(), indices.cpu())
        elif evaltype == "f1":
            acc = f1_score(labels.cpu(), indices.cpu())
        elif evaltype == "auc":
            acc = roc_auc_score(labels.cpu(), indices.cpu())        
    return acc

def evaluate(model, features, labels, mask, mode='teacher', evaltype='acc'):
    model.eval()
    with torch.no_grad():
        if mode == 'teacher':
            logits = model(features)
        else:
            logits = model(features)
            if type(logits) == tuple:
                logits = logits[0]
        logits = logits[mask]
        labels = labels[mask]
        acc = calculate_score(logits, labels, evaltype)
        return acc

def evaluate_inductive(model, features, labels, evaltype='acc'):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        acc = calculate_score(logits, labels, evaltype)
        return acc

def process_ba_graph(g, val_ratio, test_ratio, reverse=False):
    num_nodes = g.num_nodes()
    if reverse:
        rv_func = dgl.transforms.AddReverse()
        g = rv_func(g)
    val_num = int(num_nodes * val_ratio)
    test_num = int(num_nodes * test_ratio)
    train_num = num_nodes - val_num - test_num
    rand = torch.randperm(num_nodes)
    train_nodes = rand[:train_num]
    val_nodes = rand[train_num:train_num+val_num]
    test_nodes = rand[train_num+val_num:]
    g.ndata['train_mask'] = torch.zeros((num_nodes,))
    g.ndata['train_mask'][train_nodes] = 1.
    g.ndata['train_mask'] = g.ndata['train_mask'].to(torch.bool)
    g.ndata['val_mask'] = torch.zeros((num_nodes,))
    g.ndata['val_mask'][val_nodes] = 1.
    g.ndata['val_mask'] = g.ndata['val_mask'].to(torch.bool)
    g.ndata['test_mask'] = torch.zeros((num_nodes,))
    g.ndata['test_mask'][test_nodes] = 1.
    g.ndata['test_mask'] = g.ndata['test_mask'].to(torch.bool)
    return g

# using for gnnexplainer datasets
def check_cache(data, name, pre_process=None, add_reverse=True):
    suffix = "dir"
    if add_reverse:
        suffix = "bidir"
    if os.path.exists("./datasets/%s.g"%name):
        with open("./datasets/%s.g"%name, 'rb') as f:
            g = pkl.load(f)
    else:
        g = data[0]
        if not pre_process is None:
            pre_process(g)
        feat = g.ndata['feat']
        in_degrees = std_normalize((g.in_degrees(g.nodes()) + 1).to(torch.float32))
        out_degrees = std_normalize((g.out_degrees(g.nodes()) + 1).to(torch.float32))
        g.ndata['feat'] = torch.cat([feat, torch.unsqueeze(in_degrees, 1), torch.unsqueeze(out_degrees, 1)], 1)
        g.ndata['feat'] = g.ndata['feat'].to(torch.float32)
        g = process_ba_graph(g, 0.1, 0.1, add_reverse)
        with open("./datasets/%s_%s.g"%(name, suffix), 'wb') as f:
            pkl.dump(g, f)
    return g

def load_data(dataset, graph_type="", n_classes=-1, selected_features=None, skip_features=False, add_reverse=False):
    # load and preprocess dataset
    if dataset == 'cora':
        data = CoraGraphDataset()
        g = data[0]
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
        g = data[0]
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()
        g = data[0]
    elif dataset == "amazon":
        data = FraudAmazonDataset()
        g = data[0]
        g.ndata['feat'] = std_normalize(g.ndata['feature'])
    elif dataset == "yelp":
        data = FraudYelpDataset()
        g = data[0]
        g.ndata['feat'] = g.ndata['feature']
    elif dataset == "BAS":
        # python train.py --teacher-name gcn --dataset BAS --n-epochs 1000 --n-hidden 32 --self-loop --n-layers 5 --lr 0.01
        data = BAShapeDataset()
        g = check_cache(data, "ba_shape", None, add_reverse=add_reverse)
    elif dataset == "BAC":
        # python train.py --teacher-name gcn --dataset BAS --n-epochs 1000 --n-hidden 64 --self-loop --n-layers 5 --lr 0.01
        data = BACommunityDataset()
        g = check_cache(data, "ba_community", None, add_reverse=add_reverse)
    elif dataset == "TRC": # Tree-cycles
        data = TreeCycleDataset()
        g = check_cache(data, "tree_cycle", add_reverse=add_reverse)        
    elif dataset == "TRG": # Tree-Grid
        data = TreeGridDataset()
        g = check_cache(data, "tree_grid", add_reverse=add_reverse)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    print(g)
    n_classes = data.num_classes
    if not selected_features is None:
        selected_features = torch.tensor(selected_features)
        g.ndata['feat'] = torch.index_select(g.ndata['feat'], -1, selected_features)

    if skip_features:
        # g.ndata['feat'] = torch.ones((g.num_nodes(),8), dtype=g.ndata['feat'].dtype).to(g.ndata['feat'].device)
        g.ndata['feat'] = torch.ones_like(g.ndata['feat']).to(g.ndata['feat'].device)

    if dataset in ["yelp", "amazon"]:
        if graph_type != "homo":
            g = g[graph_type]
        else:
            g = dgl.to_homogeneous(g, ndata=['feat', 'train_mask', 'val_mask', 'test_mask', 'label'])        
        print(graph_type, g)
    return g, n_classes

def init_teacher(args, g, in_feats, n_classes):
    if args.teacher_name == "appnp":
        teacher =  APPNP(g, in_feats, args.hidden_sizes, n_classes,
                        F.relu, args.dropout, args.dropout, 0.1, 20)
    if args.teacher_name == "appnp2": # use for online knowledge distll
        teacher =  APPNP2(in_feats, args.hidden_sizes, n_classes,
                        F.relu, args.dropout, args.dropout, 0.1, 20)
    elif args.teacher_name == "graphsage":
        teacher = GraphSAGE(g, in_feats, args.n_hidden, n_classes,
                            args.n_layers, F.relu, args.dropout, "mean")
    elif args.teacher_name == "graphsage2": # use for online knowledge distll
        teacher = GraphSAGE2(in_feats, args.n_hidden, n_classes,
                            args.n_layers, F.relu, args.dropout, "mean")
    elif args.teacher_name == "gcn":
        teacher = GCN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    elif args.teacher_name == "gcn2": # use for online knowledge distll
        teacher = GCN2(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.all_layer_dp, not args.skip_norm)
    elif args.teacher_name == "gcn_mlp": # standalone mlp to learn adjacency
        teacher = GCN_MLP2(g, in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout, args.n_hidden * 2, F.relu, 'softmax', 'bn', args.all_layer_dp, False)
    else:
        heads = None
        if args.n_layers:
            heads = ([args.n_heads] * args.n_layers) + [args.n_out_heads]
        teacher = GAT(g, args.n_layers, in_feats, args.n_hidden, n_classes, heads, F.elu,
                    args.dropout, args.dropout, args.negative_slope, args.residual)
    print(teacher)
    return teacher

def init_graph_student(student_type, g, in_feats, num_classes, dropout, n_hidden=None, n_layers=None,
                        hidden_sizes=None, n_lpa=10, slb=None, all_layer_dp=False, skip_norm=False):
    if student_type == 'lpa':
        graph_x = GCN_LPA(in_feats, n_hidden, num_classes, n_layers, F.relu, dropout, n_lpa, slb)
        graph_x.init_lpa_adj(g, 'norm')
    elif student_type == 'appnp':
        graph_x = APPNP_MLP(in_feats, hidden_sizes, num_classes, F.relu, dropout, dropout, 0.1, 20)
    elif student_type == 'graphsage':
        graph_x = GraphSAGE_MLP(in_feats, n_hidden, num_classes, n_layers, F.relu, dropout, "mean")
    elif student_type == 'gcn2':
        graph_x = student_online.GCN_MLP2(in_feats, n_hidden, num_classes, n_layers, dropout, n_hidden * 2, F.relu,
                    'softmax', norm_type='bn', all_layer_dp=all_layer_dp, graph_norm_type=not skip_norm)
    else:
        graph_x = GCN_MLP(in_feats, n_hidden, num_classes, n_layers, dropout, n_hidden * 2, F.relu,
                    'softmax', norm_type='bn', all_layer_dp=all_layer_dp, graph_norm_type=not skip_norm)
    return graph_x
    
def find_sinks(g):
    g1_df = pd.DataFrame(g.out_degrees(g.nodes()).numpy())
    g1_df.columns=['deg']
    sinks = g1_df[g1_df['deg'] == 0].reset_index()['index'].to_numpy()
    return sinks

def compute_pagerank(g, d, num_nodes, check_sink=True):
    sink_values = 0
    if check_sink:
        sinks = find_sinks(g)
        if len(sinks):
            # sum sinks' pageranks
            sink_values = d * torch.index_select(g.ndata['pv'], 0, sinks).sum() / num_nodes
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                reduce_func=fn.sum(msg='m', out='m_sum'))
    g.ndata['pv'] = (1 - d) / num_nodes + sink_values + d * g.ndata['m_sum'] 
    prop = g.ndata['pv']
    prop = prop * torch.log2(prop) * -1.0
    return torch.sum(prop)

def pagerank(g, num_iter, d, check_sink=True):
    num_nodes = g.num_nodes()
    g.ndata['pv'] = torch.ones(num_nodes) / num_nodes
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()
    for i in range(num_iter):
        ent = compute_pagerank(g, d, num_nodes, check_sink)

def compute_ppr(g, d, num_nodes, check_sink=True, custom_transition=False):
    sink_values = 0
    if check_sink:
        sinks = find_sinks(g)
        if len(sinks):
            sink_values = d * torch.index_select(g.ndata['pv'], 0, sinks).sum() / num_nodes
    if custom_transition:
        g.update_all(message_func=fn.u_mul_e('pv', 'edge_importance', 'm'),
                    reduce_func=fn.sum(msg='m', out='m_sum'))
    else:
        g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
        g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                    reduce_func=fn.sum(msg='m', out='m_sum'))
    
    reset = (1-d) * g.ndata['p_mask'] 
    g.ndata['pv'] = reset + sink_values + d * g.ndata['m_sum']

def personalized_pagerank(g, num_iter, d, preferences=None, check_sink=True, transition_matrix=None):
    custom_transition = False
    if not transition_matrix is None:
        g.edata['edge_importance'] = transition_matrix
        custom_transition = True
    num_nodes = g.num_nodes()
    g.ndata['pv'] = torch.ones(num_nodes) / num_nodes
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()
    if not preferences is None:
        mask = torch.zeros((num_nodes,))
        mask[preferences] = 1.0
        g.ndata['p_mask'] =  mask / len(preferences)
    else:
        N = num_nodes
        g.ndata['p_mask'] = torch.ones(num_nodes) / num_nodes
    for i in range(num_iter):
        compute_ppr(g, d, num_nodes, check_sink, custom_transition)

def transform_embeddings(emb):
    vis = TSNE(n_jobs=1)
    emb = vis.fit_transform(emb)
    return emb

def visualize(emb, labels, savefig=None):
    _, ax = plt.subplots()
    ax.scatter(emb[:,0], emb[:,1], c=labels)
    if savefig:
        plt.savefig("./%s.png" % savefig)
    else:
        plt.show()

def make_slice_l2(g):
    nodes = g.nodes()
    src, dst = [], []
    for t in nodes:
        l1, l0 = g.in_edges(t)
        l2, _ = g.in_edges(l1)  
        for u in l2:
            u = u.item()
            if u == t:
                continue
            src.append(u)
            dst.append(t)
    l2, l0 = torch.tensor(src), torch.tensor(dst)
    g1 = dgl.graph((l2, l0))
    for k in g.ndata.keys():
        g1.ndata[k] = g.ndata[k].cpu()
    for k in g.edata.keys():
        g1.edata[k] = g.edata[k].cpu()
    return g1