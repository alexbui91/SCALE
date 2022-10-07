import collections
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
from dgl.data import GINDataset, BA2MotifDataset
from xgnn_src.graph.dataloader import AmazonDataset, BA3MotifDataset, MutagenicityDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.3, reduction='sum'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, y_hat, y):
        p = F.log_softmax(y_hat / self.temperature, 1)
        weight = F.softmax(y / self.temperature, 1)
        loss = -(p * weight)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

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

def load_data(dataset, datapath="", learn_eps=None, degree_as_nlabel=None, **karg):
    if dataset == 'Mutagenicity':
        dataset = MutagenicityDataset(datapath, karg['neg_ratio'])
        gclasses = dataset.num_classes
        dim_nfeats = dataset.dim_nfeats
    elif dataset == 'BA':
        dataset = BA2MotifDataset()
        gclasses = 2
        dim_nfeats = 10
        dataset.labels = torch.argmax(dataset.labels, dim=1)
        for g in dataset.graphs:
            g.ndata['attr'] = g.ndata.pop('feat')
    elif dataset == 'BA3':
        dataset = BA3MotifDataset(datapath)
        gclasses = dataset.num_classes
        dim_nfeats = dataset.dim_nfeats
    elif dataset == 'Amazon':
        dataset = AmazonDataset(datapath)
        gclasses = dataset.num_classes
        dim_nfeats = dataset.dim_nfeats
    else:
        dataset = GINDataset(dataset, not learn_eps, degree_as_nlabel)
        dim_nfeats = dataset.dim_nfeats
        gclasses = dataset.gclasses
    return dataset, dim_nfeats, gclasses

def draw_mutag(g, weight=None, undir=True, node_size=1000, margin=0.1):
    node_names = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    colors = ['orange','#fd8dff','lime','#5cb85c','#06b8ff','orchid','darksalmon','#5cbaba','gold','bisque','tan','lightseagreen','indigo','navy']
    gx = dgl.to_networkx(g)
    edge_label = g.edata['edge_labels'].numpy()
    node_label = g.ndata['node_labels'].flatten().numpy()
    ids = collections.defaultdict(list)
    labels = {}
    for i, n in enumerate(node_label):
        ids[n].append(i)
        labels[i] = node_names[n]
            
    src, dst = g.edges()
    src, dst = src.numpy(), dst.numpy()
    
    for i, (s, d) in enumerate(zip(src, dst)):
        if not weight is None:
            gx[s][d][0]['w'] = weight[i]
        gx[s][d][0]['edge_labels'] = edge_label[i]
    if undir:
        gx = gx.to_undirected()
        
    edge_label = [v for k, v in nx.get_edge_attributes(gx, 'edge_labels').items()]
    weight = [v for k, v in nx.get_edge_attributes(gx, 'w').items()]
    if not weight:
        weight = 1
        edge_colors = "grey"
    else:
        edge_colors = []
        for i, w in enumerate(weight):
            if w > 1.0:
                edge_colors.append("red")
            elif edge_label[i]:
                edge_colors.append("pink")
            else:
                edge_colors.append("grey")
    pos = nx.kamada_kawai_layout(gx)
    ax = plt.subplot()
    ax.margins(margin)
    for k, v in ids.items():
        nx.draw_networkx_nodes(gx, pos, ax=ax, nodelist=v, node_color=colors[k], node_size=node_size)
    
    nx.draw_networkx_edges(gx, pos, width=weight, edge_color=edge_colors)
    nx.draw_networkx_labels(gx, pos, labels, font_size=14, font_color="black")

def draw_simple_graph(g, weight=None, undir=True, node_size=1000, margin=0.05):
    gx = dgl.to_networkx(g)
    src, dst = g.edges()
    for s, d, w in zip(src, dst, weight):
        gx[s.item()][d.item()][0]['w'] = w
    if undir:
        gx = gx.to_undirected()
    weight = [v for k, v in nx.get_edge_attributes(gx, 'w').items()]
    pos = nx.kamada_kawai_layout(gx)
    nodes = g.nodes().tolist()
    ax = plt.subplot()
    ax.margins(margin)
    nx.draw_networkx_nodes(gx, pos, nodelist=nodes, node_color='orange', node_size=node_size)
    if not weight:
        weight = 1
        edge_colors = "grey"
    else:
        edge_colors = []
        for w in weight:
            if w > 1.0:
                edge_colors.append("red")
            else:
                edge_colors.append("grey")
    nx.draw_networkx_edges(gx, pos, width=weight,  edge_color=edge_colors)
    labels = {i:i for i in nodes}
    nx.draw_networkx_labels(gx, pos, labels, font_size=14, font_color="black")

# return mask for an explainer w/ a given graph
def get_mask(g, base, explainer, undir=True, beta=1.):
    base.eval()
    explainer.eval()
    with torch.no_grad():
        base(g, g.ndata['attr'])
        embedding = g.ndata['emb']
        edge_weight = explainer.edge_mask.compute_adj(g, embedding)
        edge_weight = explainer.edge_mask.edge_mlp(edge_weight)
        mask = explainer.edge_mask.concrete(edge_weight, beta)
        # mask = F.sigmoid(edge_weight)
    with g.local_scope():
        num_nodes = g.num_nodes()
        adj = [[0.] * num_nodes for _ in range(num_nodes)] 
        src, dst = g.edges()
        for i, (s, d) in enumerate(zip(src, dst)):
            s, d = s.item(), d.item()
            m = mask[i].item()
            if m < 0.5:
                adj[s][d] = 0.0
                if undir:
                    continue
                adj[d][s] = 0.0
            else:
                adj[s][d] = m
        weight = []
        for s, d in zip(src, dst):
            s, d = s.item(), d.item()
            weight.append(adj[s][d])
        mask = np.array(weight)
    return mask