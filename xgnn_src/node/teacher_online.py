import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import APPNPConv, SAGEConv, GINConv, GATConv, GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from xgnn_src.shared_networks import MLP, EdgeMask2

class GCN2(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 all_layer_dp=False,
                 norm_type=True): # use all_layer_dp when last layer drop
        super(GCN2, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # input layer
        self.activation = activation
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None, norm="both"))
        if norm_type:
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None, norm="both"))
            if norm_type:
                self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm="both"))
        self.dropout = nn.Dropout(p=dropout)
        self.all_layer_dp = all_layer_dp
        self.norm_type = norm_type
        print("using norm in graph", self.norm_type)

    def forward(self, g, features, adj=None):
        h = features
        for i, layer in enumerate(self.layers):
            if (self.all_layer_dp and i != 0) or (i == len(self.layers) - 1 and not self.all_layer_dp):
                h = self.dropout(h)
            h = layer(g, h, edge_weight=adj)
            if i < len(self.layers) - 1:
                if self.norm_type:
                    h = self.batch_norms[i](h)
                h = self.activation(h)
            if i == len(self.layers) - 2 and adj is None:
                g.ndata['emb'] = h
        return h

class APPNP2(nn.Module):
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP2, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features, adj=None):
        # prediction step
        h = features
        h = self.feat_drop(h)
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(layer(h))
            if i == len(self.layers) - 2 and adj is None:
                g.ndata['emb'] = h
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h, adj)
        return h

class GraphSAGE2(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE2, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, g, inputs, adj=None):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            if l == (len(self.layers) - 2) and adj is None:
                g.ndata['emb'] = h # store the last hidden before output as embedding
        return h