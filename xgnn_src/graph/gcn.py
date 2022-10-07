# from cv2 import line
import torch
import torch.nn as nn
from torch.nn import functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from xgnn_src.shared_networks import MLP, EdgeMask, EdgeMask2

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 graph_pooling_type="mean",
                 linear_pooling_type="sum"):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))
        self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        # output layer
        self.linear_pooling_type = linear_pooling_type
        if self.linear_pooling_type == 'sum':
            self.linears_prediction = torch.nn.ModuleList()
            for i in range(n_layers):
                if i == 0:
                    self.linears_prediction.append(nn.Linear(in_feats, n_classes))
                else:
                    self.linears_prediction.append(nn.Linear(n_hidden, n_classes))
        else:
            self.linears_prediction = nn.Linear(n_hidden, n_classes)
        
        self.dropout = None
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, features, adj=None):
        h = features
        hiddens = None
        if self.linear_pooling_type == 'sum':
            hiddens = [features]
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=adj)
            h = self.batch_norms[i](h)
            h = F.relu(h) # before
            if hiddens:
                hiddens.append(h)
        g.ndata['emb'] = h
        # readout        
        if hiddens:
            scores = 0
            pls = []
            for i, h in enumerate(hiddens):
                pooled = self.pool(g, h)
                if not self.dropout is None:
                    pooled = self.dropout(pooled)
                # scores += self.dropout(self.linears_prediction[i](pooled))
                scores += self.linears_prediction[i](pooled)
                pls.append(pooled)
        else:
            pls = self.pool(g, h)
            if not self.dropout is None:
                pls = self.dropout(pls)
            scores = self.linears_prediction(pls)
        return scores, pls

class GCN_MLP(GCN): # learn adj from given node embeddings
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 edge_dim,
                 graph_pooling_type="mean",
                 linear_pooling_type="sum",
                 adj_norm='sigmoid',
                 adj_sym=False,
                 norm_type='ln',
                 graph_dropout=0.5):
        super(GCN_MLP, self).__init__(in_feats, n_hidden, n_classes, n_layers, graph_dropout,
                                    graph_pooling_type, linear_pooling_type)
        self.edge_mask = EdgeMask2(edge_dim, n_hidden, dropout, adj_norm, adj_sym, norm_type)

    def get_mask_loss(self):
        return self.edge_mask.get_mask_loss()

    def get_size_loss(self):
        return self.edge_mask.get_size_loss()

    def forward(self, g, features, node_embeddings=None, beta=1., use_norm_adj=True):
        if node_embeddings is None:
            node_embeddings = features
        adj = self.edge_mask(g, node_embeddings, beta=beta, use_norm_adj=use_norm_adj)
        return super().forward(g, features, adj)

class GCN_MLP2(GCN): # use seperate mlp to learn adj
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 edge_dim,
                 graph_pooling_type="mean",
                 linear_pooling_type="sum",
                 adj_norm='sigmoid',
                 adj_sym=False,
                 norm_type='ln',
                 graph_dropout=0.5):
        super(GCN_MLP2, self).__init__(in_feats, n_hidden, n_classes, n_layers, graph_dropout,
                                    graph_pooling_type, linear_pooling_type)
        self.node_mlp = MLP(in_feats, [n_hidden], n_hidden, F.relu, 0.0, True, norm_type)
        self.edge_mask = EdgeMask2(edge_dim, n_hidden, dropout, adj_norm, adj_sym, norm_type)
        self.use_norm_adj = False
        self.beta = 1.

    def get_mask_loss(self):
        return self.edge_mask.get_mask_loss()

    def get_size_loss(self):
        return self.edge_mask.get_size_loss()

    def forward(self, g, features, node_embeddings=None, beta=None, use_norm_adj=None):
        if beta is None:
            beta = self.beta
        if use_norm_adj is None:
            use_norm_adj = self.use_norm_adj
        node_embeddings = self.node_mlp(features)
        adj = self.edge_mask(g, node_embeddings, beta=beta, use_norm_adj=use_norm_adj)
        return super().forward(g, features, adj)