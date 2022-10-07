import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, APPNPConv
import dgl.function as fn
from xgnn_src.node.teacher_online import APPNP2, GCN2, GraphSAGE2
from xgnn_src.shared_networks import EdgeMask2, MLP

# new customized gcn w/ teacher embeddings; embeddings must be detached
class GCN_MLP(GCN2):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 edge_dim,
                 activation,
                 adj_norm='sigmoid',
                 adj_sym=False,
                 norm_type='ln',
                 all_layer_dp=False,
                 graph_norm_type=True):
        super(GCN_MLP, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, all_layer_dp, graph_norm_type)
        self.edge_mask = EdgeMask2(edge_dim, n_hidden, dropout, adj_norm, adj_sym, norm_type)

    def forward(self, g, features, node_embeddings=None):
        if node_embeddings is None:
            node_embeddings = features
        adj = self.edge_mask(g, node_embeddings)
        return super().forward(g, features, adj)

class GCN_MLP2(GCN2):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 edge_dim,
                 activation,
                 adj_norm='sigmoid',
                 adj_sym=False,
                 norm_type='ln',
                 all_layer_dp=False,
                 graph_norm_type=True):
        super(GCN_MLP2, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, all_layer_dp, graph_norm_type)
        self.node_mlp = MLP(in_feats, [n_hidden], n_hidden, F.relu, 0.0, True, norm_type)
        self.edge_mask = EdgeMask2(edge_dim, n_hidden, dropout, adj_norm, adj_sym, norm_type)

    def forward(self, g, features, node_embeddings=None):
        node_embeddings = self.node_mlp(features)
        adj = self.edge_mask(g, node_embeddings)
        return super().forward(g, features, adj)

class LPA(nn.Module):

    def lp(self, adj, labels, g=None):
        mes_func = fn.u_mul_e('_label', '_edge_weight', '_plabel')
        red_func = fn.sum(msg='_plabel', out='_label')
        with g.local_scope():
            g.ndata['_label'] = labels
            g.edata['_edge_weight'] = adj
            g.update_all(mes_func, red_func)
            res = g.ndata['_label']
        return res

    def forward(self, adj, labels, n_lpa, soft_label=0, g=None):
        if soft_label == 0.:
            # lpa style
            for _ in range(n_lpa):
                labels = self.lp(adj, labels, g)
        else:
            # appnp style
            # z = torch.softmax(h0, 1)
            z = labels.clone()
            for _ in range(n_lpa):
                z = (1 - soft_label) * self.lp(adj, labels, g) + soft_label * labels
            labels = z
        return labels

class GCN_LPA(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 n_lpa=10,
                 slb=0.1):
        super(GCN_LPA, self).__init__()
        self.layers = nn.ModuleList()
        self.lpa_adj = None
        self.lpa = LPA()
        self.n_lpa = n_lpa
        self.soft_label = slb
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm = 'none'))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm = 'none'))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, norm = 'none'))
        self.dropout = nn.Dropout(p=dropout)

    def init_lpa_adj(self, g, type='random'):
        if type == 'random':
            lpa_adj = torch.Tensor(g.num_edges(), 1)
            lpa_adj = torch.nn.init.xavier_uniform_(lpa_adj)
        else:
            out_degs = g.out_degrees().float().clamp(min=1)
            out_norm = torch.pow(out_degs, -0.5)
            in_degs = g.in_degrees().float().clamp(min=1)
            in_norm = torch.pow(in_degs, -0.5)
            msg = fn.u_mul_v('_out_deg', '_in_deg', '_norm_adj')
            with g.local_scope():
                g.ndata['_out_deg'] = out_norm
                g.ndata['_in_deg'] = in_norm
                g.apply_edges(msg)
                lpa_adj = g.edata['_norm_adj']
        self.lpa_adj = nn.Parameter(lpa_adj)

    def get_graph(self, g):
        lpa_adj = dgl.nn.functional.edge_softmax(g, self.lpa_adj)
        g.edata['weight'] = lpa_adj
        return g

    def forward(self, g, features):
        h = features
        lpa_adj = dgl.nn.functional.edge_softmax(g, self.lpa_adj, norm_by='dst')
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=lpa_adj)
        labels = None
        if self.n_lpa:
            if self.soft_label == 0.:
                labels = self.lpa(self.lpa_adj, h.detach(), self.n_lpa, self.soft_label, g)
            else:
                labels = self.lpa(self.lpa_adj, h.detach(), self.n_lpa, self.soft_label, g)
        return h, labels

class APPNP_MLP(APPNP2):
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 feat_drop,
                 edge_drop,
                 alpha,
                 k):
        super(APPNP_MLP, self).__init__(in_feats, hiddens, n_classes, activation, feat_drop, edge_drop, alpha, k)
        self.edge_mask = EdgeMask2(hiddens[-1]*2, hiddens[-1], feat_drop, 'softmax', False, 'bn')

    def forward(self, g, features, node_embeddings):
        # prediction step
        adj = self.edge_mask(g, node_embeddings)
        h = super().forward(g, features, adj)
        return h

class GraphSAGE_MLP(GraphSAGE2):
    def __init__(self, in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_MLP, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type)
        self.edge_mask = EdgeMask2(n_hidden*2, n_hidden, dropout, 'softmax', False, 'bn')
    
    def forward(self, g, inputs, node_embeddings):
        adj = self.edge_mask(g, node_embeddings)
        h = super().forward(g, inputs, adj)
        return h