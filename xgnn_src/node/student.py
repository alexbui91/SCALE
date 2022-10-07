import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import dgl.function as fn

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        hiddens = [in_feats] + hiddens + [n_classes]
        for i in range(0, len(hiddens)-1):
            self.layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        logits = self.layers[-1](h)
        return logits

class LPA(nn.Module):
    def __init__(self, g):
        super(LPA, self).__init__()
        self.g = g

    def lp(self, adj, labels, g=None):
        mes_func = fn.u_mul_e('_label', '_edge_weight', '_plabel')
        red_func = fn.sum(msg='_plabel', out='_label')
        if g is None:
            g = self.g
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
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 n_lpa=10,
                 slb=0.1):
        super(GCN_LPA, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        lpa_adj = self.init_lpa_adj(type='norm')
        self.lpa_adj = nn.Parameter(lpa_adj)
        self.lpa = LPA(self.g)
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

    def init_lpa_adj(self, type='random'):
        if type == 'random':
            lpa_adj = torch.Tensor(self.g.num_edges(), 1)
            lpa_adj = torch.nn.init.xavier_uniform_(lpa_adj)
        else:
            out_degs = self.g.out_degrees().float().clamp(min=1)
            out_norm = torch.pow(out_degs, -0.5)
            in_degs = self.g.in_degrees().float().clamp(min=1)
            in_norm = torch.pow(in_degs, -0.5)
            msg = fn.u_mul_v('_out_deg', '_in_deg', '_norm_adj')
            with self.g.local_scope():
                self.g.ndata['_out_deg'] = out_norm
                self.g.ndata['_in_deg'] = in_norm
                self.g.apply_edges(msg)
                lpa_adj = self.g.edata['_norm_adj']
        return lpa_adj

    def get_graph(self):
        lpa_adj = dgl.nn.functional.edge_softmax(self.g, self.lpa_adj)
        self.g.edata['weight'] = lpa_adj
        return self.g

    def forward(self, features, g=None):
        if g is None:
            g = self.g
        h = features
        lpa_adj = dgl.nn.functional.edge_softmax(g, self.lpa_adj, norm_by='dst')
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=lpa_adj)
        labels = None
        if self.n_lpa:
            if self.soft_label == 0.:
                labels = self.lpa(self.lpa_adj, h.detach(), self.n_lpa, self.soft_label)
            else:
                labels = self.lpa(self.lpa_adj, h.detach(), self.n_lpa, self.soft_label)
        return h, labels

class SGAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(SGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_dim, num_hidden, 1,
                                    feat_drop, attn_drop, negative_slope, False, activation))
        
        # hidden layers
        for _ in range(1, num_layers):
            self.gat_layers.append(GraphConv(num_hidden, num_hidden, norm="none"))
        
        self.gat_layers.append(GraphConv(num_hidden, num_classes, norm="none"))

    def get_graph(self):
        self.eval()
        inputs = self.g.ndata['feat']
        _, edge_weights = self.gat_layers[0](self.g, inputs, get_attention=True)
        self.g.edata['weight'] = edge_weights
        return self.g

    def forward(self, inputs):
        h, edge_weights = self.gat_layers[0](self.g, inputs, get_attention=True)
        h = h.squeeze()
        for l in range(1, self.num_layers):
            h = self.gat_layers[l](self.g, h, edge_weight=edge_weights)
        # output projection
        logits = self.gat_layers[-1](self.g, h, edge_weight=edge_weights)
        logits = logits.squeeze()
        return logits

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_dim, num_hidden, 1,
                                    feat_drop, attn_drop, negative_slope, False, activation))
        
        # hidden layers
        for _ in range(1, num_layers):
            self.gat_layers.append(GraphConv(num_hidden, num_hidden, norm="none"))
        
        self.gat_layers.append(GraphConv(num_hidden, num_classes, norm="none"))

    def forward(self, inputs):
        h, edge_weights = self.gat_layers[0](self.g, inputs, get_attention=True)
        h = h.squeeze()
        for l in range(1, self.num_layers):
            h = self.gat_layers[l](self.g, h, edge_weight=edge_weights)
        # output projection
        logits = self.gat_layers[-1](self.g, h, edge_weight=edge_weights)
        logits = logits.squeeze()
        return logits

class EGNN(nn.Module):
    def __init__(self,
                g,
                g1,
                in_dim,
                num_hidden,
                num_classes,
                dropout):
        super(EGNN, self).__init__()
        self.g = g
        self.g1 = g1
        self.linear1 = nn.Linear(in_dim, num_hidden)
        self.linear2 = nn.Linear(in_dim, num_hidden)
        self.edge_linear = nn.Linear(num_hidden*2, 1)
        self.final_linear = nn.Linear(num_hidden, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def concrete(self, adj, bias=0., beta=1.):
        random_noise = torch.rand(adj.size()).to(adj.device)
        if bias > 0. and bias < 0.5:
            r = 1 - bias - bias
            random_noise = r * random_noise + bias
        gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
        gate_inputs = (gate_inputs + adj) / beta
        gate_inputs = F.sigmoid(gate_inputs)
        return gate_inputs

    def compute_adj(self, g, embeddings):
        def concat_message(edges):
            return {'edge_emb': torch.cat([edges.src['emb'], edges.dst['emb']], dim=1)}

        with g.local_scope():
            g.ndata['emb'] = embeddings
            g.apply_edges(concat_message)
            emb = self.edge_linear(g.edata['edge_emb'])
            att = dgl.nn.functional.edge_softmax(g, emb)
            return self.concrete(att)

    def compute_node_embedding(self, g, x, mask):
        msg = dgl.function.e_mul_v("alpha", "emb", "m")
        reduce = dgl.function.sum("m", "new_emb")
        with g.local_scope():
            g.ndata['emb'] = x
            g.edata['alpha'] = mask
            g.update_all(msg, reduce)
            return g.ndata["new_emb"]
    
    def compute_masks(self, inputs):
        x1 = self.linear1(inputs)
        x2 = self.linear2(inputs)
        mask1 = self.compute_adj(self.g, x1)
        mask2 = self.compute_adj(self.g1, x2)
        return x1, x2, mask1, mask2

    def forward(self, inputs):
        x1, x2, mask1, mask2 = self.compute_masks(inputs)
        emb1 = self.compute_node_embedding(self.g, x1, mask1)
        emb2 = self.compute_node_embedding(self.g1, x2, mask2)
        z = emb1 + emb2
        z = self.dropout(z)
        logits = self.final_linear(z)
        return logits

        