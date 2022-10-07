import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.transforms import ToSimple, AddReverse

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
                 batch_norm=False,
                 norm_type='ln'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        hiddens = [in_feats] + hiddens + [n_classes]
        print("norm type:", norm_type)
        for i in range(0, len(hiddens)-1):
            self.layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            if i == len(hiddens) - 2:
                continue
            if not batch_norm:
                continue
            if norm_type == 'ln':
                self.norms.append(nn.LayerNorm(hiddens[i+1]))
            else:
                self.norms.append(nn.BatchNorm1d(hiddens[i+1]))
        self.use_norm = batch_norm
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            if self.use_norm:
                h = self.norms[i](h)
            h = self.activation(h)
            if not self.dropout is None:
                h = self.dropout(h)
        logits = self.layers[-1](h)
        return logits

class MLP2(nn.Module):
    def __init__(self,
                 hiddens,
                 activation,
                 dropout):
        super(MLP2, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(hiddens)-1):
            self.layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            if i != len(hiddens) - 2:
                self.batch_norms.append(nn.BatchNorm1d(hiddens[i+1]))
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            h = self.batch_norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        logits = self.layers[-1](h)
        return logits

class MLP_NORM(nn.Module): # norm input
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
                 batch_norm=False,
                 norm_type='ln'):
        super(MLP_NORM, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        hiddens = [in_feats] + hiddens + [n_classes]
        print("norm type:", norm_type)
        self.input_batch_norm = None
        if batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(in_feats)
        for i in range(0, len(hiddens)-1):
            self.layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
            if i == len(hiddens) - 2:
                continue
            if not batch_norm:
                continue
            if norm_type == 'ln':
                self.norms.append(nn.LayerNorm(hiddens[i+1]))
            else:
                self.norms.append(nn.BatchNorm1d(hiddens[i+1]))
        self.use_norm = batch_norm
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        if not self.input_batch_norm is None:
            h = self.input_batch_norm(h)
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            if self.use_norm:
                h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
        logits = self.layers[-1](h)
        return logits

class MLP_PRED(MLP): # use for explanation
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
                 batch_norm=False,
                 norm_type='ln'):
        super(MLP_PRED, self).__init__(in_feats, hiddens, n_classes, activation, dropout, batch_norm, norm_type)
    
    def predict_proba(self, features: np.ndarray): # use for explanation
        features = torch.from_numpy(features)
        logits = self.forward(features)
        return logits.detach().numpy()

    def forward(self, features: torch.Tensor):
        logits = super().forward(features)
        logits = F.softmax(logits, 1)
        return logits

class EdgeMask(nn.Module):
    def __init__(self, input_dim, n_hidden, dropout, adj_norm='sigmoid', sym=False, norm_type='ln'):
        super(EdgeMask, self).__init__()
        self.edge_mlp = MLP(input_dim, [n_hidden, n_hidden//2], 1, F.relu, dropout, True, norm_type)
        self.adj_norm = adj_norm
        self.reverse = AddReverse(copy_edata=True)
        self.simplify = ToSimple(aggregator='mean')
        self.mask = None
        self.sym = sym
    
    def compute_edge_norm(self, g):
        out_degs = g.out_degrees().float().clamp(min=1)
        out_norm = torch.pow(out_degs, -0.5)
        in_degs = g.in_degrees().float().clamp(min=1)
        in_norm = torch.pow(in_degs, -0.5)
        msg = fn.u_mul_v('_out_deg', '_in_deg', '_norm_adj')
        with g.local_scope():
            g.ndata['_out_deg'] = out_norm
            g.ndata['_in_deg'] = in_norm
            g.apply_edges(msg)
            return g.edata['_norm_adj'].detach()

    def compute_adj(self, g, embeddings):
        def concat_message(edges):
            return {'edge_emb': torch.cat([edges.src['emb'], edges.dst['emb']], dim=1)}
        with g.local_scope():
            g.ndata['emb'] = embeddings
            g.apply_edges(concat_message)
            return g.edata['edge_emb']

    def concrete(self, adj, bias=0., beta=5.):
        random_noise = torch.rand(adj.size()).to(adj.device)
        if bias > 0. and bias < 0.5:
            r = 1 - bias - bias
            random_noise = r * random_noise + bias
        gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
        gate_inputs = (gate_inputs + adj) / beta
        gate_inputs = F.sigmoid(gate_inputs)
        return gate_inputs
    
    def symmetrize_mask(self, g, mask):
        with g.local_scope():
            g.edata['w'] = mask
            gt = self.reverse(g)
            gt = self.simplify(gt.to('cpu'))
            masked_adj = gt.edata['w']
        return masked_adj.to(g.device)

    def get_mask_loss(self):
        mask = self.mask*0.99+0.005
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        return mask_ent.sum()

    def get_size_loss(self):
        if not self.mask is None:
            return self.mask.sum()
        return 0.

    def forward(self, g, node_embeddings, beta=1., use_norm_adj=True):
        # use_norm_adj = true for ba, false for mutag
        edge_weight = self.compute_adj(g, node_embeddings)
        edge_weight = self.edge_mlp(edge_weight).flatten() 
        if self.adj_norm == 'sigmoid':
            mask = self.concrete(edge_weight, beta=beta)
            if self.sym:
                mask = self.symmetrize_mask(g, mask)
            self.mask = mask
            adj = mask
            if use_norm_adj:
                adj = adj * self.compute_edge_norm(g)
        else:
            adj = edge_softmax(g, edge_weight, norm_by='dst')
        return adj

class EdgeMask2(EdgeMask):
    def __init__(self, input_dim, n_hidden, dropout, adj_norm='sigmoid', sym=False, norm_type='ln'):
        super(EdgeMask2, self).__init__(input_dim, n_hidden, dropout, adj_norm, sym, norm_type)
        self.edge_mlp = MLP_NORM(input_dim, [n_hidden, n_hidden//2], 1, F.relu, dropout, True, norm_type)

class ComponentAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ComponentAttention, self).__init__()
        self.weight = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, logits):
        h = self.weight(logits)
        h = F.softmax(h, 1)
        h = torch.swapaxes(h, 1, 2)
        h = torch.matmul(h, logits)
        return torch.squeeze(h)

class NaiveTeacher(nn.Module):
    def __init__(self, input_dim, pooling_type='mean'):
        super(NaiveTeacher, self).__init__()
        self.att = None
        if pooling_type == 'att':
            self.att = ComponentAttention(input_dim)

    def forward(self, logits):
        combined_logits = torch.cat([l.unsqueeze(1) for l in logits], 1)
        if not self.att is None:
            z_e = self.att(combined_logits)
        else:
            z_e = combined_logits.mean(1).squeeze()
        return z_e

class Ensemble(nn.Module):
    def __init__(self, hiddens, num_classes, dropout, pooling_type='mean'):
        super(Ensemble, self).__init__()
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for h in hiddens:
            self.batch_norms.append(nn.BatchNorm1d(h))
            self.linears.append(MLP2([h, h//2, num_classes], F.relu, dropout))
        self.pooling_type = pooling_type
    
    def forward(self, inputs):
        if len(self.linears) == 1:
            h = self.batch_norms[0](inputs)
            scores = self.linears[0](h)
        else:
            scores = 0
            for i, inp in enumerate(inputs):
                h = self.batch_norms[i](inp)
                scores += self.linears[i](h)
        return scores    

class OnlineKG(nn.Module):
    def __init__(self, base, explainer, teacher):
        super(OnlineKG, self).__init__()
        self.base = base
        self.explainer = explainer
        self.teacher = teacher

    def get_size_loss(self):
        return self.explainer.get_size_loss()

    def get_mask_loss(self):
        return self.explainer.get_mask_loss()

    def forward(self, g, features, beta=1., use_norm_adj=True):
        base_logits, base_h = self.base(g, features)
        if self.training:
            self.base.eval()
            with torch.no_grad():
                with g.local_scope():
                    self.base(g, features)
                    node_embeddings = g.ndata['emb'].clone().detach()
            self.base.train()
        else:
            node_embeddings = g.ndata['emb'].clone().detach()
        ex_logits, ex_h = self.explainer(g, features, node_embeddings, beta=beta, use_norm_adj=use_norm_adj)
        t_logits = self.teacher([base_logits, ex_logits])
        return base_logits, ex_logits, t_logits, base_h, ex_h
