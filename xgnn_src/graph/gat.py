import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope=0.2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.batch_norms.append(nn.BatchNorm1d(num_hidden * heads[l-1]))
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, None))
        self.batch_norms.append(nn.BatchNorm1d(num_hidden * heads[1]))
        self.dropout = nn.Dropout(p=feat_drop)
        self.pooling = MaxPooling()
        self.linears_prediction = nn.Linear(num_hidden * heads[-1], num_classes)

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.batch_norms[l](h)
            h = self.activation(h)
        # output projection
        h = self.pooling(g, h)
        logits = self.linears_prediction(h)
        return logits, h