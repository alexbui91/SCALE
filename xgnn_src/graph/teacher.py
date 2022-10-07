import torch
import torch.nn as nn
import torch.nn.functional as F
from xgnn_src.graph.mlp import MLP2

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
        