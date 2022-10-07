import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
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
            if norm_type == 'ln':
                self.norms.append(nn.LayerNorm(hiddens[i+1]))
            else:
                self.norms.append(nn.BatchNorm1d(hiddens[i+1]))
            
        self.activation = activation
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            h = self.activation(h)
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