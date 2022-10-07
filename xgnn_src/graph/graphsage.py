import torch.nn as nn
from xgnn_src.shared_networks import EdgeMask2
from dgl.nn.pytorch.conv import APPNPConv, SAGEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 graph_pooling_type='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.linears_prediction = nn.Linear(n_hidden, n_classes)
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, inputs, adj=None):
        h = self.dropout(inputs)
        for layer in self.layers:
            h = layer(g, h, adj)
            h = self.activation(h)
            h = self.dropout(h)

        g.ndata['emb'] = h # store the last hidden before output as embedding
        pls = self.pool(g, h)
        score_over_layer = self.dropout(self.linears_prediction(pls))
        return score_over_layer, pls

class GraphSAGE_MLP(GraphSAGE):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 graph_pooling_type='mean'):
        super(GraphSAGE_MLP, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type, graph_pooling_type)
        self.edge_mask = EdgeMask2(n_hidden*2, n_hidden, dropout, 'sigmoid', False, 'bn')

    def forward(self, g, features, node_embeddings=None):
        if node_embeddings is None:
            node_embeddings = features
        adj = self.edge_mask(g, node_embeddings)
        return super().forward(g, features, adj)