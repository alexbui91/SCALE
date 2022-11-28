"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse
from secrets import choice


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        # dataset
        self.parser.add_argument(
            '--dataset', type=str, default="MUTAG",
            choices=['Amazon', 'BA', 'BA3', 'MUTAG', 'Mutagenicity', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'NCI1', 'PROTEINS', 'PTC', 'REDDITBINARY', 'REDDITMULTI5K'],
            help='name of dataset (default: MUTAG)')
        self.parser.add_argument(
            '--datapath', type=str,
            help="path to dataset"
        )
        self.parser.add_argument(
            '--batch_size', type=int, default=32,
            help='batch size for training and validation (default: 32)')
        self.parser.add_argument(
            '--split_name', type=str, default='fold10',
            help='Select fold to split'
        )
        self.parser.add_argument(
            '--fold_idx', type=int, default=0,
            help='the index(<10) of fold in 10-fold validation.')
        self.parser.add_argument(
            '--split_ratio', type=float, default=0.7,
            help='Use when split_name is rand.')
        self.parser.add_argument(
            '--filename', type=str, default="",
            help='output file')
        self.parser.add_argument(
            '--degree_as_nlabel', action="store_true",
            help='use one-hot encodings of node degrees as node feature vectors')
        self.parser.add_argument(
            '--load_data', type=str,
            help="load pre-processing dataset"
        )
        self.parser.add_argument(
            '--store_data', type=str,
            help="store pre-processing data to"
        )
        # device
        self.parser.add_argument(
            '--disable-cuda', action='store_true',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=0,
            help='which gpu device to use (default: 0)')

        # net
        self.parser.add_argument('--model_name', type=str,
            default='gin', choices=['gin', 'gcn', 'gcn2', 'gcn_mlp', 'gin_mlp', 'graphsage', 'gat'],
            help='type of model for message passing')
        self.parser.add_argument(
            '--num_layers', type=int, default=5,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=2,
            help='number of MLP layers(default: 2). 1 means linear model.')
        self.parser.add_argument(
            '--hidden_dim', type=int, default=64,
            help='number of hidden units (default: 64)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="sum", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--linear_pooling_type', type=str,
            default="sum", choices=["sum", "last"],
            help='type of linear prediction pooling: sum or last')
        self.parser.add_argument(
            '--norm_type', type=str,
            default='ln',
            help='Type of explainer norm',
            choices=['bn', 'ln']
        )
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--epochs', type=int, default=350,
            help='number of epochs to train (default: 350)')
        self.parser.add_argument(
            '--lr', type=float, default=0.01,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0.5,
            help='final layer dropout (default: 0.5)')
        self.parser.add_argument(
            '--graph_dropout', type=float, default=0.5,
            help='dropout using in graph layer (default: 0.5)')
        self.parser.add_argument(
            '--beta', type=float, default=5.,
            help='Use bias for discretize mask')
        self.parser.add_argument(
            '--use_norm_adj', action="store_true",
            help='Use norm adj or not')
        self.parser.add_argument(
            '--temp', type=float, default=2.,
            help='temperature for soft-kd')
        self.parser.add_argument(
            '--kl_term', type=float, default=4.,
            help='coefficient of soft-entropy loss or kl loss')
        self.parser.add_argument(
            '--mk_term', type=float, default=0.,
            help='coefficient of mask entropy loss (control masking explainer)')
        self.parser.add_argument(
            '--sl_term', type=float, default=0.,
            help='coefficient of size loss (control masking explainer)')
        self.parser.add_argument(
            '--budget', type=float, default=0.,
            help='use w/ sl_term to control size loss')
        self.parser.add_argument(
            '--kd_strategy', type=str, default='naive',
            choices=['naive', 'ensemble', 'ensemble_model'],
            help='Online KD strategy')
        self.parser.add_argument(
            '--neg_ratio', type=float, default=1.,
            help='')
        self.parser.add_argument(
            '--sym', action="store_true",
            help='create symmetric mask')
        self.parser.add_argument(
            '--load_from', type=str,
            default="",
            help="model path checkpoint"
        )
        self.parser.add_argument(
            '--model_path', type=str,
            default="gcn",
            help="model path checkpoint"
        )

        # done
        self.args = self.parser.parse_args()
