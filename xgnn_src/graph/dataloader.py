"""
PyTorch compatible dataloader
"""
import math, random
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset


class GCDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=None,
                 seed=0,
                 shuffle=True,
                 split_name='fold10',
                 fold_idx=0,
                 split_ratio=0.7):

        self.shuffle = shuffle
        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        
        labels = [l for _, l in dataset]
        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle)
        elif split_name == 'fold5':
            train_idx, valid_idx = self._split_fold5(
                labels, fold_idx, seed, shuffle)
        elif split_name == 'rand':
            train_idx, valid_idx = self._split_rand(
                labels, split_ratio, seed, shuffle)
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_idx = train_idx
        self.val_idx = valid_idx
        self.train_loader = GraphDataLoader(
            dataset, sampler=train_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)
        self.valid_loader = GraphDataLoader(
            dataset, sampler=valid_sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader
    
    def train_valid_idx(self):
        return self.train_idx, self.val_idx

    # def balance_training(self, data: np.array, labels: np.array):
    #     # only work with yelp & amazon dataset
    #     train_pos_idx = labels.nonzero()[0]
    #     train_neg_idx = (labels == 0).nonzero()[0]
    #     # boostrapping
    #     pos_idx_bt = np.random.choice(train_pos_idx, (len(train_neg_idx),), replace=True)
    #     selected_bg_idx = np.concatenate([pos_idx_bt, train_neg_idx])
    #     selected_bg_idx.sort()
    #     balanced_data = data[selected_bg_idx]
    #     return balanced_data

    def _split_fold(self, labels, n_splits, fold_idx=0, seed=0, shuffle=True, balance=False):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx) 
        for idx in skf.split(np.zeros(len(labels)), labels):    # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]
        if balance:
            train_labels = np.array(labels)[train_idx]
            # train_idx = self.balance_training(train_idx, train_labels)
        return train_idx, valid_idx
    
    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True, balance=False):
        ''' 10 flod '''
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 9.")
        train_idx, valid_idx = self._split_fold(labels, 10, fold_idx, seed, shuffle, balance)
        print(
            "train_set : test_set = %i : %i" %
            (len(train_idx), len(valid_idx)))

        return train_idx, valid_idx
    
    def _split_fold5(self, labels, fold_idx=0, seed=0, shuffle=True, balance=False):
        ''' 10 flod '''
        assert 0 <= fold_idx and fold_idx < 5, print(
            "fold_idx must be from 0 to 5.")

        train_idx, valid_idx = self._split_fold(labels, 5, fold_idx, seed, shuffle, balance)
        
        print(
            "train_set : test_set = %i : %i" %
            (len(train_idx), len(valid_idx)))

        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(seed)
        np.random.shuffle(indices)
        if split_ratio < 1.:
            split = int(math.floor(split_ratio * num_entries))
            train_idx, valid_idx = indices[:split], indices[split:]
        else:
            split = int(math.floor(0.8 * num_entries))
            train_idx, valid_idx = indices, indices[split:]
        print(
            "train_set : test_set = %i : %i" %
            (len(train_idx), len(valid_idx)))

        return train_idx, valid_idx

class AmazonDataset(DGLDataset):
    def __init__(self, raw_dir=""):
        super(AmazonDataset, self).__init__('Amazon Ego')
        with open(raw_dir, 'rb') as f:
            graphs, labels, egos = pkl.load(f)
        self.num_classes = 2
        for g in graphs:
            g.ndata['attr'] = g.ndata.pop('feat')
        self.graphs = graphs            
        self.labels = torch.LongTensor(labels)
        self.egos = egos
        self.dim_nfeats = 25
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
        
class BA3MotifDataset(DGLDataset):
    def __init__(self, raw_dir=""):
        super(BA3MotifDataset, self).__init__('BA3 Motifs')
        with open(raw_dir, 'rb') as f:
            graphs, labels = pkl.load(f)
        self.num_classes = 3
        self.graphs = graphs            
        self.labels = labels
        self.dim_nfeats = 10
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

class MutagenicityDataset(DGLDataset):
    def __init__(self, raw_dir="", neg_ratio=1.):
        super(MutagenicityDataset, self).__init__('Mutagenicity')
        with open(raw_dir, 'rb') as f:
            graphs, labels = pkl.load(f)
        self.num_classes = 2
        graphs1, graphs2, graphs3 = [], [], []
        for g, l in zip(graphs, labels):
            g.ndata['attr'] = self.create_node_features(g.ndata['node_labels'].to(torch.int64), 14)
            if l == 0:
                if g.edata['edge_labels'].sum() > 0:
                    graphs2.append(g)
                else:
                    graphs3.append(g)
            elif l == 1:
                graphs1.append(g)
        random.shuffle(graphs1)
        random.shuffle(graphs3)
        l = len(graphs2)
        r = 0
        if neg_ratio != 1.:
            l *= neg_ratio
            l = int(l)
            if l > len(graphs1):
                r = l - len(graphs1)
                l = len(graphs1)
        labels2 = [0] * len(graphs2) + [1] * l
        self.graphs = graphs2 + graphs1[:l]
        if r:
            labels2 += [0] * r
            self.graphs += graphs3[:r]
        self.labels = torch.LongTensor(labels2)
        self.dim_nfeats = 14
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def create_node_features(self, labels, num_labels=14):
        return F.one_hot(labels, num_classes=num_labels).float()