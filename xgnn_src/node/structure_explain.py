import os
import argparse
import torch
import torch.functional as F
import pickle as pkl
import dgl
from dgl.nn.functional import edge_softmax
from tqdm import tqdm

from utils import load_data, personalized_pagerank
from parser_args import *

def main(args, transition_matrix, g):
    with g.local_scope():
        preferences = torch.LongTensor([args.target_node])
        gt = dgl.reverse(g, False, True)
        personalized_pagerank(gt, 1, 0.85, preferences, True, transition_matrix)
        sampling_probs = gt.ndata['pv']
    # measure_nodes = torch.topk(sampling_probs, args.topk)
    print(sampling_probs)
    

def add_explain_args(parser):
    parser.add_argument("--target-node", help="Node to explain", type=int)
    parser.add_argument("--topk", type=int, default=10,
                        help="Select K important nodes (neighbors)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainer')
    add_common_args(parser)
    add_explain_args(parser)
    args = parser.parse_args()
    print(args)
    
    g, n_classes = load_data(args)
    if args.self_loop:
        print("using self loop")
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    in_feats = g.ndata['feat'].size(-1)

    # load student
    transition_matrix = None
    if os.path.exists(args.student_graph):
        with open(args.student_graph, 'rb') as f:
            sg = pkl.load(f)
            if not args.self_loop:
                sg = dgl.remove_self_loop(sg)
            transition_matrix = sg.edata['weight'].detach()
            adj = edge_softmax(sg, transition_matrix)
            
    main(args, adj, g)