import os
import argparse
import torch
import torch.functional as F
import pickle as pkl
import dgl
from dgl.nn.functional import edge_softmax
from tqdm import tqdm

from utils import init_teacher, load_data, personalized_pagerank
from parser_args import *
from shapley import mc_sampling


def main(args, teacher, transition_matrix, g):
    with g.local_scope():
        preferences = torch.LongTensor([args.target_node])
        gt = dgl.reverse(g, False, True)
        personalized_pagerank(gt, 1, 0.85, preferences, True, transition_matrix)
        sampling_probs = gt.ndata['pv']
    measure_nodes = torch.topk(sampling_probs, args.topk)
    revenues = mc_sampling(teacher, g, args.target_node, measure_nodes, sampling_probs, args.sample_size, args.cls, args.sample_num)
    print(revenues)
    with g.local_scope():
        feat = g.ndata['feat']
        logits = teacher(feat, g=g)
        logit_v = torch.softmax(logits[args.target_node, :], 0)
        if args.cls == -1:
            args.cls = torch.argmax(logit_v).item()
        print("pred logit of %i:" % args.cls, logit_v[args.cls].item())
    return revenues

def add_explain_args(parser):
    parser.add_argument("--target-node", help="Node to explain", type=int)
    parser.add_argument("--topk", type=int, default=10,
                        help="Select K important nodes (neighbors)")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Size of background samples to compute value function")
    parser.add_argument("--sample-num", type=int, default=200,
                        help="Number of Monte Carlo Iterations")
    parser.add_argument("--cls", type=int, default=-1,
                        help="Class to explain")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainer')
    add_common_args(parser)
    add_appnp_args(parser)
    add_gat_args(parser)
    add_explain_args(parser)
    args = parser.parse_args()
    print(args)
    
    if os.path.exists(args.teacher_pretrain + "_args"):
        with open(args.teacher_pretrain + "_args", 'rb') as f:
            args = pkl.load(f)

    model_ckpt = torch.load(args.teacher_pretrain)
    g, n_classes = load_data(args)
    if args.self_loop:
        print("using self loop")
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    in_feats = g.ndata['feat'].size(-1)
    teacher = init_teacher(args, g, in_feats, n_classes)
    teacher.load_state_dict(model_ckpt)

    # load student
    transition_matrix = None
    if os.path.exists(args.student_graph):
        with open(args.student_graph, 'rb') as f:
            sg = pkl.load(f)
            if not args.self_loop:
                sg = dgl.remove_self_loop(sg)
            transition_matrix = sg.edata['weight'].detach()
            adj = edge_softmax(sg, transition_matrix)
            
    main(args, teacher, adj, g)