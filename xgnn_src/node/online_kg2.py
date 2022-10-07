import warnings
warnings.filterwarnings("ignore")

import argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from xgnn_src.node.teacher import GraphSAGE

from xgnn_src.utils import SoftCrossEntropyLoss, imbalanceCriteria
from xgnn_src.shared_networks import MLP
from xgnn_src.node.utils import accuracy, init_teacher, init_graph_student, load_data, eval_student, calculate_score
from xgnn_src.node.parser_args import *

class AllOnlineKG(nn.Module):
    def __init__(self, base, graph_explainer, mlp_explainer, teacher=None, graph_student_name=""):
        super(AllOnlineKG, self).__init__()
        self.teacher = teacher
        self.base = base
        self.graph_x = graph_explainer
        self.mlp_x = mlp_explainer
        self.graph_student_name = graph_student_name

    def forward(self, g, features):
        base_logits = self.base(g, features)
        p_labels = None
        if self.graph_student_name != "lpa":
            if self.training:
                self.base.eval()
                with torch.no_grad():
                    with g.local_scope():
                        self.base(g, features)
                        node_embeddings = g.ndata['emb'].clone().detach()
                self.base.train()
            else:
                node_embeddings = g.ndata['emb'].clone().detach()
            with g.local_scope():
                ex_logits = self.graph_x(g, features, node_embeddings)
        else:
            ex_logits, p_labels = self.graph_x(g, features)
        mlp_logits = self.mlp_x(features)
        t_logits = None
        if not self.teacher is None:
            t_logits = self.teacher([base_logits, ex_logits, mlp_logits])
        return base_logits, ex_logits, mlp_logits, t_logits, p_labels

def eval(model, g, mask, dataset=""):
    model.eval()
    labels = g.ndata['label']
    with torch.no_grad():
        base_logits, ex_logits, mlp_logits, _, p_labels = model(g, g.ndata['feat'])
        if not dataset in ["amazon", "yelp"]:
            b_acc = accuracy(base_logits[mask], labels[mask])
            g_acc = accuracy(ex_logits[mask], labels[mask])
            m_acc = accuracy(mlp_logits[mask], labels[mask])
            print("Teacher accuracy:%.2f, Graph-based accuracy: %.2f, MLP accuracy: %.2f" % (b_acc, g_acc, m_acc))
        else:
            b = calculate_score(base_logits[mask], labels[mask], "rec")
            g = calculate_score(ex_logits[mask], labels[mask], "rec")
            m = calculate_score(mlp_logits[mask], labels[mask], "rec")
            print("Teacher score:%.2f, Graph-based score: %.2f, MLP score: %.2f" % (b, g, m))

def main(args):
    g, num_classes = load_data(args.dataset, args.graph_type, add_reverse=args.add_reverse)
    if args.self_loop:
        g = g.remove_self_loop()
        g = g.add_self_loop()
    # init GCN_LPA student
    in_feats = g.ndata['feat'].size()[-1]
    base = init_teacher(args, g, in_feats, num_classes)
    # init graph student
    graph_x = init_graph_student(args.student_type, g, in_feats, num_classes, args.dropout, n_hidden=args.n_hidden,
                n_layers=args.n_layers, hidden_sizes=args.hidden_sizes, n_lpa=args.n_lpa, slb=args.slb, all_layer_dp=args.all_layer_dp, skip_norm=args.skip_norm)
    # init MLP student
    mlp = MLP(in_feats, args.std_hiddens, num_classes, F.relu, args.dropout, batch_norm=True, norm_type='bn') # 
    # init APPNP teacher
    print(graph_x, mlp)
    # teacher = NaiveTeacher(args.n_hidden)
    online_kg = AllOnlineKG(base, graph_x, mlp, None, args.student_type)

    optimizer = torch.optim.Adam(online_kg.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.gpu >= 0:
        g  = g.to(args.gpu)
        online_kg = online_kg.to(args.gpu)
    train_mask = g.ndata['train_mask']
    labels = g.ndata['label']
    train_labels = labels[train_mask]
    if args.dataset in ["amazon", "yelp", "BAC"]:
        ce_loss = imbalanceCriteria(labels, args.gpu, num_classes)
    else:
        ce_loss = CrossEntropyLoss()
    sce_loss = SoftCrossEntropyLoss(args.temp)
    st = time.time()
    for i in range(args.n_epochs):
        online_kg.train()
        base_logits, ex_logits, mlp_logits, _, p_labels = online_kg(g, g.ndata['feat'])
        if (i + 1) % 10 == 0:
            eval(online_kg, g, g.ndata['val_mask'], args.dataset)
        # soft loss from students to teachers
        base_logits_new =  base_logits.clone().detach()
        soft_gx_loss = sce_loss(ex_logits, base_logits_new)
        soft_mlp_loss = sce_loss(mlp_logits, base_logits_new)
        kl_loss = soft_mlp_loss + soft_gx_loss

        base_logits = base_logits[train_mask]
        ex_logits = ex_logits[train_mask]
        mlp_logits = mlp_logits[train_mask]
        # hard loss
        base_loss = ce_loss(base_logits, train_labels)
        ex_loss = ce_loss(ex_logits, train_labels)
        mlp_loss = ce_loss(mlp_logits, train_labels)
        hard_loss = base_loss + ex_loss + mlp_loss

        loss = hard_loss + kl_loss * args.sl_factor
        if not p_labels is None:
            loss += args.lpa_factor * ce_loss(p_labels[train_mask], train_labels)

        optimizer.zero_grad()
        loss.backward()
        print("epoch %i" % i, loss.item())
        optimizer.step()
  
    if args.teacher_pretrain:
        print("saving ckpt")
        torch.save(online_kg.state_dict(), args.teacher_pretrain)

    du = time.time() - st
    print("Training time", du)
    online_kg.eval()
    test_mask = g.ndata['test_mask']
    with torch.no_grad():
        base_logits, ex_logits, mlp_logits, _, _ = online_kg(g, g.ndata['feat'])
        if args.dataset not in ["amazon", "yelp"]:
            b_acc = accuracy(base_logits[test_mask], labels[test_mask])
            g_acc = accuracy(ex_logits[test_mask], labels[test_mask])
            m_acc = accuracy(mlp_logits[test_mask], labels[test_mask])
            print("Teacher accuracy:", b_acc)
            print("Graph-based model accuracy:", g_acc)
            print("MLP accuracy:", m_acc)
        else:
            b = calculate_score(base_logits[test_mask], labels[test_mask], "rec")
            g = calculate_score(ex_logits[test_mask], labels[test_mask], "rec")
            m = calculate_score(mlp_logits[test_mask], labels[test_mask], "rec")
            print("Teacher score", b)
            print("Graph-based model score", g)
            print("MLP score", m)

        k1, a1 = eval_student(base_logits, ex_logits)
        print("GNN Student to teacher", k1.item(), a1)
        k2, a2 = eval_student(base_logits, mlp_logits)
        print("MLP to teacher", k2.item(), a2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--load", action='store_true',
            help="Load pretrained teacher (default=False)")
    parser.add_argument("--kl-mlp", type=float, default=1, help="0 mean doesn't take kmlc")
    add_common_args(parser)
    add_student_lpa_args(parser)
    add_appnp_args(parser)
    add_gat_args(parser)
    args = parser.parse_args()
    print(args)

    main(args)