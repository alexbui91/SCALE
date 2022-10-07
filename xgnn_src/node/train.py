from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
import argparse, time, os, math
import numpy as np
import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

import pickle as pkl

import dgl
from dgl.data import register_data_args

from xgnn_src.utils import SoftCrossEntropyLoss
from xgnn_src.shared_networks import MLP
from xgnn_src.node.student import GCN_LPA, SGAT, GAT, EGNN
from xgnn_src.node.utils import evaluate, init_teacher, load_data, make_slice_l2
from xgnn_src.node.parser_args import *

def store_student_graph(path, student):
    graph = student.get_graph()
    with open(path, 'wb') as f:
        pkl.dump(graph, f)

def balance_training(labels):
    # only work with yelp & amazon dataset
    train_pos_idx = labels.nonzero().flatten() 
    train_neg_idx = (labels == 0).nonzero().flatten()
    # boostrapping
    pos_idx_bt = np.random.choice(train_pos_idx, (len(train_neg_idx),), replace=True)
    selected_bg_idx = np.concatenate([pos_idx_bt, train_neg_idx])
    selected_bg_idx.sort()
    return torch.LongTensor(selected_bg_idx)

def train_student_batch(student, train_features, hard_labels, soft_labels,
                        loss_fcn, soft_loss_fcn, optimizer, sl_factor=1., batch_size=32, n_epochs=200):
    balance_idx = balance_training(hard_labels.cpu()).to(train_features.device)
    bt_features, bt_hard_labels, bt_soft_labels = train_features[balance_idx], hard_labels[balance_idx], soft_labels[balance_idx]
    l = len(bt_features)
    num_batches = int(math.ceil(l / batch_size))
    for epoch in range(n_epochs):
        student.train()
        # forward
        perms = torch.randperm(l, device=train_features.device)
        epoch_features, epoch_labels, epoch_soft_lbs = bt_features[perms,:], bt_hard_labels[perms], bt_soft_labels[perms]
        epoch_loss = 0.
        for b in range(num_batches):
            st = b * batch_size
            ed = st + batch_size
            if ed > l:
                ed = l
            batch_features = epoch_features[st:ed,:]
            logits = student(batch_features)
            # loss to hard labels
            batch_labels = epoch_labels[st:ed]
            loss = loss_fcn(logits, batch_labels) 
            # loss to soft labels
            if sl_factor:
                soft_batch_lb = epoch_soft_lbs[st:ed]
                soft_loss = soft_loss_fcn(logits, soft_batch_lb)
                loss += sl_factor * soft_loss
                               
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print("Training student epoch %i" % epoch, epoch_loss)

def train_student_trans(student, student_type, features, hard_train_mask, hard_train_labels, soft_logits,
                        loss_fcn, soft_loss_fcn, optimizer, sl_factor=1., lpa_factor=10., n_epochs=200, store_embedding=False):
    student.train()
    # writer = SummaryWriter('./student')
    for epoch in range(n_epochs):
        # forward        
        if student_type == "lpa":
            logits, p_labels = student(features)
        else:
            logits = student(features)
            p_labels = None
        # loss to hard labels
        hard_loss = loss_fcn(logits[hard_train_mask], hard_train_labels) 
        if not p_labels is None and lpa_factor:
            hard_loss += lpa_factor * loss_fcn(p_labels[hard_train_mask], hard_train_labels)        
        
        loss = hard_loss
        # loss to soft labels
        if sl_factor:
            soft_loss = soft_loss_fcn(logits, soft_logits)
            print("soft loss %i" % epoch, soft_loss.item())
            loss += sl_factor * soft_loss
                
        # writer.add_scalars('loss', {'total_loss': loss.item(), 'hard_loss': hard_loss.item(), 'soft_loss': soft_loss.item()}, epoch)
        print("Training student epoch %i" % epoch, loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if store_embedding:
        student.eval()
        if student_type == "lpa":
            logits, _ = student(features)
        else:
            logits = student(features)
        return logits
    return None

def train_student(teacher, g, args, in_feats, num_classes):
    s = time.time()
    teacher.eval()
    if args.student_type == "lpa":
        student = GCN_LPA(g, in_feats, args.n_hidden, num_classes, args.n_layers,
                        F.relu, args.dropout, args.n_lpa, args.slb)
    elif args.student_type == "sgat":
        student = SGAT(g, args.n_layers, in_feats, args.n_hidden, num_classes,
                        F.elu, args.dropout, args.dropout, args.negative_slope)
    elif args.student_type == "gat":
        student = GAT(g, args.n_layers, in_feats, args.n_hidden, num_classes,
                        F.elu, args.dropout, args.dropout, args.negative_slope)
    elif args.student_type == "egnn":
        g1 = make_slice_l2(g).to(g.device)
        student = EGNN(g, g1, in_feats, args.n_hidden, num_classes, args.dropout)
    elif args.student_type == "mgcn":
        student = GCN_MLP(g, g.ndata['feat'], in_feats, args.n_hidden, num_classes, args.n_layers, F.relu, args.dropout)
    else:
        student = MLP(in_feats, args.std_hiddens, num_classes, F.relu, args.dropout)

    print(args.student_type, student)
    if args.gpu >= 0:
        student = student.to(args.gpu)

    features = g.ndata['feat']
    hard_labels = g.ndata['label']
    hard_train_mask = g.ndata['train_mask']
    # hard_val_mask = g.ndata['val_mask']
    hard_train_labels = hard_labels[hard_train_mask]

    if args.dataset in ["yelp", "amazon"]:
        num_train = len(hard_labels)
        train_pos_idx = hard_labels.nonzero().flatten()
        num_pos = len(train_pos_idx)
        num_neg = num_train - num_pos
        pos_weight = num_train / (2 * num_pos)
        neg_weight = num_train / (2 * num_neg)
        loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight]).to(features.device))
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()

    soft_loss_fcn = SoftCrossEntropyLoss()
    # soft_loss_fcn = KLLoss()
    # use optimizer
    optimizer = torch.optim.Adam(student.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    with torch.no_grad():
        soft_logits = teacher(features).detach()
    
    # or not args.student_type in ["mlp", "graphsage"]
    if args.student_type == "mlp" and args.dataset in ["yelp", "amazon"]:
        print("minibatch training")
        train_student_batch(student, features[hard_train_mask], hard_train_labels, soft_logits[hard_train_mask], torch.nn.CrossEntropyLoss(),
                           soft_loss_fcn, optimizer, args.sl_factor, batch_size=args.batchsize, n_epochs=args.n_epochs)
    else:
        print("transductive training")
        logits = train_student_trans(student, args.student_type, features, hard_train_mask, hard_train_labels, soft_logits,
                                loss_fcn, soft_loss_fcn, optimizer, args.sl_factor, args.lpa_factor, args.n_epochs, args.std_emb)
        if args.student_type == 'mgcn':
            adj = student.get_adj()
            print(adj.numpy().tolist())
            print(adj.mean(), adj.std())
        if not logits is None:
            g.ndata['emb'] = logits.detach()
        
    acc = evaluate(student, features, hard_labels, g.ndata['test_mask'], mode='student', evaltype=args.eval_type)
    print("Student test accuracy {:.2%}".format(acc))
    if args.student_graph:
        store_student_graph(args.student_graph, student)

    if args.student_model:
        torch.save(student.state_dict(), args.student_model)

    print("Time for training student", time.time() - s)

def train_teacher(args):
    g, n_classes = load_data(args.dataset, args.graph_type, args.n_classes, skip_features=args.skip_features, add_reverse=args.add_reverse)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        print("using self loop")
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    
    # create teacher model
    teacher = init_teacher(args, g, in_feats, n_classes)
    train_labels = labels[train_mask]
    if cuda:
        teacher = teacher.to(g.device)
        train_labels = train_labels.to(g.device)

    if args.dataset in ["yelp", "amazon"]:
        num_train = len(train_labels)
        train_pos_idx = train_labels.nonzero().flatten()
        num_pos = len(train_pos_idx)
        num_neg = num_train - num_pos
        pos_weight = num_train / (2 * num_pos)
        neg_weight = num_train / (2 * num_neg)
        loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight]).cuda())
    else:
        loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(teacher.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    dur = []
    print("#trained label ratio", train_mask.sum() / len(labels))
    for epoch in range(args.n_epochs):
        teacher.train()
        if epoch >= 3:
            t0 = time.time()
        # forward            
        logits = teacher(features)
        loss = loss_fcn(logits[train_mask], train_labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(teacher, features, labels, val_mask, evaltype=args.eval_type)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    acc = evaluate(teacher, features, labels, test_mask, evaltype=args.eval_type)
    print("Test accuracy {:.2%}".format(acc))
    if args.teacher_pretrain:
        torch.save(teacher.state_dict(), args.teacher_pretrain)

        with open(args.teacher_pretrain + '_.g', 'wb') as f:
            pkl.dump(g.cpu(), f)

    return teacher, g, n_classes
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--load", action='store_true',
            help="Load pretrained teacher (default=False)")
    add_common_args(parser)
    add_student_lpa_args(parser)
    add_appnp_args(parser)
    add_gat_args(parser)
    args = parser.parse_args()
    print(args)
    
    if not args.load:
        if args.teacher_pretrain:
            with open(args.teacher_pretrain + "_args", 'wb') as f:
                pkl.dump(args, f)
        teacher, g, n_classes = train_teacher(args)
        in_feats = g.ndata['feat'].size(-1)
    else:
        teacher_args = deepcopy(args)
        args_path = args.teacher_pretrain + "_args"
        path = args.teacher_pretrain
        if os.path.exists(args_path):
            with open(args_path, 'rb') as f:
                teacher_args = pkl.load(f)
        model = torch.load(path)
        g, n_classes = load_data(teacher_args.dataset, teacher_args.graph_type, teacher_args.n_classes, skip_features=args.skip_features, add_reverse=args.add_reverse)
        if teacher_args.self_loop:
            print("using self loop")
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        in_feats = g.ndata['feat'].size(-1)
        teacher_args.gpu = args.gpu
        if teacher_args.gpu >= 0:
            g = g.to(teacher_args.gpu)
        teacher = init_teacher(teacher_args, g, in_feats, n_classes)
        teacher.load_state_dict(model)
        if teacher_args.gpu >= 0:
            teacher = teacher.to(teacher_args.gpu)
    
    if args.surrogate:
        train_student(teacher, g, args, in_feats, n_classes)


