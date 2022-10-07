import sys, os
from copy import deepcopy
import pickle as pkl

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from xgnn_src.graph.dataloader import GCDataLoader
from xgnn_src.graph.graphsage import GraphSAGE, GraphSAGE_MLP
from xgnn_src.graph.parser_args import Parser
from xgnn_src.graph.gin import GIN, GIN_MLP
from xgnn_src.graph.gcn import GCN, GCN_MLP, GCN_MLP2
from xgnn_src.shared_networks import NaiveTeacher, Ensemble, OnlineKG
from xgnn_src.graph.utils import SoftCrossEntropyLoss, load_data
from xgnn_src.utils import imbalanceCriteria

def update_ema_variables(model, ema_model, alpha=0.999, global_step=-1):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def soft_logits(logits, term):
    return [F.softmax(l / term, 1) for l in logits]

def naive_strategy(e_logits, b_logits, criteria):
    # base model only updates its params based ce loss
    # explainer updates its params based on ce & kd to base
    s_e_loss = criteria(e_logits, b_logits)
    return s_e_loss

def ensemble_strategy(b_logits, e_logits, t_logits, term, criteria):
    s_b_logits, s_e_logits, s_t_logits = soft_logits([b_logits, e_logits, t_logits], term)
    s_b_loss = criteria(s_b_logits, s_t_logits)
    s_e_loss = criteria(s_e_logits, s_t_logits)
    s_loss = s_b_loss + s_e_loss
    return s_loss

def peer_strategy(base_h, ex_h, base_z, ex_z, labels, ensemble_model, kl_loss, ce_loss, temp=2):
    if type(base_h) == list:
        h = []
        for b, e in zip(base_h, ex_h):
            h.append(torch.cat([b, e], dim=-1))
    else:
        h = torch.cat([base_h, ex_h], dim=-1)
    z_t = ensemble_model(h)
    p_loss = ce_loss(z_t, labels)
    return ensemble_strategy(base_z, ex_z, z_t, temp, kl_loss), p_loss    

def train(args, net, trainloader, optimizer, ce_loss, kd_loss, epoch, ensemble_model=None, avg_model=None):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)
    for pos, (graphs, labels) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        graphs = graphs.to(args.device)
        feat = graphs.ndata.pop('attr')
        b_logits, e_logits, t_logits, b_h, e_h = net(graphs, feat, beta=args.beta, use_norm_adj=args.use_norm_adj)
        h_b_loss = ce_loss(b_logits, labels)
        h_e_loss = ce_loss(e_logits, labels)
        loss = h_b_loss + h_e_loss
        if args.kd_strategy == 'ensemble_model':
            # g = epoch * args.batch_size + pos
            s_loss, p_loss = peer_strategy(b_h, e_h, b_logits, e_logits, labels, ensemble_model, kd_loss, ce_loss)
            loss += p_loss
        
        elif args.kd_strategy == 'ensemble':
            h_t_loss = ce_loss(t_logits, labels)
            loss += h_t_loss
            s_loss = ensemble_strategy(b_logits, e_logits, t_logits, args.temp, kd_loss)
        else:
            b_logit_news = b_logits.clone().detach()
            s_loss = naive_strategy(e_logits, b_logit_news, kd_loss)    
        loss += s_loss * args.kl_term
        if args.sl_term > 0:
            size_loss = net.get_size_loss()
            if args.budget > 0:
                size_loss = F.relu(size_loss - args.budget)
            loss += size_loss * args.sl_term # size loss
        if args.mk_term > 0:
            mask_loss = net.get_mask_loss()
            loss += args.mk_term * mask_loss
        running_loss = loss.item()
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, num_models=3):
    net.eval()
    total = 0
    total_correct = [0] * num_models

    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        total += len(labels)
        outputs = net(graphs, feat, args.beta, args.use_norm_adj)
        for i in range(num_models):
            l = outputs[i]
            _, predicted = torch.max(l.data, -1)
            total_correct[i] += (predicted == labels.data).sum().item()

    acc = [1.0*v / total for v in total_correct]
    net.train()

    return acc

def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    dataset, dim_nfeats, gclasses = load_data(args.dataset, args.datapath, args.learn_eps, args.degree_as_nlabel, neg_ratio=args.neg_ratio)

    if args.load_data:
        print('Reload data from', args.load_data)
        with open(args.load_data, 'rb') as f:
            dataloader, _ = pkl.load(f)
    else:
        dataloader = GCDataLoader(
            dataset, batch_size=args.batch_size, device=args.device,
            seed=args.seed, shuffle=True,
            split_name=args.split_name, fold_idx=args.fold_idx,
            split_ratio=args.split_ratio)
        idx = dataloader.train_valid_idx()
        dataloader = dataloader.train_valid_loader()
        if args.store_data:
            with open(args.store_data, 'wb') as f:
                pkl.dump((dataloader, idx), f)
    trainloader, validloader = dataloader
    # train_idx, val_idx = idx
    # or split_name='rand', split_ratio=0.7
    if args.model_name == 'gcn':
        base = GCN(dim_nfeats, args.hidden_dim, gclasses,
                args.num_layers, args.graph_dropout, args.graph_pooling_type,
                args.linear_pooling_type).to(args.device)
        explainer = GCN_MLP(dim_nfeats, args.hidden_dim, gclasses,
                args.num_layers, args.final_dropout, args.hidden_dim * 2, args.graph_pooling_type,
                args.linear_pooling_type, adj_sym=args.sym, norm_type=args.norm_type, graph_dropout=args.graph_dropout).to(args.device)
    elif args.model_name == 'gcn2':
        base = GCN(dim_nfeats, args.hidden_dim, gclasses,
                args.num_layers, args.graph_dropout, args.graph_pooling_type,
                args.linear_pooling_type).to(args.device)
        explainer = GCN_MLP2(dim_nfeats, args.hidden_dim, gclasses,
                args.num_layers, args.final_dropout, args.hidden_dim * 2, args.graph_pooling_type,
                args.linear_pooling_type, adj_sym=args.sym, norm_type=args.norm_type, graph_dropout=args.graph_dropout).to(args.device)
    elif args.model_name == 'graphsage':
        base = GraphSAGE(dim_nfeats, args.hidden_dim, gclasses, args.num_layers, F.relu,
                args.final_dropout, 'mean', args.graph_pooling_type)
        explainer = GraphSAGE_MLP(dim_nfeats, args.hidden_dim, gclasses, args.num_layers, F.relu,
                args.final_dropout, 'mean', args.graph_pooling_type)
    else:
        base = GIN(args.num_layers, args.num_mlp_layers,
                dim_nfeats, args.hidden_dim, gclasses,
                args.final_dropout, args.learn_eps,
                args.graph_pooling_type, args.neighbor_pooling_type,
                args.linear_pooling_type).to(args.device)
        explainer = GIN_MLP(args.num_layers, args.num_mlp_layers,
                dim_nfeats, args.hidden_dim, gclasses,
                args.final_dropout, args.learn_eps,
                args.graph_pooling_type, args.neighbor_pooling_type,
                args.linear_pooling_type, args.hidden_dim*2,
                adj_sym=args.sym, norm_type=args.norm_type).to(args.device)
    print("base model", base)
    print("explainer", explainer)
    teacher = NaiveTeacher(gclasses, 'mean').to(args.device)

    # avg_model = deepcopy(base)
    avg_model, ensemble = None, None
    if args.kd_strategy == 'ensemble_model':
        if args.linear_pooling_type == 'last':
            ensemble = Ensemble([args.hidden_dim*2], gclasses, args.final_dropout, 'mean').to(args.device)
        else:
            ensemble = Ensemble([dim_nfeats*2] + [args.hidden_dim*2]*(args.num_layers-1),
                                gclasses, args.final_dropout, 'mean').to(args.device)

    online_kg = OnlineKG(base, explainer, teacher).to(args.device)
    if args.load_from:
        print('Reload model from', args.load_from)
        model = torch.load(args.load_from)
        online_kg.load_state_dict(model)
    ce_loss = nn.CrossEntropyLoss()  # defaul reduce is true
    if args.kd_strategy == 'naive':
        kd_loss = SoftCrossEntropyLoss(args.temp, 'mean')
    else:
        kd_loss = nn.KLDivLoss()
    optimizer = optim.Adam(online_kg.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)

    for epoch in vbar:
        train(args, online_kg, trainloader, optimizer, ce_loss, kd_loss, epoch, ensemble, avg_model)
        scheduler.step()
        v1, v2, v3 = eval_net(args, online_kg, validloader)
        vbar.set_description('valid set - base acc: {:.2f}% | explainer acc: {:0.2f}% | teacher acc: {:0.2f}%'
                            .format(100. * v1, 100. * v2, 100. * v3))
        if (epoch + 1) % 50 == 0:
            torch.save(online_kg.state_dict(), args.model_path)

    vbar.close()

    torch.save(online_kg.state_dict(), args.model_path)


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)
