import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from xgnn_src.graph.dataloader import GCDataLoader
from xgnn_src.graph.parser_args import Parser
from xgnn_src.graph.gin import GIN
from xgnn_src.graph.gcn import GCN, GCN_MLP2
from xgnn_src.graph.gat import GAT
from xgnn_src.graph.utils import load_data

def train(args, net, trainloader, optimizer, criterion, epoch):
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
        outputs, _ = net(graphs, feat)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

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


def eval_net(args, net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        total += len(labels)
        outputs, _ = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc


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

    dataset, dim_nfeats, gclasses = load_data(args.dataset, args.datapath, args.learn_eps, args.degree_as_nlabel)
    trainloader, validloader = GCDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=True,
        split_name=args.split_name, fold_idx=args.fold_idx).train_valid_loader()
    # or split_name='rand', split_ratio=0.7

    if args.model_name == 'gin':
        model = GIN(
            args.num_layers, args.num_mlp_layers,
            dim_nfeats, args.hidden_dim, gclasses,
            args.final_dropout, args.learn_eps,
            args.graph_pooling_type, args.neighbor_pooling_type,
            args.linear_pooling_type).to(args.device)
    elif args.model_name == 'gcn_mlp':
        model = GCN_MLP2(dim_nfeats, args.hidden_dim, gclasses,
                args.num_layers, args.final_dropout, args.hidden_dim * 2, args.graph_pooling_type,
                args.linear_pooling_type, norm_type='bn', graph_dropout=args.final_dropout).to(args.device)
    elif args.model_name == "gat":
        heads = None
        if args.num_layers:
            heads = [3] * args.num_layers
        model = GAT(args.num_layers, dim_nfeats, args.hidden_dim, gclasses, heads,
                    args.final_dropout, args.final_dropout, 0.2).to(args.device)
    else:
        model = GCN(dim_nfeats, args.hidden_dim, gclasses,
                args.num_layers, args.final_dropout, args.graph_pooling_type,
                args.linear_pooling_type).to(args.device)
    print(model)
    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        train_loss, train_acc = eval_net(
            args, model, trainloader, criterion)
        tbar.set_description(
            'train set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(train_loss, 100. * train_acc))

        valid_loss, valid_acc = eval_net(
            args, model, validloader, criterion)
        vbar.set_description(
            'valid set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(valid_loss, 100. * valid_acc))

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type,
                    epoch
                ))
                f.write("\n")
                f.write("%f %f %f %f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))
                f.write("\n")
        if (epoch + 1) < args.epochs and (epoch + 1) % 50 == 0 and args.model_path:
            torch.save(model.state_dict(), args.model_path)
    torch.save(model.state_dict(), args.model_path)
        # lrbar.set_description(
        #     "Learning eps with learn_eps={}: {}".format(
        #         args.learn_eps, [layer.eps.data.item() for layer in model.ginlayers]))

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)
