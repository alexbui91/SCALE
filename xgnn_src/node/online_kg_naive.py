import argparse, time
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from student import GCN_LPA, MLP
from utils import accuracy, evaluate, init_teacher, load_data, eval_student
from parser_args import *

class ComponentAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ComponentAttention, self).__init__()
        self.weight = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, logits):
        h = self.weight(logits)
        h = F.softmax(h, 1)
        h = torch.swapaxes(h, 1, 2)
        h = torch.matmul(h, logits)
        return torch.squeeze(h)

class OnlineKG(nn.Module):
    def __init__(self, teacher, lpa, mlp, att):
        super(OnlineKG, self).__init__()
        self.teacher = teacher
        self.lpa = lpa
        self.mlp = mlp
        self.att = att

    def forward(self, features):
        lpa_logits, p_labels = self.lpa(features)
        mlp_logits = self.mlp(features)
        a_logits = self.teacher(features)

        combined_logits = torch.cat((a_logits.unsqueeze(1), lpa_logits.unsqueeze(1), mlp_logits.unsqueeze(1)), 1)
        z_e = self.att(combined_logits)
        # z_e = combined_logits.mean(1).squeeze()
        return a_logits, (lpa_logits, p_labels), mlp_logits, z_e

def main(args):
    g, num_classes = load_data(args.dataset)
    if args.gpu >= 0:
        g = g.to(args.gpu)
    # init GCN_LPA student
    in_feats = g.ndata['feat'].size()[-1]
    lpa = GCN_LPA(g, in_feats, args.n_hidden, num_classes, args.n_layers,
                        F.relu, args.dropout, args.n_lpa, args.slb)
    # init MLP student
    # mlp = MLP(in_feats, args.std_hiddens, num_classes, F.relu, args.dropout) # regularly, paper submitted
    mlp = MLP(in_feats, args.std_hiddens, num_classes, F.relu, args.dropout, batch_norm=True) # 
    # init APPNP teacher
    appnp = init_teacher(args, g, in_feats, num_classes)
    com_att = ComponentAttention(num_classes)
    online_kg = OnlineKG(appnp, lpa, mlp, com_att)

    optimizer = torch.optim.Adam(online_kg.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    train_mask = g.ndata['train_mask']
    labels = g.ndata['label']
    if args.gpu >= 0:
        online_kg = online_kg.to(args.gpu)
    ce_loss = CrossEntropyLoss()
    kl_loss = KLDivLoss(log_target=True)
    # sce_loss = SoftCrossEntropyLoss(args.temp)
    st = time.time()
    for i in range(args.n_epochs):
        online_kg.train()

        a_logits, (lpa_logits, p_labels), mlp_logits, z_e = online_kg(g.ndata['feat'])
        lpa_loss = ce_loss(lpa_logits[train_mask], labels[train_mask])
        lpa_loss += args.lpa_factor * ce_loss(p_labels[train_mask], labels[train_mask])   
        mlp_loss = ce_loss(mlp_logits[train_mask], labels[train_mask])
        a_loss = ce_loss(a_logits[train_mask], labels[train_mask])
        z_e = F.softmax(z_e / args.temp, 1)
        soft_llogits = F.softmax(lpa_logits / args.temp, 1)
        kll = kl_loss(soft_llogits, z_e) 
        soft_alogits = F.softmax(a_logits / args.temp, 1)
        kla = kl_loss(soft_alogits, z_e) 
        soft_mlogits = F.softmax(mlp_logits / args.temp, 1)
        klmc = kl_loss(soft_mlogits, z_e) 
        if args.kl_mlp:
            klm2 = kl_loss(soft_mlogits, soft_alogits) 
            klm3 = kl_loss(soft_mlogits, soft_llogits) 
            klmc += klm2 + klm3
            klmc *= args.kl_mlp
        kl = kll + kla + klmc
        loss = lpa_loss + a_loss + mlp_loss + args.temp * args.temp * kl
        optimizer.zero_grad()
        loss.backward()
        print("epoch %i" % i, loss.item())
        optimizer.step()
    
    du = time.time() - st
    print("Training time", du)
    online_kg.eval()
    test_mask = g.ndata['test_mask']
    with torch.no_grad():
        a_logits, (lpa_logits, _), mlp_logits, _ = online_kg(g.ndata['feat'])
        a_acc = accuracy(a_logits[test_mask], labels[test_mask])
        l_acc = accuracy(lpa_logits[test_mask], labels[test_mask])
        m_acc = accuracy(mlp_logits[test_mask], labels[test_mask])
        print("Teacher accuracy:", a_acc)
        print("GCN-LPA accuracy:", l_acc)
        print("MLP accuracy:", m_acc)

        k1, a1 = eval_student(a_logits, lpa_logits)
        print("LPA to teacher", k1.item(), a1)
        k2, a2 = eval_student(a_logits, mlp_logits)
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





    
