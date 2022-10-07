from copy import deepcopy
from tqdm import tqdm
import dgl
import torch
import torch.nn.functional as F
import dgl.nn.functional as fn
import numpy as np
from xgnn_src.node.utils import personalized_pagerank
from xgnn_src.node.draw import draw_simple_graph
from sklearn.metrics import precision_score, recall_score, roc_curve

def scaler(v): # for drawing only
    mn = v.min()
    mx = v.max()
    if mx != mn:
        v = (v - mn) / (mx - mn)
    return v * 0.9 + 0.1

def get_node_map(g, lb, threshold=2, mask=None):
    map_nodes = {}
    labels = g.ndata['label']
    nodes = g.nodes()
    if not mask is None:
        labels = labels[mask]
        nodes = nodes[mask]
    
    degrees = g.in_degrees()
    for l, n, d in zip(labels, nodes, degrees):
        if l != lb or d <= threshold:
            continue
        map_nodes[n.item()] = l.item()
    print(map_nodes.keys())

def confirm_label(edges):
    s1 = (edges.src['label'] != 0) & (edges.src['label'] != 4) & (edges.dst['label'] != 0) & (edges.dst['label'] != 4)
    return {'d' : s1.to(torch.int32)}

def insert_edge_label(g):
    with g.local_scope():
        g.apply_edges(confirm_label)
        return g.edata['d']

# def recall_score(true_graph, pred_graph):
#     true_edges = set(true_graph.edata["_ID"].numpy().tolist())
#     pred_edges = pred_graph.edata["_ID"]
#     total = 0
#     for e in pred_edges:
#         if e.item() not in true_edges:
#             continue
#         total += 1
#     return total / true_graph.num_edges()

def node_importance(g, preferences=None, dumping=0.85, num_iter=10):
#     print(preferences)
    transition_matrix = g.edata['weight']
    if len(transition_matrix.shape) > 1:
        transition_matrix = transition_matrix.squeeze()
    num_nodes = g.num_nodes()
    with g.local_scope():
        gt = dgl.reverse(g, False, True)
        personalized_pagerank(gt, num_iter, dumping, preferences, True, transition_matrix)
        sampling_probs = gt.ndata['pv'].detach()
    return sampling_probs

def edge_importance(g, node_probs):
    with g.local_scope():
        g.ndata['pv'] = node_probs
        g.apply_edges(dgl.function.e_mul_v('weight', 'pv', 'epv'))
        return g.edata['epv'].detach()
    
def compute_importance(g, preferences=None, scale=False, dumping=0.85, num_iter=10):
    nprobs = node_importance(g, preferences, dumping=dumping, num_iter=num_iter)
    eprobs = edge_importance(g, nprobs)
    if scale:
        nprobs = nprobs * g.num_nodes()
        eprobs = eprobs * g.num_edges()
    return nprobs, eprobs


def khop_neighbors(dg, nid, k=3, ignore_self_loop=True):
    frontier = torch.LongTensor([nid])
    all_neighbors = [frontier]
    i = k
    while i > 0:
        src, _ = dg.in_edges(frontier)
        all_neighbors.append(src)
        frontier = src
        i -= 1
    sg = dgl.node_subgraph(dg, torch.hstack(all_neighbors).unique())
    if ignore_self_loop:
        sg = sg.remove_self_loop()
    return sg

def compute_edge_weights(model, g, model_name='gcn_mlp', sym=False): # use for gcn_mlp & gat
    model.eval()
    with torch.no_grad():
        h = g.ndata['feat']
        if model_name == "gcn_mlp":
            emb = model.node_mlp(h)
            weights = model.edge_mask(g, emb)
        else:
            atts = []
            for l in range(model.num_layers):
                h, att1 = model.gat_layers[l](g, h, True)
                h = h.flatten(1)
                att1 = att1.sum(1).squeeze()
                emb = h
                atts.append(att1)
            _, att1 = model.gat_layers[-1](g, h, True)
            atts.append(att1.squeeze())
            atts = torch.vstack(atts).sum(0).squeeze()
            # weights = atts
            weights = fn.edge_softmax(g, atts)
        if sym:
            adj = g.adj(scipy_fmt='coo')
            adj.data = weights.numpy()
            adj_t = adj.transpose()
            adj = adj + adj_t
            weights = torch.tensor(adj.data)
        g.ndata['emb'] = emb
        g.edata['weight'] = weights
        
def khop_batch(model, g, nids, k=3, ignore_self_loop=True, model_name="gcn_mlp", sym=False):
    dg = deepcopy(g)
    compute_edge_weights(model, dg, model_name, sym=sym)
    graphs = [khop_neighbors(dg, n, k, ignore_self_loop) for n in nids]
    return graphs

def explain_test(g, node_id, top=10, ax=None, undir=True, scale=15, plot=True,
                 draw_zero_edge=False, dumping_factor=0.85, num_iter=10,
                 aggressive=True, **karg): 
    """
    has_self_edge: use when graph is directed and include self-edge
    """
    node_probs, edge_probs = compute_importance(g, torch.LongTensor([node_id]),
                                                scale=True, dumping=dumping_factor, num_iter=num_iter)
    top_nodes = torch.topk(node_probs, top)
    g.edata['prob'] = edge_probs
    ng = dgl.node_subgraph(g, top_nodes.indices)
    # filter out disconnected graph
    z_out_degrees = (ng.out_degrees() == 0).nonzero().flatten()
    z_in_degrees = (ng.in_degrees() == 0).nonzero().flatten()
    f2 = np.intersect1d(z_out_degrees, z_in_degrees)
    if aggressive:
        ng_out_degrees = (ng.out_degrees() == 1).nonzero().flatten()
        ng_in_degrees = (ng.in_degrees() == 1).nonzero().flatten()
        f1 = np.intersect1d(ng_out_degrees, ng_in_degrees)
        filtered_nodes = np.concatenate([f1, f2])
    else:
        filtered_nodes = f2
    ng = dgl.remove_nodes(ng, filtered_nodes)
    nodes = ng.ndata['_ID']
    labels = {k.item(): v.item() for k, v in zip(ng.nodes(), nodes)}
    edge_probs = ng.edata['prob']
    if plot:
        if len(edge_probs):
            edge_probs = scaler(edge_probs)
            edge_probs = edge_probs*scale
            edge_probs = edge_probs.numpy()
        else:
            edge_probs = [scale] * ng.num_edges()
        if draw_zero_edge:
            edge_probs += 1
        draw_simple_graph(ng, edge_probs, undir, labels=labels, node_id=node_id, ax=ax, **karg)
    return ng

def extract_true_motif(g, node_idx, base_idx, size, ignore_self_loop=False):
    pos = (node_idx - base_idx - 1) % size
    start_idx = node_idx - pos 
    end_idx = start_idx + size
    ng = dgl.node_subgraph(g, torch.arange(start_idx, end_idx))
    if ignore_self_loop:
        ng = ng.remove_self_loop()
    return ng

def get_preds(true_graph, pred_graph): # use to get predictions of transition matrix & pagerank
    true_edges = true_graph.edata["_ID"].numpy()
    pred_edges = set(pred_graph.edata["_ID"].numpy().tolist())
    total = 0
    real, pred = [], []
    
    for e in true_edges:
        real.append(1)
        if e in pred_edges:
            pred.append(1)
            total += 1
            pred_edges.remove(e)
        else:
            pred.append(0)
    for e in pred_edges: # edges not in true_edges
        real.append(0)
        pred.append(1)
    return real, pred, total / true_graph.num_edges()

def evaluate_dataset(g, selected_node, motifs, s=6, e=14, dumping_factor=0.85, ignore_self_loop=False, **karg):
    # edge_labels = insert_edge_label(g)
    # g.edata['label'] = edge_labels
    best_options = {}
    reals, preds = [], []
    for i, n in tqdm(enumerate(selected_node)):
        best_idx = -1
        best_rec = 0.
        best_real, best_pred = [], []
        for j in range(s,e):
            ng = explain_test(g, n, j, plot=False, dumping_factor=dumping_factor, **karg)
            if ignore_self_loop:
                ng = ng.remove_self_loop()
#             temp_one = ng.edata['label'].to(torch.float32).mean()
            real, pred, tmp_rec = get_preds(motifs[i], ng)
            if best_rec < tmp_rec:
                best_rec = tmp_rec
                best_idx = j
                best_real, best_pred = real, pred
            if best_rec > 0.95:
                break
        best_options[n] = best_idx
        reals.extend(best_real)
        preds.extend(best_pred)
    all_pre = precision_score(reals, preds)
    all_rec = recall_score(reals, preds)
    return all_pre, all_rec, best_options

def obtain_predictions(tpr, fpr, threshold, preds):
    optimal_proba_cutoff = sorted(zip(tpr - fpr, threshold), key=lambda i: i[0], reverse=True)[0][1]
    roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in preds]
    return roc_predictions

def compare_real_pred(real, pred, nid=None): # use to compare two graph with probability (gcn_mlp & gat cases)
    if nid:
        feat = pred.ndata['emb']
        target_mask = (pred.ndata['_ID'] == nid)
        new_nid = target_mask.nonzero()[0][0].item()
        target_feat = feat[new_nid,:]
        nfeat = feat @ target_feat
        nfeat = nfeat * (~target_mask).to(torch.float32)
        pred.ndata['node_weight'] = nfeat
        pred.apply_edges(dgl.function.u_mul_e('node_weight', 'weight', 'weight'))
    true_edges = set(real.edata['_ID'].tolist())
    labels = []
    for e in pred.edata['_ID']:
        if e.item() in true_edges:
            labels.append(1)
            true_edges.remove(e.item())
        else:
            labels.append(0)
    pred_probs = pred.edata['weight'].tolist()
    remain = len(true_edges)
    if remain:
        labels.extend([1] * remain)
        pred_probs.extend([0] * remain)
    if len(np.unique(labels)) == 1:
        if labels[0] == 1:
            labels.append(0)
            pred_probs.append(0)
        else:
            raise ValueError("All labels are 0")
#     print({v:k for k, v in zip(labels, pred_probs)})
    fpr, tpr, thres = roc_curve(labels, pred_probs)
    pred_labels = obtain_predictions(tpr, fpr, thres, pred_probs)
    return labels, pred_labels

# eval using weight only; gcn_mlp & gat cases
def eval_dataset2(base_graphs, pred_graphs, nids=None):
    reals, preds = [], []
    if nids:
        for r, p, n in zip(base_graphs, pred_graphs, nids):
            real, pred = compare_real_pred(r, p, n)
            reals.extend(real)
            preds.extend(pred)
    else:
        for r, p in zip(base_graphs, pred_graphs):
            real, pred = compare_real_pred(r, p)
            reals.extend(real)
            preds.extend(pred)
    pre = precision_score(reals, preds)
    rec = recall_score(reals, preds)
    f1 = 2*pre*rec / (pre + rec)
    print("F1 Score: %4f"% f1)
    print("P Score: %4f"% pre)
    print("R score: %4f"% rec)

def kl(p1, p2):
    # DL(p1|p2) = p1 * log(p1/p2) = p1*(log(p1) - log(p2))
    p1 = F.softmax(p1, 1)
    p2 = F.softmax(p2, 1)
    dv = torch.log(p1) - torch.log(p2)
    return (p1 * dv).sum(1)

def predict(base, explainer, g, is_all=False):
    base.eval()
    explainer.eval()
    with torch.no_grad():
        if is_all:
            test_labels = g.ndata['label']
        else:
            test_mask = g.ndata['test_mask']
            test_labels = g.ndata['label'][test_mask]
        pred_logits = base(g, g.ndata['feat'])
        emb = g.ndata['emb'].detach().clone()
        weights = explainer.edge_mask(g, emb)
        e_logits = explainer(g, g.ndata['feat'], emb)
        g.edata['weight'] = weights
        if is_all:
            b_preds, e_preds = torch.argmax(pred_logits, 1), torch.argmax(e_logits, 1)
        else:
            b_preds, e_preds = torch.argmax(pred_logits, 1)[test_mask], torch.argmax(e_logits, 1)[test_mask]
        b_acc = (b_preds == test_labels).to(torch.float32).mean()
        e_acc = (e_preds == test_labels).to(torch.float32).mean()
        kl_score = kl(F.softmax(pred_logits, 1), F.softmax(e_logits, 1))
        agr = (b_preds == e_preds).to(torch.float32).mean()
        print("Base accuracy: %.4f, Explainer accuracy: %.4f" % (b_acc, e_acc))
        print("Agreement score: %.4f, KL Score: %.4f" % (agr, kl_score.mean()))
        return b_preds, e_preds