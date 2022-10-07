def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from tqdm import tqdm
import shap
import matplotlib.pyplot as pl; 

def marginal_contribution(model: any, g: dgl.heterograph, target_node: int, measure_node: int,\
                        sampling_probs: np.array, sample_size: int, cls: int=-1, feat: str='feat'):
    
    nodes = g.nodes()
    keep_sampling = np.random.choice(nodes, sample_size, replace=False, p=sampling_probs)
    keep_sampling.sort()
    idx = np.searchsorted(keep_sampling, target_node)
    if keep_sampling[idx] != target_node:
        keep_sampling = np.insert(keep_sampling, idx, target_node)
    
    idx = np.searchsorted(keep_sampling, measure_node)
    if keep_sampling[idx] != measure_node:
        wo_sampling = keep_sampling.copy()
        keep_sampling = np.insert(keep_sampling, idx, measure_node)
    else:
        wo_sampling = np.delete(keep_sampling, idx)
    # with v
    g_v = dgl.node_subgraph(g, keep_sampling)
    old_idx = g_v.ndata['_ID']
    features = g_v.ndata[feat]
    logits = model(features, g=g_v)
    target_node_nidx = (old_idx == target_node).nonzero().item()
    logit_with_v = torch.softmax(logits[target_node_nidx, :], 0)
    if cls == -1:
        cls = torch.argmax(logit_with_v)
        print("Auto explain class: %i" % cls.item())
    logit_with_v = logit_with_v[cls]
    
    # without v
    g_wv = dgl.node_subgraph(g, wo_sampling)
    features = g_wv.ndata[feat]
    logits = model(features, g=g_wv)
    old_idx = g_wv.ndata['_ID']
    target_node_nidx = (old_idx == target_node).nonzero().item()
    logit_wo_v = torch.softmax(logits[target_node_nidx, :], 0)
    logit_wo_v = logit_wo_v[cls]
    return (logit_with_v - logit_wo_v).item()
        
def mc_sampling(model, g, target_node, measure_nodes, sampling_probs, sample_size, cls, sample_num=1000):
    revenues = {}
    for _ in tqdm(range(sample_num)):
        for i, m in enumerate(measure_nodes.indices):
            if i == 0:
                continue
            m_id = m.item()
            v = marginal_contribution(model, g, target_node, m_id, sampling_probs.numpy(), sample_size, cls)
            if m_id not in revenues:
                revenues[m_id] = v
            else:
                revenues[m_id] += v
    for k, v in revenues.items():
        revenues[k] = v / sample_num
    return revenues

def feature_importance(model, bg_data, test_data=None, explainer_type="sampling", model_type="classification"):
    """Given a feature-based model, output feature attributions
        model: pytorch model, has to take features as input and output logits
        bg_data: background data for shapley computation (mean), usually training data (should be numpy array or torch tensor)
        explainer_type: refer to SHAP kernel for deep nn models Kernel|Deep
        ===========
        return: shap_values of explainer
    """
    # f = lambda x : model(torch.FloatTensor(x)).detach().numpy()
    f = lambda x : F.softmax(model(torch.FloatTensor(x)), 1).detach().numpy()
    if type(bg_data) == torch.Tensor:
        bg_data = bg_data.detach().numpy()
    
    explainer = None
    if explainer_type == "kernel":
        explainer = shap.KernelExplainer(f, bg_data)
    elif explainer_type == "sampling":
        explainer = shap.SamplingExplainer(f, bg_data)
    elif explainer_type == "deep":
        explainer = shap.DeepExplainer(f, bg_data)
    if explainer is None:
        raise ValueError("Explainer must be kernel or deep")
    
    if type(test_data) == torch.Tensor:
        test_data = test_data.detach().numpy()
    if not test_data is None:
        shap_values = explainer.shap_values(test_data)
    else:
        shap_values = None
    return explainer, shap_values
    
def summary_plot(shap_values, plot_type="bar", feature_names=None, filepath="", savefig=False):
    shap.summary_plot(shap_values, plot_type=plot_type, feature_names=feature_names, show=not savefig)
    if savefig:
        pl.savefig(filepath)

def bar_plot(shap_value, feature_names=None, filepath="", savefig=False):
    shap.bar_plot(shap_value, feature_names=feature_names, show=not savefig)
    if savefig:
        pl.savefig(filepath)

def force_plot(exp, shap_value, feature_names=None, filepath="", savefig=False):
    shap.force_plot(exp, shap_value, feature_names=feature_names)
    if savefig:
        pl.savefig(filepath)