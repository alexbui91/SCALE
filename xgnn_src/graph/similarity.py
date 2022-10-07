import sys, os
import faiss
import torch
import numpy as np
import pickle as pkl

def build_index(embeddings, num_centroids=100):
    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
    dim = embeddings[0].shape[0]
    quantizer = faiss.IndexFlatL2(dim)
    index_flat = faiss.IndexIVFFlat(quantizer, dim, num_centroids, faiss.METRIC_L2) # 100 centroids
    index_flat.train(embeddings)
    index_flat.add(embeddings)
    return index_flat

def search(query, background=None, index_flat=None, k=10):
    if index_flat is None:
        if type(background) == torch.tensor:
            background = background.numpy()
        elif type(background) == list:
            background = np.asarray(background, dtype=np.float32)
        index_flat = build_index(background)
    if type(query) != np.array:
        query = np.asarray(query, dtype=np.float32)
    _, I = index_flat.search(query, k)
    return I

def get_embeddings(model, dataset, idx):
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, v in enumerate(idx):
            g, l = dataset[v]
            preds, graph_emb = model(g, g.ndata['attr'])
            embeddings.append(graph_emb)
            labels.append(torch.argmax(preds, 1).item())
    return embeddings, labels