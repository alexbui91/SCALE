import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.path import Path as MplPath  # To avoid collisions with pathlib.Path
import matplotlib.patches as patches

def to_undirected(gx, reduce="sum"):
    ngx = nx.Graph()
    ngx.add_edges_from(gx.edges(), w=0)
    for u, v, d in gx.edges(data=True):
        ngx[u][v]['w'] += d['w']
    return ngx

# Some useful functions
def normalize_vector(vector: np.array, normalize_to: float) -> np.array:    
    vector_norm = np.linalg.norm(vector)
    return vector * normalize_to / vector_norm

def orthogonal_vector(point: np.array, width: float, normalize_to: Optional[float] = None) -> np.array:
    EPSILON = 0.000001
    x = width
    y = -x * point[0] / (point[1] + EPSILON)

    ort_vector = np.array([x, y])

    if normalize_to is not None:
        ort_vector = normalize_vector(ort_vector, normalize_to)

    return ort_vector

def draw_self_loop(point: np.array, ax: Optional[plt.Axes] = None, padding: float = 1.5,
                   width: float = 0.3, plot_size: int = 10, linewidth = 0.2, color: str = "pink",
                   alpha: float = 0.5) -> plt.Axes:
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    
    point_with_padding = padding * point

    ort_vector = orthogonal_vector(point, width, normalize_to=width)

    first_anchor = ort_vector + point_with_padding
    second_anchor = -ort_vector + point_with_padding

    verts = [point, first_anchor, second_anchor, point]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    path = MplPath(verts, codes)

    patch = patches.FancyArrowPatch(
        path=path,
        facecolor='none',
        lw=linewidth,
        arrowstyle="<|-",
        color=color,
        alpha=alpha,
        mutation_scale=30  # arrowsize in draw_networkx_edges()
    )
    ax.add_patch(patch)
# arrowstyle=, style="dashed"
    return ax

def draw_simple_graph(g, weight=None, undir=True, node_size=1000, margin=0.05,
                      labels=None, node_id=-1, ax=None, self_loop=False, angle=90):
    gx = dgl.to_networkx(g)
    src, dst = g.edges()
    src, dst = src.tolist(), dst.tolist()
    for s, d, w in zip(src, dst, weight):
        gx[s][d][0]['w'] = w
    if undir:
        gx = to_undirected(gx)
    weight = [v for k, v in nx.get_edge_attributes(gx, 'w').items()]
    pos = nx.kamada_kawai_layout(gx)
    nodes = gx.nodes()
    if ax is None:
        ax = plt.subplot()
        ax.margins(margin)
    edge_colors = "green"
    if not weight:
        weight = 1
    if node_id < 0:
        node_colors = 'orange'
    else:
        node_ids = g.ndata["_ID"]
        node_colors = ["red" if node_ids[n] == node_id else "orange" for n in nodes]
    nx.draw_networkx_edges(gx, pos, width=weight,  edge_color=edge_colors, arrows=True,
                           arrowsize=20, alpha=.8, ax=ax)
    if self_loop:
        for node in gx.nodes:
            if (node, node) not in gx.edges:
                continue
            w = gx.get_edge_data(node, node)[0]['w']
            draw_self_loop(point=pos[node], ax=ax, color="green", alpha=0.8, linewidth=w)
    nx.draw_networkx_nodes(gx, pos, nodelist=nodes, node_color=node_colors, node_size=node_size, alpha=.9, ax=ax)
    if labels:
        selected_labels = {v:labels[v] for v in nodes}        
        nx.draw_networkx_labels(gx, pos, selected_labels, font_size=14, font_color="black", ax=ax)