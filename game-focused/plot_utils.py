import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import cm
from collections import OrderedDict
from celluloid import Camera

cmaps = OrderedDict()

cmaps['Sequential'] = [
        cm.Purples, cm.Blues, cm.Greens, cm.Oranges, cm.Reds,
        cm.YlOrBr, cm.YlOrRd, cm.OrRd, cm.PuRd, cm.RdPu, cm.BuPu,
        cm.GnBu, cm.PuBu, cm.YlGnBu, cm.PuBuGn, cm.BuGn, cm.YlGn]

cmaps['Sequential'] = [cm.Blues, cm.Reds, cm.Greens]
alpha_default = [0.9, 0.9, 0.9, 0.9, 0.9]
a_maxs = [0.5] * 3 #, 0.2, 0.2]
a_mins = [0.15] * 3 # , 0.0, 0]
thresholds = [a_max + 0.1 for a_max in a_maxs]

def plot_graph(G, T, epoch):
    # colors is a #edges x 3 matrix, which represents the RGB color of the corresponding edge
    N = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(10,8))
    print('sources', G.graph['sources'], 'targets', G.graph['targets'])

    # edge_labels = {edge: ' '.join(['{:.3f}'.format(tt) for tt in t]) for edge, t in zip(G.edges(), T)}
    # for edge, label in zip(G.edges(), edge_labels):
    #     G[edge[0]][edge[1]]['state'] = label

    # edge_labels = nx.get_edge_attributes(G,'state')
    print(T)
    T_prob = (T + 1e-3) / np.sum(T + 1e-3, axis=1, keepdims=True)
    choices = np.array([np.random.choice(len(T_row), p=T_row) for T_row in T_prob])
    # choices = np.argmax(T, axis=1)
    print('T shape:', T.shape)

    for t in range(T.shape[1]):
        threshold = thresholds[t]
        # alpha = alphas[t]
        a_min, a_max = a_mins[t], a_maxs[t]
        cmap = cmaps['Sequential'][t]
        indices = np.where(choices == t)[0]
        bound = 0.05
        intensities = np.clip(np.mean(T[choices==t], axis=1) / bound * (a_max - a_min) + a_min, a_min=a_min, a_max=a_max)
        alphas      = np.clip(np.mean(T[choices==t], axis=1) / bound * (0.75 - 0.5) + 0.5, a_min=0.5, a_max=0.75)
        widths      = np.clip(np.mean(T[choices==t], axis=1) / bound * (12 - 3) + 3, a_min=3, a_max=10)
        edges = [list(G.edges())[index] for index in indices]
        for edge, intensity, width, alpha in zip(edges, intensities, widths, alphas):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=[intensity], width=width, edge_vmin=a_min, edge_vmax=threshold, alpha=alpha, edge_cmap=cmap)
    # plt.show()
    nx.draw_networkx_edges(G, pos, edgelist=list(set(G.edges())), width=1, alpha=1, edge_color='black', style='dashed')

    nx.draw_networkx_nodes(G, pos, nodelist=list(set(G.nodes()) - set(G.graph['sources']) - set(G.graph['targets'])), node_color='grey', node_shape='o', node_size=10, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.graph['sources']), node_color='violet', node_size=700, node_shape='^', alpha=1)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.graph['targets']), node_color='orange', node_size=1000, node_shape='*', alpha=1)
    plt.axis('off')
    plt.savefig('results/visualization/epoch{}.png'.format(epoch), bbox_inches='tight')
    plt.clf()


def vectors2colors(T):
    T_reduces = reduce_dimension(T)
    np.argmax(T)
    pass


def reduce_dimension(T, output_dim=3):
    pca = PCA(n_components=output_dim)
    newT = pca.fit_transform(T)
    newT = 1 - (newT - np.min(newT)) / np.ptp(newT)
    return newT
