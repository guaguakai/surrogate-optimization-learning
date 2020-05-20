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

cmaps['Sequential'] = [cm.Reds, cm.Blues]
thresholds = [0.2, 0.1, 0.2, 0.2, 0.2]
alphas = [0.9, 0.9, 0.9, 0.9, 0.9]
a_maxs = [0.2, 0.2]
a_mins = [0, 0.0]

def plot_graph(G, T, epoch):
    # colors is a #edges x 3 matrix, which represents the RGB color of the corresponding edge
    N = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    pos = nx.get_node_attributes(G, 'pos')

    print('sources', G.graph['sources'], 'targets', G.graph['targets'])
    nx.draw_networkx_nodes(G, pos, nodelist=list(set(G.nodes()) - set(G.graph['sources']) - set(G.graph['targets'])), node_color='grey', node_shape='o', node_size=10, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.graph['sources']), node_color='green', node_size=80, node_shape='^', alpha=1)
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.graph['targets']), node_color='green', node_size=120, node_shape='*', alpha=1)

    # edge_labels = {edge: ' '.join(['{:.3f}'.format(tt) for tt in t]) for edge, t in zip(G.edges(), T)}
    # for edge, label in zip(G.edges(), edge_labels):
    #     G[edge[0]][edge[1]]['state'] = label

    # edge_labels = nx.get_edge_attributes(G,'state')
    choices = np.argmax(T, axis=1)
    print('T shape:', T.shape)
    for t in range(T.shape[1]):
        threshold = thresholds[t]
        # alpha = alphas[t]
        a_min, a_max = a_mins[t], a_maxs[t]
        cmap = cmaps['Sequential'][t]
        indices = np.where(choices == t)[0]
        intensity = np.clip(T[choices==t,t], a_min=a_min, a_max=a_max)
        alpha = intensity / a_max * alphas[t] * 0.6 + 0.2
        edges = [list(G.edges())[index] for index in indices]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=intensity, width=3.0, edge_vmin=0, edge_vmax=threshold, alpha=alpha, edge_cmap=cmap)
    # plt.show()
    nx.draw_networkx_edges(G, pos, edgelist=list(set(G.edges())), width=0.5, alpha=1, edge_color='black', style='dashed')
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
