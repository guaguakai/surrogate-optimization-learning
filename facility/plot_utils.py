import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from sklearn.decomposition import PCA
from collections import OrderedDict

cmaps = OrderedDict()

cmaps['Sequential'] = [
        cm.Purples, cm.Blues, cm.Greens, cm.Oranges, cm.Reds,
        cm.YlOrBr, cm.YlOrRd, cm.OrRd, cm.PuRd, cm.RdPu, cm.BuPu,
        cm.GnBu, cm.PuBu, cm.YlGnBu, cm.PuBuGn, cm.BuGn, cm.YlGn]

cmaps['Sequential'] = [cm.Blues, cm.Reds, cm.Greens, cm.Oranges, cm.Purples]
alphas = [0.5, 0.5, 0.5, 0.5, 0.5]
a_maxs = [0.2, 0.2, 0.2, 0.2, 0.2]
a_mins = [0.01, 0.01, 0.01, 0.01, 0.01]

def reduce_dimension(T, output_dim=3):
    T = (T - np.min(T, axis=0, keepdims=True)) / np.ptp(T, axis=0, keepdims=True)

    # PCA
    pca = PCA(n_components=output_dim)
    newT = pca.fit_transform(T)

    # normalization
    newT = (newT - np.min(newT, axis=0, keepdims=True)) / np.ptp(newT, axis=0, keepdims=True)
    return newT

def manually_reduce_dimension(T, output_dim=2):
    newT = np.zeros((len(T), 2))
    newT[:,0] = np.array([np.mean(T_row) for T_row in T])
    newT[:,1] = np.array([np.std(T_row[T_row > 0]) for T_row in T])
    return newT

def plot_graph(labels, T, epoch):
    # c is the ground truth rating of all the users x all the movies
    # T is the meta-variable that governs the decision variables (selected movies)

    # locations = reduce_dimension(labels[0], 2)
    locations = manually_reduce_dimension(labels[0], 2)

    print(labels.shape, T.shape, locations.shape)

    sizes = np.mean(T, axis=1) * 30000 + 10
    print(sizes)

    T_prob = (T + 1e-3) / np.sum(T + 1e-3, axis=1, keepdims=True)
    choices = np.array([np.random.choice(len(T_row), p=T_row) for T_row in T_prob])

    print('T shape:', T.shape)

    plt.figure(figsize=(10,10))
    for t in range(T.shape[1]):
        vmin, vmax = a_mins[t], a_maxs[t]
        cmap = cmaps['Sequential'][t]
        alpha = alphas[t]
        indices = np.where(choices == t)[0]
        colors = np.clip(T[choices==t,t], a_min=vmin, a_max=vmax)

        plt.scatter(x=locations[indices,0], y=locations[indices,1], s=sizes[indices], c=colors, cmap=cmap, vmin=vmin, vmax=vmax,  alpha=alpha, edgecolors='b')

    plt.xlabel("average rating", fontsize=28, fontweight='bold')
    plt.ylabel("rating std", fontsize=28, fontweight='bold')
    plt.xlim(-0.04, 0.5)
    plt.ylim(-0.02, 0.25)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig('movie_results/visualization/epoch{}.png'.format(epoch), bbox_inches='tight')
    plt.clf()
