# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:52:08 2019

@author: Aditya
"""

import networkx as nx 
import matplotlib as plt
import numpy as np

G= nx.navigable_small_world_graph(5,p=1, q=5, r=4)
ratios=[]
N=1
p=0.3
for _ in range(N):
    connected=False
    while not connected:
        G=nx.generators.random_geometric_graph(15,p)
        connected=nx.is_connected(G)
    for edge in G.edges():
            G[edge[0]][edge[1]]['capacity'] = 1
            G[edge[1]][edge[0]]['capacity'] = 1
    s,t=list(G.nodes())[0],list(G.nodes())[-1]
    value, partition = nx.minimum_cut(G,s,t)
    print("VALUE",value)
    partition0, partition1 = set(partition[0]), set(partition[1])
    cut = []
    for idx, edge in enumerate(G.edges()):
        if edge[0] in partition0 and edge[1] in partition1:
            cut.append(idx)
        elif edge[0] in partition1 and edge[1] in partition0:
            cut.append(idx)
    #print (G.nodes())
        
    ratio=(1.0*len(cut))/nx.number_of_edges(G)
    ratios.append(ratio)
    print(len(cut), nx.number_of_edges(G))
    print ("DIAM",nx.diameter(G))
    print (s,t)
    #nx.draw(G)


if True:
        layers = [8,4,2,2]
        nodes = list(range(sum(layers)))
        sources, layer1, layer2, targets = nodes[:layers[0]], nodes[layers[0]:layers[0]+layers[1]], nodes[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]], nodes[-layers[3]:]
        transients = nodes[:-layers[3]]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(s, m1) for s in sources for m1 in layer1] + [(m1, m2) for m1 in layer1 for m2 in layer2] + [(m2, t) for m2 in layer2 for t in targets]) # undirected graph
        G.graph['sources']=sources
        G.graph['targets']=targets

        G.graph['U']=np.concatenate([np.random.rand(layers[-1]) * 10, np.array([0])])
        #G.graph['budget']=budget
        
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution
    
print(sum(ratios)/N)
nx.draw(G)