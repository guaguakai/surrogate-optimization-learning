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
N=20
p=0.5
for _ in range(N):
    connected=False
    while not connected:
        G=nx.generators.random_geometric_graph(20,p)
        connected=nx.is_connected(G)
    for edge in G.edges():
            G[edge[0]][edge[1]]['capacity'] = 1
            G[edge[1]][edge[0]]['capacity'] = 1
    s,t=list(G.nodes())[0],list(G.nodes())[-1]
    value, partition = nx.minimum_cut(G,s,t)
    #print(value)
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
    
print(sum(ratios)/N)
nx.draw(G)