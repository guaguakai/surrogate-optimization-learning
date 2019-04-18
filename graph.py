# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:33:57 2019

@author: Aditya
"""
import networkx as nx 
import numpy as np
import torch

from gcn import * 

"""
Create the graph here. Define source and target. compute the adjacency matrix here. Compute all possible paths. 
Randomly also generate features (for now) for every node in the graph.  
Next, handle the data generation also here. So as part of this: 
    make a gcn object with suitable layers (more layers) for generation. 
    Pass the features through the gcn and obtain edge probabilities---> path probs for all paths
    Generate a dataset (train+test) by randomly sampling from the above prob distribution (can use one-hot vectors to denote paths)
    Split the data set into training and testing

"""

def generateSyntheticData(node_feature_size):
    
    # define an arbitrary graph with a source and target node
    source=0
    target=6
    G= nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)])
    N=nx.number_of_nodes(G) 
    #nx.draw(G)
    
    #  Define node features for each of the n nodes
    for node in G.nodes():
        node_features=np.random.randn(node_feature_size)
        # TODO: Use a better feature computation for a given node
        G.node[node]['node_features']=node_features
        
    # COMPUTE ADJACENCY MATRIX
    A=nx.to_numpy_matrix(G)
    #print("A:",A)
    
    # COMPUTE ALL POSSIBLE PATHS
    all_paths=list(nx.all_simple_paths(G, source, target))
    #print ("Paths:",all_paths)
    
    # GENERATE SYNTHETIC DATA:
    # Generate features
    Fv=np.zeros((N,node_feature_size))
    for node in G.nodes():
        Fv[node]=G.node[node]['node_features']
    print ("Feat matrix:",N, node_feature_size,Fv.shape)    
    
    # Generate attractiveness values for nodes
    A_torch, Fv_torch=torch.as_tensor(A, dtype=torch.float),torch.as_tensor(Fv, dtype=torch.float) 
    net1= GCNDataGenerationNet(A_torch, node_feature_size)        
    y=net1.forward(Fv_torch).view(1,-1)
    phi=y.data.numpy()
    '''
    phi is the attractiveness function, phi(v,f) for each of the N nodes, v
    '''
    
    #
    
    





if __name__=="__main__":
    generateSyntheticData(25)
