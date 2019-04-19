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

def generate_PathProbs_from_Attractiveness(G, coverage_prob,  phi, all_paths, n_paths,omega=4):
    
    #coverage_prob=torch.from_numpy(coverage_prob)
    #all_paths=torch.from_numpy(all_paths)
    #n_paths=torch.from_numpy(n_paths)

    N=nx.number_of_nodes(G) 

    # GENERATE EDGE PROBABILITIES 
    edge_probs=torch.zeros((N,N))
    for i, node in enumerate(list(G.nodes())):
        neighbors=list(nx.all_neighbors(G,node))
        
        smuggler_probs=torch.zeros(len(neighbors))
        for j,neighbor in enumerate(neighbors):
            e=(node, neighbor)
            #pe= G.edge[node][neighbor]['coverage_prob']
            pe=coverage_prob[node][neighbor]
            smuggler_probs[j]=torch.exp(-omega*pe+phi[neighbor])
        
        smuggler_probs=smuggler_probs*1.0/torch.sum(smuggler_probs)
        
        for j,neighbor in enumerate(neighbors):
            edge_probs[node,neighbor]=smuggler_probs[j]
          
            
    
    # GENERATE PATH PROBABILITIES
    path_probs=torch.zeros(n_paths)
    for path_number, path in enumerate(all_paths):
        path_prob=torch.ones(1)
        for i in range(len(path)-1):
            path_prob=path_prob*(edge_probs[path[i], path[i+1]])
        path_probs[path_number]=path_prob
    path_probs=path_probs/torch.sum(path_probs)
    path_probs[-1]=torch.max(torch.zeros(1), path_probs[-1]-(sum(path_probs)-1.000))
    #print ("SUM1: ",sum(path_probs))
    #path_probs=torch.from_numpy(path_probs)
    #print ("Path probs:", path_probs, sum(path_probs))
    
    return path_probs

def generateSyntheticData(G,node_feature_size, omega=4, n_data_samples=1000, testing_data_fraction=0.2):
    
    data=[]
    # COMPUTE ADJACENCY MATRIX
    A=nx.to_numpy_matrix(G)
    A_torch=torch.as_tensor(A, dtype=torch.float) 
    #print("A:",A)
        
    net1= GCNDataGenerationNet(A_torch, node_feature_size)        

    for sample_number in range(n_data_samples):
        N=nx.number_of_nodes(G) 
        #nx.draw(G)
        
        #  Define node features for each of the n nodes
        for node in list(G.nodes()):
            node_features=np.random.randn(node_feature_size)
            # TODO: Use a better feature computation for a given node
            G.node[node]['node_features']=node_features
            
        # Randomly assign coverage probability
        private_coverage_prob=np.random.rand(nx.number_of_edges(G))
        private_coverage_prob/=sum(private_coverage_prob)
        coverage_prob=torch.zeros(N,N)
        for i, e in enumerate(list(G.edges())):
            #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
            coverage_prob[e[0]][e[1]]=private_coverage_prob[i]
            coverage_prob[e[1]][e[0]]=private_coverage_prob[i]
          
            
        
        
        # COMPUTE ALL POSSIBLE PATHS
        source=G.graph['source']
        target=G.graph['target']
        all_paths=list(nx.all_simple_paths(G, source, target))
        n_paths=len(all_paths)
        
        
        # GENERATE SYNTHETIC DATA:
        # Generate features
        Fv=np.zeros((N,node_feature_size))
        for node in list(G.nodes()):
            Fv[node]=G.node[node]['node_features']
    
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        # Generate attractiveness values for nodes
        phi=net1.forward(Fv_torch).view(-1)
        #phi=y.data.numpy()
        '''
        phi is the attractiveness function, phi(v,f) for each of the N nodes, v
        '''
         
        path_probs=generate_PathProbs_from_Attractiveness(G,coverage_prob,phi, all_paths, n_paths)
        #print ("SUM2:", torch.sum(path_probs), path_probs)
        #data_point=np.random.choice(n_paths,size=1, p=path_probs)
        #data_point=(Fv, coverage_prob, path_probs)
        data_point=(Fv, coverage_prob, phi, path_probs)
        data.append(data_point)
        
    training_data=data[:int(n_data_samples*(1.0-testing_data_fraction))]
    testing_data=data[int(n_data_samples*(1.0-testing_data_fraction)):]
    
    return training_data, testing_data


if __name__=="__main__":
    
    # define an arbitrary graph with a source and target node
    source=0
    target=6
    G= nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)], source=0, target=6)
    generateSyntheticData(G,25)
    #pass
