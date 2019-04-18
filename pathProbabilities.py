# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:49:05 2019

@author: Aditya
"""

import torch
import torch.optim as optim

from gcn import * 
from graphData import *


def get_one_hot(value, n_values):
    
    t=torch.zeros(n_values)
    
def returnGraph():
    
    # define an arbitrary graph with a source and target node
    source=0
    target=6
    G= nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)], source=0, target=6)
    return G

def learnPathProbs(G, data, coverage_probs, Fv_torch, all_paths):
    
    A=nx.to_numpy_matrix(G)
    A_torch = torch.as_tensor(A, dtype=torch.float) 
    
    feature_size=Fv_torch.size()[1]
    
    net2= GCNPredictionNet(A_torch, feature_size)
    optimizer=optim.SGD(net2.parameters(), lr=0.01)
    n_iterations=400
    #out=net2(x).view(1,-1)
    #print("out:", out)
    #print(out.size())
    #print (len(list(net2.parameters())))
    #print (list(net2.parameters())[5].size())
    #loss=nn.MSELoss()
    #print (loss(out, y))
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss_function=nn.CrossEntropyLoss()
        #loss_function=nn.MSELoss()
        
        phi_pred=net2(Fv_torch).view(-1)
        path_probs_pred=generate_PathProbs_from_Attractiveness(G, coverage_probs, phi_pred, all_paths, n_paths=len(all_paths))
        #data_sample=data[n_iterations%len(data)]
        
        loss=torch.zeros(1)
        print ("Sizes::", (path_probs_pred.view(1,-1)).size(), data[0].view(1,-1) )
        loss=sum([loss_function(path_probs_pred.view(1,-1),data_sample.view(1)) for data_sample in data])
        print("Loss: ", loss)
        #net2.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    
    

if __name__=='__main__':
    
    G= returnGraph()
    feature_size=25
    d=generateSyntheticData(G,feature_size)
    
    data=d['data']
    all_paths=d['paths']
    coverage_probs=d['coverage_probs']
    Fv_torch=d['features'] 
    """
    data: numpy array of samples, where each sample is the path_number, picked according to path probs. 
    all_paths: all possible paths from source to target
    coverage_prob: randomly generated coverage prob
    Fv_torch: Feature tensor
    """
    
    
    learnPathProbs(G, data, coverage_probs, Fv_torch, all_paths)
    
    
    
    
    
    """
    A=torch.rand(10,10)
    x=torch.rand(10,25)
    
    
    net1= GCNDataGenerationNet(A, 25)
    y=net1.forward(x).view(1,-1)
    #print (y.size())
    #print("Y:", y)
    """