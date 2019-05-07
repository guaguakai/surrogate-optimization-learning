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
    
def learnPathProbs(G, data, coverage_probs, Fv, all_paths):
    
    A=nx.to_numpy_matrix(G)
    A_torch = torch.as_tensor(A, dtype=torch.float) 
    Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
    feature_size=Fv_torch.size()[1]
    
    net2= GCNPredictionNet(A_torch, feature_size)
    net2.train()
    optimizer=optim.SGD(net2.parameters(), lr=0.3)
    n_iterations=400
    #out=net2(x).view(1,-1)
    #print("out:", out)
    #print(out.size())
    #print (len(list(net2.parameters())))
    #print (list(net2.parameters())[5].size())
    #loss=nn.MSELoss()
    #print (loss(out, y))
    
    for _ in range(n_iterations):
        optimizer.zero_grad()
        loss_function=nn.CrossEntropyLoss()
        #loss_function=nn.MSELoss()
        
        phi_pred=net2(Fv_torch).view(-1)
        print("Flag:",phi_pred.requires_grad)
        #path_probs_pred=generate_PathProbs_from_Attractiveness(G, coverage_probs, phi_pred, all_paths, n_paths=len(all_paths))
        #data_sample=data[n_iterations%len(data)]
        N=nx.number_of_nodes(G) 

        # GENERATE EDGE PROBABILITIES 
        omega=4
        edge_probs=torch.zeros((N,N))
        for i, node in enumerate(list(G.nodes())):
            neighbors=list(nx.all_neighbors(G,node))
            
            smuggler_probs=torch.zeros(len(neighbors))
            for j,neighbor in enumerate(neighbors):
                e=(node, neighbor)
                #pe= G.edge[node][neighbor]['coverage_prob']
                pe=coverage_probs[node][neighbor]
                smuggler_probs[j]=torch.exp(-omega*pe+phi_pred[neighbor])
            
            smuggler_probs=smuggler_probs/torch.sum(smuggler_probs)
            
            for j,neighbor in enumerate(neighbors):
                edge_probs[node,neighbor]=smuggler_probs[j]
                print(edge_probs[node, neighbor].requires_grad)        

                
        # GENERATE PATH PROBABILITIES
        n_paths=len(all_paths)
        path_probs=torch.zeros(n_paths)
        for path_number, path in enumerate(all_paths):
            path_prob=torch.ones(1)
            for i in range(len(path)-1):
                path_prob*=edge_probs[path[i], path[i+1]]
            path_probs[path_number]=path_prob
        path_probs=path_probs/torch.sum(path_probs)
        print(path_probs[0].requires_grad)
        #print ("SUM: ",torch.sum(path_probs))
        #path_probs=torch.from_numpy(path_probs)
        print ("Path probs:", path_probs, sum(path_probs))
        
        loss=torch.zeros(1)
        #print ("Sizes::", (path_probs.view(1,-1)).size(), data[0].view(1,-1) )
        for data_sample in data:
            loss+=loss_function(path_probs.view(1,-1),data_sample)
        #print("Loss before:",loss.grad_fn.next_functions[0][0].grad)
        #loss=([loss_function(path_probs.view(1,-1),data_sample.view(1)) for data_sample in data])
        print("Loss: ", loss)
        #net2.zero_grad()
        loss.backward()
        print("Loss after:",loss.is_leaf, loss.grad)
        
        optimizer.step()
    
def learnPathProbs_simple(train_data, test_data, training_graphs, testing_graphs):
    
    net2= GCNPredictionNet(feature_size)
    net2.train()
    optimizer=optim.SGD(net2.parameters(), lr=0.3)
    
    n_epochs=150
    n_iterations=n_epochs*len(train_data)
    
    print ("N_training graphs/ N_samples: ",len(training_graphs), len(train_data))
    print ("N_testing graphs/N_samples: ",len(testing_graphs), len(test_data))

    # TESTING LOOP before training    
    batch_loss=0.0
    for iter_n in range(len(test_data)):
        graph_index,Fv, coverage_prob, phi, path_probs=test_data[iter_n]
        G=testing_graphs[graph_index]
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
        
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        all_paths=list(nx.all_simple_paths(G, source, target))
        n_paths=len(all_paths)
        path_probs_pred=generate_PathProbs_from_Attractiveness(G, coverage_prob,  phi_pred, all_paths, n_paths)
        
        #loss_function=nn.CrossEntropyLoss()
        loss_function=nn.MSELoss()
        
        loss=loss_function(path_probs_pred,path_probs)
        batch_loss+=loss
        #print ("Loss: ", loss)
    print("Testing batch loss per sample before training:", batch_loss/len(test_data))
    
    # TRAINING LOOP
    batch_loss=0.0
    for iter_n in range(n_iterations):
        optimizer.zero_grad()
        if iter_n%len(train_data)==0:
            print("Epoch number/Batch loss/ Batch loss per sample: ", iter_n/len(train_data),batch_loss, batch_loss/len(train_data))
            batch_loss=0.0
        
        graph_index, Fv, coverage_prob, phi, path_probs=train_data[iter_n%len(train_data)]
        G=training_graphs[graph_index]
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
    
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        all_paths=list(nx.all_simple_paths(G, source, target))
        n_paths=len(all_paths)
        path_probs_pred=generate_PathProbs_from_Attractiveness(G, coverage_prob,  phi_pred, all_paths, n_paths)
        

        #loss_function=nn.CrossEntropyLoss()
        loss_function=nn.MSELoss()
        
        loss=loss_function(path_probs_pred,path_probs)
        batch_loss+=loss
        #print ("Loss: ", loss)
        loss.backward(retain_graph=True)
        optimizer.step()    



    # TESTING LOOP    
    batch_loss=0.0
    for iter_n in range(len(test_data)):
        graph_index,Fv, coverage_prob, phi, path_probs=test_data[iter_n]
        G=testing_graphs[graph_index]
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
        
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        all_paths=list(nx.all_simple_paths(G, source, target))
        n_paths=len(all_paths)
        path_probs_pred=generate_PathProbs_from_Attractiveness(G, coverage_prob,  phi_pred, all_paths, n_paths)
        
        #loss_function=nn.CrossEntropyLoss()
        loss_function=nn.MSELoss()
        
        loss=loss_function(path_probs_pred,path_probs)
        
        batch_loss+=loss
        #print ("Loss: ", loss)
    print("Testing batch loss per sample:", batch_loss/len(test_data))    
    
    print ("N_training graphs/ N_samples: ",len(training_graphs), len(train_data))
    print ("N_testing graphs/N_samples: ",len(testing_graphs), len(test_data))

if __name__=='__main__':
    
    #G= returnGraph(fixed_graph=True)
    feature_size=25
    #d=generateSyntheticData(G,feature_size)
    
    train_data, test_data, training_graphs, testing_graphs=generateSyntheticData(feature_size, n_training_graphs=800)
    '''
    data=d['data']
    all_paths=d['paths']
    coverage_probs=d['coverage_probs']
    Fv_torch=d['features']
    '''
    """
    data: numpy array of samples, where each sample is the path_number, picked according to path probs. 
    all_paths: all possible paths from source to target
    coverage_prob: randomly generated coverage prob
    Fv_torch: Feature tensor
    """
    
    
    learnPathProbs_simple(train_data,test_data, training_graphs, testing_graphs)
    
