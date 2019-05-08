# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:49:05 2019

@author: Aditya 
"""

import torch
import torch.optim as optim

from gcn import * 
from graphData import *
from coverageProbability import *

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
    
def learnPathProbs_simple(train_data, test_data, lr=0.1):
    
    net2= GCNPredictionNet(feature_size)
    net2.train()
    optimizer=optim.SGD(net2.parameters(), lr=lr)
    
    n_epochs=150
    n_iterations=n_epochs*len(train_data)
    
    #print ("N_training graphs/ N_samples: ",len(training_graphs), len(train_data))
    #print ("N_testing graphs/N_samples: ",len(testing_graphs), len(test_data))

    # TESTING LOOP before training
    batch_loss=0.0    
    for iter_n in range(len(test_data)):
        G,Fv, coverage_prob, phi, path_probs=test_data[iter_n]
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
        
        G, Fv, coverage_prob, phi, path_probs=train_data[iter_n%len(train_data)]
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
        G,Fv, coverage_prob, phi, path_probs=test_data[iter_n]
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


def learnEdgeProbs_simple(train_data, test_data, lr=0.1, path_model='random_walk'):
    
    net2= GCNPredictionNet(feature_size)
    net2.train()
    optimizer=optim.SGD(net2.parameters(), lr=lr)
    
    n_epochs=150
    n_iterations=n_epochs*len(train_data)
    
    #print ("N_training graphs/ N_samples: ",len(training_graphs), len(train_data))
    #print ("N_testing graphs/N_samples: ",len(testing_graphs), len(test_data))

    # TESTING LOOP before training
    batch_loss=0.0    
    for iter_n in range(len(test_data)):
        if path_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=test_data[iter_n]
        else:
            G,Fv, coverage_prob, phi, path=test_data[iter_n]
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
        
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        edge_probs_pred=generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred)
        
        loss=torch.zeros(1)
        for e in path: 
            loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        
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
        
        if path_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=train_data[iter_n%len(train_data)]
        else:
            G,Fv, coverage_prob, phi, path=train_data[iter_n%len(train_data)]       
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
    
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        edge_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred)
        
        '''
        loss=torch.zeros(1)
        for e in path: 
            loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        '''
        if path_model=='random_walk_distribution':
            log_prob_pred=torch.zeros(1)
            for e in path: 
                log_prob_pred-=torch.log(edge_probs_pred[e[0]][e[1]]) 
            
            
            loss_function=nn.MSELoss()
            loss=loss_function(log_prob_pred,log_prob)
        else:
            loss=torch.zeros(1)
            for e in path: 
                loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        
        batch_loss+=loss
        #print ("Loss: ", loss)
        loss.backward(retain_graph=True)
        optimizer.step()    
        #defender_utility=calculateDefenderUtility(net2, test_data)



    # TESTING LOOP    
    batch_loss=0.0
    for iter_n in range(len(test_data)):
        if path_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=test_data[iter_n]
        else:
            G,Fv, coverage_prob, phi, path=test_data[iter_n]        
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        edge_probs_pred=generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred)
        
        loss=torch.zeros(1)
        for e in path: 
            loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        
    
        batch_loss+=loss
        #print ("Loss: ", loss)
    print("Testing batch loss per sample:", batch_loss/len(test_data))
    
    return net2

def calculateDefenderUtility(net2, test_data):
    
    total_ideal_defender_utility=0.0
    total_pred_defender_utility=0.0
    
    for indx in range(len(test_data)):
        G,Fv, coverage_prob, phi, path=test_data[indx]
        # TODO: add these two in createGraph() method while creating random graphs
        #budget= 
        #U=
        #Fv_torch=
        #A_torch=
        
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        pred_optimal_coverage=get_optimal_coverage_prob(G, phi_pred, U, initial_distribution, budget, omega=4)
        ideal_optimal_coverage=get_optimal_coverage_prob(G, phi, U, initial_distribution, budget, omega=4)
        
        total_pred_defender_utility+=   -(objective_function(pred_optimal_coverage,G, phi, U, omega=4))
        total_ideal_defender_utility+=  -(objective_function(ideal_optimal_coverage,G, phi, U, omega=4))
        
    return total_ideal_defender_utility, total_pred_defender_utility
    
    


if __name__=='__main__':
    
    feature_size=25
    #path_model_type='random_walk_distribution'      
    path_model_type='random_walk'      
         
    train_data, test_data=generateSyntheticData(feature_size, 
                                             path_type=path_model_type, 
                                                n_data_samples=1000, fixed_graph=False)
    
    
    # Learn the neural networks:
    if path_model_type=='simple':
        net2=learnPathProbs_simple(train_data,test_data)
    elif path_model_type=='random_walk':
        net2=learnEdgeProbs_simple(train_data,test_data, lr=0.05)
    elif path_model_type=='random_walk_distribution':
        net2=learnEdgeProbs_simple(train_data,test_data, lr=0.05, path_model=path_model_type)
    
    # Determine the optimal coverage: 
        
    
