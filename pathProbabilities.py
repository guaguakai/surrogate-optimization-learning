# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:49:05 2019

@author: Aditya 
"""

import torch
import torch.optim as optim
import time 
from termcolor import cprint

from gcn import * 
from graphData import *
from coverageProbability import *

def get_one_hot(value, n_values):
    
    t=torch.zeros(n_values)
    
def learnPathProbs(G, data, coverage_probs, Fv, all_paths, omega=4):
    
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


def learnEdgeProbs_simple(train_data, test_data, lr=0.1, path_model='random_walk'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4):
    
    time1=time.time()
    net2= GCNPredictionNet(feature_size)
    net2.train()
    if optimizer=='adam':
        optimizer=optim.Adam(net2.parameters(), lr=lr)
    elif optimizer=='sgd':
        optimizer=optim.SGD(net2.parameters(), lr=lr)
    elif optimizer=='adamax':
        optimizer=optim.Adamax(net2.parameters(), lr=lr)
    
    #optimizer=optim.SGD(net2.parameters(), lr=lr)
 
    
    
    n_iterations=n_epochs*len(train_data)
    
    #print ("N_training graphs/ N_samples: ",len(training_graphs), len(train_data))
    #print ("N_testing graphs/N_samples: ",len(testing_graphs), len(test_data))
    print ("Testing performance BEFORE training:")
    testModel(test_data,net2,path_model, omega=omega)
    print ("Training...") 
    time2=time.time()
    if time_analysis:
        cprint (("TOTAL TESTING TIME: ", time2-time1),'red')
    # TRAINING LOOP
    training_loss=0.0
    batch_loss=torch.zeros(1)
    time3=time.time()
    for iter_n in range(n_iterations):

        optimizer.zero_grad()
        if iter_n%len(train_data)==0:
            np.random.shuffle(train_data)
            print("Epoch number/Training loss/ Training loss per sample: ", iter_n/len(train_data),training_loss, training_loss/len(train_data))
            time4=time.time()
            if time_analysis:
                cprint (("TRAINING TIME FOR THIS BATCH:", time4-time3),'red')
            time3=time4
            training_loss=0.0
            np.random.shuffle(train_data)
        if path_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=train_data[iter_n%len(train_data)]
        else:
            G,Fv, coverage_prob, phi, path=train_data[iter_n%len(train_data)]
            path=getSimplePath(G, path)
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
    
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        #print ("PHI PRED:", phi_pred)
        #print ("GCN1:", list(net2.parameters())[0].grad)
        #print ("PHI ACTUAL:", phi)
        
        edge_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
        
        
        if path_model=='random_walk_distribution':
            log_prob_pred=torch.zeros(1)
            for e in path: 
                log_prob_pred-=torch.log(edge_probs_pred[e[0]][e[1]]) 
            loss_function=nn.MSELoss()
            loss=loss_function(log_prob_pred,log_prob)
        elif path_model=='random_walk':
            loss=torch.zeros(1)
            for e in path: 
                loss-=torch.log(edge_probs_pred[e[0]][e[1]])
            #print (loss)    
        
        batch_loss+=loss
        training_loss+=loss
        if iter_n%batch_size==(batch_size-1):
            #print ("Loss: ", loss)
            batch_loss.backward(retain_graph=True)
            optimizer.step()
            #print ("GCN1:", list(net2.parameters())[0].grad)
            batch_loss=torch.zeros(1)
            

        #defender_utility=calculateDefenderUtility(net2, test_data)

    print ("Testing performance AFTER training:")
    testModel(test_data,net2,path_model, omega=omega)
    
    '''
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
    '''
    return net2



def testModel(test_data, net2, path_model, omega=4):
    
    time1=time.time()
    # COMPUTE TEST LOSS
    total_loss=0.0    
    for iter_n in range(len(test_data)):
        if path_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=test_data[iter_n]
        elif path_model=='random_walk':
            G,Fv, coverage_prob, phi, path=test_data[iter_n]
            #print("Lenghts:", len(path), len(getSimplePath(G,path)), path,getSimplePath(G,path) )
            path=getSimplePath(G,path)
        
        


        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
        
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        edge_probs_pred=generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
        
        loss=torch.zeros(1)
        if path_model=='random_walk_distribution':
            log_prob_pred=torch.zeros(1)
            for e in path: 
                log_prob_pred-=torch.log(edge_probs_pred[e[0]][e[1]]) 
            loss_function=nn.MSELoss()
            loss=loss_function(log_prob_pred,log_prob)
            
        elif path_model=='random_walk':
            for e in path: 
                loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        
        total_loss+=loss
        #print ("Loss: ", loss)
    print("Testing loss per sample:", total_loss/len(test_data))
  
    time2=time.time()
    if time_analysis:
        cprint (("TESTING TIME: ", time2-time1),'red')
    # COMPUTE THE EXPECTED DEFENDER UTILITY  
    total_ideal_defender_utility=0.0
    total_pred_defender_utility=0.0
    path_specific_defender_utility=0.0
    
    for iter_n in range(len(test_data)):
        
        if path_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi_true, path, log_prob=test_data[iter_n]
        elif path_model=='random_walk':
            G,Fv, coverage_prob, phi_true, path=test_data[iter_n]
        
        source=G.graph['source']
        target=G.graph['target']
        
        #print ("This point: ", len(list(G.nodes())), len(Fv))
        # TODO: add these two in createGraph() method while creating random graphs
        budget=G.graph['budget'] 
        U=G.graph['U']
        initial_distribution=G.graph['initial_distribution']
        
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float)
        #print ("This point: ", len(list(G.nodes())), len(Fv), len(Fv_torch), len(A_torch), len(A))
        phi_pred=net2(Fv_torch, A_torch).view(-1).detach().numpy()
        phi_true=phi_true.detach().numpy()
        pred_optimal_coverage=get_optimal_coverage_prob(G, phi_pred, U, initial_distribution, budget, omega=omega)
        #print (pred_optimal_coverage)
        pred_optimal_coverage=pred_optimal_coverage['x']
        ideal_optimal_coverage=get_optimal_coverage_prob(G, phi_true, U, initial_distribution, budget, omega=omega)['x']
        
        total_pred_defender_utility+=   -(objective_function(pred_optimal_coverage,G, phi_true, U,initial_distribution, omega=omega))
        total_ideal_defender_utility+=  -(objective_function(ideal_optimal_coverage,G, phi_true, U,initial_distribution, omega=omega))
        
        
        # PATH SPECIFIC DEF UTILITY
        target= (path[-1])[1]
        u_target=G.node[target]['utility']
        u_caught=U[-1]
        
        
        N=nx.number_of_nodes(G)
        coverage_prob_matrix=np.zeros((N,N))
        for i, e in enumerate(list(G.edges())):
            #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
            coverage_prob_matrix[e[0]][e[1]]=pred_optimal_coverage[i]
            coverage_prob_matrix[e[1]][e[0]]=pred_optimal_coverage[i]
        
        prob_reaching_target=1.0
        for e in path: 
            prob_reaching_target*=(1.0-coverage_prob_matrix[e[0]][e[1]])
        
        E_attacker_utility=u_target*prob_reaching_target+u_caught*(1.0-prob_reaching_target)
        path_specific_defender_utility-=E_attacker_utility
    time3=time.time()
    if time_analysis:
        cprint (("DEFENDER UTILITY CALCULATION: ", time3-time2), 'red')
        
        
    print("Defender utility: ideal/model/path_specific: ", total_ideal_defender_utility, total_pred_defender_utility, path_specific_defender_utility)
    
    return    


if __name__=='__main__':
    
    
    
    time1 =time.time()
    
    time_analysis=False
    path_model_type='random_walk'
    feature_size=5
    OMEGA=4
    
    GRAPH_N_LOW=16
    GRAPH_N_HIGH=18
    GRAPH_E_PROB_LOW=0.1
    GRAPH_E_PROB_HIGH=0.3
    
    TRAINING_GRAPHS=100
    SAMPLES_PER_TRAINING_GRAPH=50
    TESTING_GRAPHS=1000
    SAMPLES_PER_TESTING_GRAPH=1
    
    N_EPOCHS=100
    LR=0.0005
    BATCH_SIZE= 200
    OPTIMIZER='adam'
    
      
    train_data, test_data=generateSyntheticData(feature_size, path_type=path_model_type, 
                        n_training_graphs=TRAINING_GRAPHS, n_testing_graphs=TESTING_GRAPHS, 
                        training_samples_per_graph=SAMPLES_PER_TRAINING_GRAPH,
                        testing_samples_per_graph=SAMPLES_PER_TESTING_GRAPH,
                        fixed_graph=False, omega=OMEGA,
                        N_low=GRAPH_N_LOW, N_high=GRAPH_N_HIGH, e_low=GRAPH_E_PROB_LOW, e_high=GRAPH_E_PROB_HIGH)
    
    time2 =time.time()
    if time_analysis:
        cprint (("DATA GENERATION: ", time2-time1), 'red')
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)

    print ("Data length train/test:", len(train_data), len(test_data))
    
    time3=time.time()
    # Learn the neural networks:
    if path_model_type=='simple_paths':
        net2=learnPathProbs_simple(train_data,test_data)
    elif path_model_type=='random_walk':
        net2=learnEdgeProbs_simple(train_data,test_data, path_model=path_model_type,
                                   lr=LR,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
                                   optimizer=OPTIMIZER, omega=OMEGA)
    elif path_model_type=='random_walk_distribution':
        net2=learnEdgeProbs_simple(train_data,test_data, path_model=path_model_type,
                                   lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE, 
                                   optimizer=OPTIMIZER, omega=OMEGA)
    time4=time.time()
    if time_analysis:
        cprint (("TOTAL TRAINING+TESTING TIME: ", time4-time3), 'red')
    #print ("Now running: ", "Large graphs sizes")    
    all_params={"Path type": path_model_type,
                "Number of Epochs: ": N_EPOCHS, 
                "Learning rate: ": LR, 
                "Batch size: ": BATCH_SIZE,
                "Optimizer": OPTIMIZER, 
                "Omega ": OMEGA,
                "Graph size (nodes)": (GRAPH_N_LOW, GRAPH_N_HIGH),
                "Training data size (#graphs, #samples)": (TRAINING_GRAPHS, SAMPLES_PER_TRAINING_GRAPH),
                "Testing data size (#graphs, #samples)": (TESTING_GRAPHS, SAMPLES_PER_TESTING_GRAPH),
                "Running time": time4-time3} 
    
    cprint (all_params, 'green')
                
        
    
