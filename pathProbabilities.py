# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:49:05 2019

@author: Aditya 
"""

import torch
import torch.optim as optim
import time 
from termcolor import cprint
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

from gcn import * 
from graphData import *
from coverageProbability import *
from obsoleteCode import *
from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form
import qpthlocal

def get_one_hot(value, n_values):
    
    t=torch.zeros(n_values)
    
def plotEverything(all_params,tr_loss, test_loss, entire_graph_def_u, path_def_u):
    
    learning_model=all_params['learning model']
    
    # Training loss
    fig1= plt.figure()
    x=range(1,len(tr_loss))
    plt.plot(x, tr_loss[1:], label='Training loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.title("Training loss plot for "+learning_model)
    plt.legend()
    plt.show()
        
    # Testing loss
    fig2= plt.figure()
    x=range(len(test_loss))
    plt.plot(x, test_loss, label='Testing loss')
        
    plt.xlabel('Epochs')
    plt.ylabel('Testing Loss')


    plt.title("Testing loss plot for "+learning_model)
    plt.legend()
    plt.show()
    
    # Entire graph Defender utility
    fig3= plt.figure()
    x=range(len(entire_graph_def_u))
    y_ideal=[entire_graph_def_u[i][0] for i in x]
    y_model=[entire_graph_def_u[i][1] for i in x]
    plt.plot(x, y_ideal, label='Ideal')
    plt.plot(x, y_model, label='ML Model')
    
    plt.xlabel('Epochs')
    plt.ylabel('Defender utility')
    
    plt.title("Overall defender utiilty for "+learning_model)
    plt.legend()
    plt.show()
    
    # Path specific defender utility
    fig4= plt.figure()
    x=range(len(path_def_u))
    y_ideal=[path_def_u[i][0] for i in x]
    y_model=[path_def_u[i][1] for i in x]
    plt.plot(x, y_ideal, label='Ideal')
    plt.plot(x, y_model, label='ML Model')
    
    plt.xlabel('Epochs')
    plt.ylabel('Defender utility')
    
    plt.title("Path specific defender utiilty for "+learning_model)
    plt.legend()
    plt.show()

    return
    

def learnEdgeProbs_simple(train_data, test_data, n_testing_graphs, lr=0.1, learning_model='random_walk'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage'):

    
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
   
    training_loss_list=[]
    testing_loss_list=[]
    path_defender_utility_list=[]
    entire_defender_utility_list=[]
    n_iterations=n_epochs*len(train_data)
    
    print ("Training...") 
    time2=time.time()
    
    
    ######################################################
    #                   TRAINING LOOP
    ######################################################
    training_loss=torch.zeros(1)
    batch_loss=torch.zeros(1)
    phi_loss=0.0
    phi_corr=0.0
    time3=time.time()


    for iter_n in range(n_iterations):
        
        optimizer.zero_grad()
        
        ################################### Print stuff after every epoch 
        if iter_n%len(train_data)==0:
            np.random.shuffle(train_data)
            print("Epoch number/Training loss/ Training loss per sample/Phi Loss/corr_coeff: ", 
                  iter_n/len(train_data),training_loss.item(), training_loss.item()/len(train_data), phi_loss, phi_corr/len(train_data))
            
            ################################### Compute performance on test data
            defender_utility, testing_loss=testModel(n_testing_graphs,test_data,net2,learning_model, omega=omega, defender_utility_computation=True)
            
            
            path_defender_utility_list.append(defender_utility['path'])
            entire_defender_utility_list.append(defender_utility['entire'])
            testing_loss_list.append(testing_loss.item())
            training_loss_list.append((training_loss.item())/len(train_data))
            ################################## Reinitialize to 0 
            training_loss=torch.zeros(1)
            phi_loss=0.0
            phi_corr=0.0
            np.random.shuffle(train_data)
        
            time4=time.time()
            if time_analysis:
                cprint (("TIME FOR THIS EPOCH:", time4-time3),'red')
            time3=time4
            
        
        
        ################################### Gather data based on learning model
        if learning_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=train_data[iter_n%len(train_data)]
        elif learning_model=='empirical_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=train_data[iter_n%len(train_data)]
        else:
            G,Fv, coverage_prob, phi, path=train_data[iter_n%len(train_data)]
            # path=getSimplePath(G, path)
        
        ################################### Compute edge probabilities
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
    
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        transition_probs_pred = phi2prob(G, phi_pred)
        #print ("PHI PRED:", phi_pred)
        #print ("GCN1:", list(net2.parameters())[0].grad)
        #print ("PHI ACTUAL:", phi)
        
        edge_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
        
        ################################### Compute loss
        if learning_model=='random_walk_distribution':
            log_prob_pred=torch.zeros(1)
            for e in path: 
                log_prob_pred-=torch.log(edge_probs_pred[e[0]][e[1]]) 
            loss_function=nn.MSELoss()
            loss=loss_function(log_prob_pred,log_prob)

        elif learning_model=='empirical_distribution':

            log_prob_pred=torch.zeros(1)
            for e in path: 
                log_prob_pred-=torch.log(edge_probs_pred[e[0]][e[1]]) 
            loss_function=nn.MSELoss()
            loss=loss_function(log_prob_pred,log_prob)

        elif learning_model=='random_walk':
            loss=torch.zeros(1)
            for e in path: 
                loss-=torch.log(edge_probs_pred[e[0]][e[1]])
            #print (loss)    
        
        # COMPUTE DEFENDER UTILITY 
        if not(training_method == "two-stage"):
            single_data = train_data[iter_n % len(train_data)]
            def_obj = getDefUtility(single_data, transition_probs_pred, learning_model, omega=omega, verbose=False)

        # backpropagate using loss when training two-stage and using -defender utility when training end-to-end
        batch_loss += loss if training_method == "two-stage" else -def_obj
        training_loss+=loss
        #phi_loss+=torch.norm((phi-torch.mean(phi))-(phi_pred-torch.mean(phi_pred))).item()
        #phi_corr+=pearsonr(torch.tensor(phi-torch.mean(phi)), torch.tensor(phi_pred-torch.mean(phi_pred)))[0]
        if iter_n%batch_size==(batch_size-1):
            #print ("Loss: ", loss)
            batch_loss.backward(retain_graph=True)
            optimizer.step()
            #print ("GCN1:", list(net2.parameters())[0].grad)
            batch_loss=torch.zeros(1)
        
    return net2 ,training_loss_list, testing_loss_list, path_defender_utility_list, entire_defender_utility_list
    


def getDefUtility(single_data, transition_probs_pred, path_model, omega=4, verbose=False):
    if path_model=='random_walk_distribution':
        G, Fv, coverage_prob, phi_true, path, log_prob = single_data
    elif path_model=='empirical_distribution':
        G, Fv, coverage_prob, phi_true, path, log_prob = single_data
    elif path_model=='random_walk':
        G, Fv, coverage_prob, phi_true, path = single_data
    
    budget = G.graph['budget']
    U = torch.Tensor(G.graph['U'])
    initial_distribution = torch.Tensor(G.graph['initial_distribution'])
    transition_probs_true = phi2prob(G, phi_true)
    options = {"maxiter": 100, "disp": verbose}

    m = G.number_of_edges()
    G_matrix = torch.cat((-torch.eye(m), torch.eye(m), torch.ones(1,m)))
    h_matrix = torch.cat((torch.zeros(m), torch.ones(m), torch.Tensor([budget])))
    # print(G_matrix.shape)
    # print(h_matrix.shape)

    while True:
        pred_optimal_res = get_optimal_coverage_prob(G, transition_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options)
        pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
        qp_solver = qpthlocal.qp.QPFunction(verbose=verbose, solver=qpthlocal.qp.QPSolvers.GUROBI,
                                       zhats=None, slacks=None, nus=None, lams=None)

        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, transition_probs_pred, U, initial_distribution, omega=omega)
        Q_sym = (Q + Q.t()) / 2
        eigenvalues, eigenvectors = np.linalg.eig(Q_sym)
        negative_eigenvalues = np.array(eigenvalues) - 0.1
        Q_regularized = Q_sym - torch.eye(m) * min(0, min(eigenvalues)-0.1)
        # Q_regularized = 5 * torch.eye(m)
        is_symmetric = np.allclose(Q_sym.numpy(), Q_sym.numpy().T)
        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, transition_probs_pred, U, initial_distribution, omega=omega, lib=torch)
        p = jac.view(1, -1) - pred_optimal_coverage @ Q_regularized

        coverage_qp_solution = qp_solver(Q_regularized, p, G_matrix, h_matrix, torch.Tensor(), torch.Tensor())[0]

        # print(Q, p)
        if verbose:
            print(pred_optimal_res)
            print("Minimum Eigenvalue: {}".format(min(eigenvalues)))
            print("Hessian: {}".format(Q_sym))
            print("Gradient: {}".format(jac))
            print("Eigen decomposition: {}".format(np.linalg.eig(Q_sym.detach().numpy())[0]))
            print("Eigen decomposition: {}".format(np.linalg.eig(Q_regularized.detach().numpy())[0]))
            print("objective value (SLSQP): {}".format(objective_function_matrix_form(pred_optimal_coverage, G, transition_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega)))
            print("objective value (QP): {}".format(objective_function_matrix_form(coverage_qp_solution, G, transition_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega)))
            print(pred_optimal_coverage, torch.sum(pred_optimal_coverage))
            print(coverage_qp_solution, torch.sum(coverage_qp_solution))
            print("Solution difference:", torch.norm(pred_optimal_coverage - coverage_qp_solution))


        break # TODO


    pred_defender_utility  = -(objective_function_matrix_form(coverage_qp_solution,  G, transition_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega))
    # ideal_defender_utility += (objective_function_matrix_form(ideal_optimal_coverage, G, phi_true, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega))

    return pred_defender_utility

    

def testModel(n_testing_graphs, test_data, net2, learning_model, omega=4, defender_utility_computation=True):
    
    #return 1.0, 1.0
    
    time1=time.time()
    
    # COMPUTE TEST LOSS
    total_loss=0.0    
    for iter_n in range(len(test_data)):
        if learning_model=='random_walk_distribution':
            G,Fv, coverage_prob, phi, path, log_prob=test_data[iter_n]
            
        elif learning_model=='empirical_distribution':
            G,Fv, coverage_prob, phi, path=test_data[iter_n]
        
        elif learning_model=='random_walk':
            G,Fv, coverage_prob, phi, path=test_data[iter_n]
            # path=getSimplePath(G,path)
                
        A=nx.to_numpy_matrix(G)
        A_torch = torch.as_tensor(A, dtype=torch.float) 
        source=G.graph['source']
        target=G.graph['target']
        
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        phi_pred=net2(Fv_torch, A_torch).view(-1)
        
        edge_probs_pred=generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
        edge_probs_pred.detach()
        
        loss=torch.zeros(1)
        if learning_model=='random_walk_distribution':
            log_prob_pred=torch.zeros(1)
            for e in path: 
                log_prob_pred-=torch.log(edge_probs_pred[e[0]][e[1]]) 
            loss_function=nn.MSELoss()
            loss=loss_function(log_prob_pred,log_prob)
        
        elif learning_model=='empirical_distribution':
            for e in path: 
                loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        
        elif learning_model=='random_walk':
            for e in path: 
                loss-=torch.log(edge_probs_pred[e[0]][e[1]])
        
        total_loss+=loss
        #print ("Loss: ", loss)
    testing_loss=total_loss/len(test_data)
    print("Testing loss per sample:", testing_loss)
  
    time2=time.time()
    if time_analysis:
        cprint (("TESTING TIME: ", time2-time1),'red')
    

    ######################################################################################################################
    ############################################################## EXPECTED DEFENDER UTILITY COMPUTATION #################    
    ######################################################################################################################
    if defender_utility_computation:

        total_ideal_defender_utility=0.0
        total_pred_defender_utility=0.0
        total_random_defender_utility=0.0
        pred_path_specific_defender_utility=0.0
        ideal_path_specific_defender_utility=0.0
        random_path_specific_defender_utility=0.0
        samples_per_testing_graph=len(test_data)/n_testing_graphs
        for iter_n in range(len(test_data)):
        #for iter_n in range(len(n_testing_graphs)):
            if iter_n%samples_per_testing_graph==0:
                if learning_model=='random_walk_distribution':
                    G,Fv, coverage_prob, phi_true, path, log_prob=test_data[iter_n]
                elif learning_model=='empirical_distribution':
                    G,Fv, coverage_prob, phi_true, path=test_data[iter_n]
                elif learning_model=='random_walk':
                    G,Fv, coverage_prob, phi_true, path=test_data[iter_n]
                
                source=G.graph['source']
                target=G.graph['target']        
                budget=G.graph['budget'] 
                U=G.graph['U']
                initial_distribution=G.graph['initial_distribution']
                
                ###################################################### DEFENDER UTILITY COMPUTATION
                # This code block computes: 
                # Expected ideal def utility, model def utility, random model defender utility, path specific def utility
                ######################################################
                Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
                A=nx.to_numpy_matrix(G)
                A_torch = torch.as_tensor(A, dtype=torch.float)
                #print ("This point: ", len(list(G.nodes())), len(Fv), len(Fv_torch), len(A_torch), len(A))
                phi_pred=net2(Fv_torch, A_torch).view(-1).detach()
                phi_true=phi_true.detach()
                transition_probs_pred = phi2prob(G, phi_pred)
                transition_probs_true = phi2prob(G, phi_true)
                
                #######################################################
                pred_optimal_coverage=get_optimal_coverage_prob(G, transition_probs_pred, U, initial_distribution, budget, omega=omega)['x']
                #print (pred_optimal_coverage)
                #pred_optimal_coverage=pred_optimal_coverage['x']
                ideal_optimal_coverage=get_optimal_coverage_prob(G, transition_probs_true, U, initial_distribution, budget, omega=omega)['x']
                random_coverage_prob=np.random.rand(nx.number_of_edges(G))
                random_coverage_prob=budget*(random_coverage_prob/np.sum(random_coverage_prob))
                
                #######################################################
                total_pred_defender_utility   += -(objective_function_matrix_form(pred_optimal_coverage,  G, transition_probs_true, U,initial_distribution, omega=omega))
                total_ideal_defender_utility  += -(objective_function_matrix_form(ideal_optimal_coverage, G, transition_probs_true, U,initial_distribution, omega=omega))
                total_random_defender_utility += -(objective_function_matrix_form(random_coverage_prob,   G, transition_probs_true, U,initial_distribution, omega=omega))
                
                if total_ideal_defender_utility<total_pred_defender_utility:
                    print("DEFENDER UTILITY ERROR")
                else:
                    pass
                    #print ("DEFEFNDER UTILITY OK")
            
            
            ##################################### Compute PATH SPECIFIC DEF UTILITY (not over entire graph, only over the sampled path)
            target= (path[-1])[1]
            u_target=G.node[target]['utility']
            u_caught=U[-1]
            N=nx.number_of_nodes(G)
            ideal_coverage_prob_matrix=np.zeros((N,N))
            pred_coverage_prob_matrix=np.zeros((N,N))
            random_coverage_prob_matrix=np.zeros((N,N))
            for i, e in enumerate(list(G.edges())):
                #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
                pred_coverage_prob_matrix[e[0]][e[1]]=pred_optimal_coverage[i]
                pred_coverage_prob_matrix[e[1]][e[0]]=pred_optimal_coverage[i]
                ideal_coverage_prob_matrix[e[0]][e[1]]=ideal_optimal_coverage[i]
                ideal_coverage_prob_matrix[e[1]][e[0]]=ideal_optimal_coverage[i]
                random_coverage_prob_matrix[e[0]][e[1]]=random_coverage_prob[i]
                random_coverage_prob_matrix[e[1]][e[0]]=random_coverage_prob[i]
            
            pred_prob_reaching_target=1.0
            ideal_prob_reaching_target=1.0
            random_prob_reaching_target=1.0
            for e in path: 
                pred_prob_reaching_target*=(1.0-pred_coverage_prob_matrix[e[0]][e[1]])
                ideal_prob_reaching_target*=(1.0-ideal_coverage_prob_matrix[e[0]][e[1]])
                random_prob_reaching_target*=(1.0-random_coverage_prob_matrix[e[0]][e[1]])
            
            E_attacker_utility_pred=u_target*pred_prob_reaching_target+u_caught*(1.0-pred_prob_reaching_target)
            E_attacker_utility_ideal=u_target*ideal_prob_reaching_target+u_caught*(1.0-ideal_prob_reaching_target)
            E_attacker_utility_random=u_target*random_prob_reaching_target+u_caught*(1.0-random_prob_reaching_target)
            pred_path_specific_defender_utility-=E_attacker_utility_pred
            ideal_path_specific_defender_utility-=E_attacker_utility_ideal
            random_path_specific_defender_utility-=E_attacker_utility_random
        
        
        
        # Normalize defender utility to per sample:
        total_ideal_defender_utility/=n_testing_graphs    
        total_pred_defender_utility/=n_testing_graphs
        total_random_defender_utility/=n_testing_graphs
        ideal_path_specific_defender_utility/=len(test_data)
        pred_path_specific_defender_utility/=len(test_data)
        random_path_specific_defender_utility/=len(test_data)
        
        time3=time.time()
        if time_analysis:
            cprint (("DEFENDER UTILITY CALCULATION: ", time3-time2), 'red')
            
            
        cprint(("Defender utility: ideal/model/random: ", total_ideal_defender_utility, total_pred_defender_utility, total_random_defender_utility), color='blue') 
        cprint (("Path specific utility: ideal/model/random: ",ideal_path_specific_defender_utility, pred_path_specific_defender_utility, random_path_specific_defender_utility),color='blue')
    
        defender_utility={'entire':(total_ideal_defender_utility,total_pred_defender_utility,total_random_defender_utility),
                          'path':(ideal_path_specific_defender_utility,pred_path_specific_defender_utility,random_path_specific_defender_utility)
                          }
    return defender_utility, testing_loss


if __name__=='__main__':
    
    
    ############################# Parameters and settings:
    time1 =time.time()
    
    time_analysis=True

    plot_everything=True
    learning_model_type= "empirical_distribution" # 'random_walk_distribution' 
    training_mode = 0
    training_method = 'two-stage' if training_mode == 0 else 'decision-focused' # 'two-stage' or 'decision-focused'
    feature_size=50
    OMEGA=0

    GRAPH_N_LOW=16
    GRAPH_N_HIGH=18
    GRAPH_E_PROB_LOW=0.2
    GRAPH_E_PROB_HIGH=0.3
    
    TRAINING_GRAPHS=20
    SAMPLES_PER_TRAINING_GRAPH=100
    TESTING_GRAPHS=20
    SAMPLES_PER_TESTING_GRAPH=20
    
    N_EPOCHS=20
    LR=0.0005
    BATCH_SIZE= 100
    OPTIMIZER='adam'    
    DEFENDER_BUDGET=0.01 # This means the budget (sum of coverage prob) is <= DEFENDER_BUDGET*Number_of_edges 

    ###############################
      
    ############################### Data genaration:
    train_data, test_data=generateSyntheticData(feature_size, path_type=learning_model_type, 
                        n_training_graphs=TRAINING_GRAPHS, n_testing_graphs=TESTING_GRAPHS, 
                        training_samples_per_graph=SAMPLES_PER_TRAINING_GRAPH,
                        testing_samples_per_graph=SAMPLES_PER_TESTING_GRAPH,
                        fixed_graph=False, omega=OMEGA,
                        N_low=GRAPH_N_LOW, N_high=GRAPH_N_HIGH, e_low=GRAPH_E_PROB_LOW, e_high=GRAPH_E_PROB_HIGH,
                        budget=DEFENDER_BUDGET)
    
    time2 =time.time()
    if time_analysis:
        cprint (("DATA GENERATION: ", time2-time1), 'red')
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)
    print ("Data length train/test:", len(train_data), len(test_data))
    ##############################

    ############################## Training the ML models:    
    time3=time.time()
    # Learn the neural networks:
    if learning_model_type=='simple_paths':
        net2=learnPathProbs_simple(train_data,test_data)
    elif learning_model_type=='random_walk':
        net2, tr_loss, test_loss, path_def_u, entire_graph_def_u=learnEdgeProbs_simple(
                                    train_data,test_data, 
                                    learning_model=learning_model_type,
                                    lr=LR,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
                                    optimizer=OPTIMIZER, omega=OMEGA,training_method=training_method)
    elif learning_model_type=='random_walk_distribution':
        net2,tr_loss, test_loss, path_def_u, entire_graph_def_u=learnEdgeProbs_simple(
                                    train_data,test_data, TESTING_GRAPHS,
                                    learning_model=learning_model_type,
                                    lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE, 
                                    optimizer=OPTIMIZER, omega=OMEGA)
    elif learning_model_type=='empirical_distribution':
        net2, tr_loss, test_loss, path_def_u, entire_graph_def_u=learnEdgeProbs_simple(
                                    train_data,test_data, TESTING_GRAPHS,
                                    learning_model=learning_model_type,
                                    lr=LR,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
                                    optimizer=OPTIMIZER, omega=OMEGA,training_method=training_method)

    time4=time.time()
    if time_analysis:
        cprint (("TOTAL TRAINING+TESTING TIME: ", time4-time3), 'red')
    #############################
    
    ############################# Print the summary:
    #print ("Now running: ", "Large graphs sizes")    
    all_params={"learning model": learning_model_type,
                "Training method": training_method,
                "Number of Epochs: ": N_EPOCHS, 
                "Learning rate: ": LR, 
                "Batch size: ": BATCH_SIZE,
                "Optimizer": OPTIMIZER, 
                "Omega ": OMEGA,
                "Graph size (nodes)": (GRAPH_N_LOW, GRAPH_N_HIGH),
                "Graph edge prob: ": (GRAPH_E_PROB_LOW, GRAPH_E_PROB_HIGH),
                "Training data size (#graphs, #samples)": (TRAINING_GRAPHS, SAMPLES_PER_TRAINING_GRAPH),
                "Testing data size (#graphs, #samples)": (TESTING_GRAPHS, SAMPLES_PER_TESTING_GRAPH),
                "Running time": time4-time3, 
                "Path defender utility:": (path_def_u[0],path_def_u[-1]),
                "Entire graph defender utility:":(entire_graph_def_u[0],entire_graph_def_u[-1]),
                "Test Loss:": test_loss} 
    
    cprint (all_params, 'green')
    #############################            
    if plot_everything:
        plotEverything(all_params,tr_loss, test_loss, entire_graph_def_u, path_def_u)
        
