# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:49:05 2019

@author: Aditya 
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time 
from termcolor import cprint
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import tqdm
import argparse

from gcn import GCNPredictionNet2
from graphData import *
from coverageProbability import *
from obsoleteCode import *
from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form
import qpthlocal

def plotEverything(all_params,tr_loss, test_loss, training_graph_def_u, testing_graph_def_u, filepath="figure/"):
    learning_model=all_params['learning model']

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    fig.text(0.5, 0.04, '# epochs', ha='center', va='center')
    fig.text(0.06, 0.75, 'KL-divergence loss', ha='center', va='center', rotation='vertical')
    fig.text(0.06, 0.25, 'Defender utility', ha='center', va='center', rotation='vertical')

    epochs = len(tr_loss) - 1
    x=range(-1, epochs)

    # Training loss
    ax1.plot(x, tr_loss, label='Training loss')
    ax1.set_title("Training")
    ax1.legend()
        
    # Testing loss
    ax2.plot(x, test_loss, label='Testing loss')
    ax2.set_title("Testing")
    ax2.legend()

    # Entire training graph Defender utility
    ax3.plot(x, training_graph_def_u, label='ML Model')
    ax3.legend()

    # Entire testing graph Defender utility
    ax4.plot(x, testing_graph_def_u, label='ML Model')
    ax4.legend()

    plt.savefig('{}'.format(filepath))
    
    return
    

def learnEdgeProbs_simple(train_data, test_data, f_save, lr=0.1, learning_model='random_walk_distribution'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage'):

    
    time1=time.time()
    net2= GCNPredictionNet2(feature_size)
    net2.train()
    if optimizer=='adam':
        optimizer=optim.Adam(net2.parameters(), lr=lr)
    elif optimizer=='sgd':
        optimizer=optim.SGD(net2.parameters(), lr=lr)
    elif optimizer=='adamax':
        optimizer=optim.Adamax(net2.parameters(), lr=lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min')    
   
    training_loss_list, testing_loss_list = [], []
    training_defender_utility_list, testing_defender_utility_list = [], []
    
    print ("Training...") 
    time2=time.time()
    
    ######################################################
    #                   TRAINING LOOP
    ######################################################
    time3 = time.time()

    f_save.write("mode, epoch, average loss, defender utility\n")

    for epoch in range(-1, n_epochs):
        for mode in ["training", "testing"]:
            if mode == "training":
                dataset = train_data
                epoch_loss_list = training_loss_list
                epoch_def_list  = training_defender_utility_list
                if epoch > 0:
                    net2.train()
                else:
                    net2.eval()
            else:
                dataset = test_data
                epoch_loss_list = testing_loss_list
                epoch_def_list  = testing_defender_utility_list
                net2.eval()

            loss_list, def_obj_list = [], []
            batch_loss = 0
            for iter_n in tqdm.trange(len(dataset)):
                ################################### Gather data based on learning model
                G, Fv, coverage_prob, phi_true, unbiased_probs_true, path_list, log_prob = dataset[iter_n]
                
                ################################### Compute edge probabilities
                Fv_torch   = torch.as_tensor(Fv, dtype=torch.float)
                edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
                phi_pred   = net2(Fv_torch, edge_index).view(-1) if epoch >= 0 else phi_true # when epoch < 0, testing the optimal loss and defender utility

                unbiased_probs_pred = phi2prob(G, phi_pred)
                biased_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
                
                ################################### Compute loss
                log_prob_pred = torch.zeros(1)
                for path in path_list:
                    for e in path: 
                        log_prob_pred -= torch.log(biased_probs_pred[e[0]][e[1]])
                log_prob_pred /= len(path_list)
                loss = log_prob_pred - log_prob

                # COMPUTE DEFENDER UTILITY 
                single_data = dataset[iter_n]
                def_obj, def_coverage = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, verbose=False)

                loss_list.append(loss.item())
                def_obj_list.append(def_obj.item())

                # backpropagate using loss when training two-stage and using -defender utility when training end-to-end
                batch_loss += loss if training_method == "two-stage" else -def_obj

                if (iter_n%batch_size == (batch_size-1)) and (epoch > 0) and (mode == "training"):
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss = 0

            # Storing loss and defender utility
            epoch_loss_list.append(np.mean(loss_list))
            epoch_def_list.append(np.mean(def_obj_list))

            ################################### Print stuff after every epoch 
            np.random.shuffle(dataset)
            print("Mode: {}/ Epoch number: {}/ Average training loss: {}/ Average defender Objective: {}".format(
                  mode, epoch, np.mean(loss_list), np.mean(def_obj_list)))

            f_save.write("{}, {}, {}, {}\n".format(mode, epoch, np.mean(loss_list), np.mean(def_obj_list)))

            # defender_utility, testing_loss=testModel(test_data,net2, learning_model, omega=omega, defender_utility_computation=True)

        
        
        time4=time.time()
        if time_analysis:
            cprint (("TIME FOR THIS EPOCH:", time4-time3),'red')
        time3=time4
            
    return net2 ,training_loss_list, testing_loss_list, training_defender_utility_list, testing_defender_utility_list
    


def getDefUtility(single_data, unbiased_probs_pred, path_model, omega=4, verbose=False):
    if path_model=='random_walk_distribution':
        G, Fv, coverage_prob, phi_true, unbiased_probs_true, path_list, log_prob = single_data
    elif path_model=='empirical_distribution':
        G, Fv, coverage_prob, phi_true, unbiased_probs_true, path_list, log_prob = single_data
    
    budget = G.graph['budget']
    U = torch.Tensor(G.graph['U'])
    initial_distribution = torch.Tensor(G.graph['initial_distribution'])
    options = {"maxiter": 100, "disp": verbose}

    m = G.number_of_edges()
    G_matrix = torch.cat((-torch.eye(m), torch.eye(m), torch.ones(1,m)))
    h_matrix = torch.cat((torch.zeros(m), torch.ones(m), torch.Tensor([budget])))

    # ========================== QP part ===========================
    for tmp_iter in range(10): # maximum 10 retries
        initial_coverage_prob = np.random.rand(nx.number_of_edges(G))
        initial_coverage_prob = initial_coverage_prob / np.sum(initial_coverage_prob) * budget * 0.1
        # initial_coverage_prob = np.zeros(nx.number_of_edges(G))

        pred_optimal_res = get_optimal_coverage_prob(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options, initial_coverage_prob=initial_coverage_prob)
        if pred_optimal_res["status"] != 0 and tmp_iter != 9:
            print("optimization failed... restart...")
            continue

        pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
        qp_solver = qpthlocal.qp.QPFunction(verbose=verbose, solver=qpthlocal.qp.QPSolvers.GUROBI,
                                       zhats=None, slacks=None, nus=None, lams=None)

        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, omega=omega)
        Q_sym = Q #+ Q.t()) / 2

        eigenvalues, eigenvectors = np.linalg.eig(Q_sym)
        eigenvalues = [x.real for x in eigenvalues]
        Q_regularized = Q_sym - torch.eye(m) * min(0, min(eigenvalues)-1)
        
        # debug only
        # print(eigenvalues)
        # new_eigenvalues, new_eigenvectors = np.linalg.eig(Q_regularized)
        # # new_eigenvalues = [x.real for x in new_eigenvalues]
        # print(new_eigenvalues)

        is_symmetric = np.allclose(Q_sym.numpy(), Q_sym.numpy().T)
        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, omega=omega, lib=torch)
        p = jac.view(1, -1) - pred_optimal_coverage @ Q_regularized

        coverage_qp_solution = qp_solver(0.5 * Q_regularized, p, G_matrix, h_matrix, torch.Tensor(), torch.Tensor())[0] # GUROBI version takes x^T Q x + x^T p; not 1/2 x^T Q x + x^T p
        # initial_coverage_prob = coverage_qp_solution.detach()

        # ======================= Defender Utility ========================
        pred_defender_utility  = -(objective_function_matrix_form(coverage_qp_solution,  G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega))

        # ========================= Error message =========================
        if (torch.norm(pred_optimal_coverage - coverage_qp_solution) > 0.001): # or pred_defender_utility > 0:
            print(pred_optimal_res)
            print("Minimum Eigenvalue: {}".format(min(eigenvalues)))
            print("Hessian: {}".format(Q_sym))
            print("Gradient: {}".format(jac))
            print("Eigen decomposition: {}".format(np.linalg.eig(Q_sym.detach().numpy())[0]))
            print("Eigen decomposition: {}".format(np.linalg.eig(Q_regularized.detach().numpy())[0]))
            print("objective value (SLSQP): {}".format(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega)))
            print("objective value (QP): {}".format(objective_function_matrix_form(coverage_qp_solution, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), omega=omega)))
            print(pred_optimal_coverage, torch.sum(pred_optimal_coverage))
            print(coverage_qp_solution, torch.sum(coverage_qp_solution))
            print("Solution difference:", torch.norm(pred_optimal_coverage - coverage_qp_solution))
            print(unbiased_probs_true)
            print(unbiased_probs_pred)

            QP_value   = p @ coverage_qp_solution  + 0.5 * coverage_qp_solution  @ Q_regularized @ coverage_qp_solution  # - jac @ pred_optimal_coverage + 0.5 * pred_optimal_coverage @ Q_sym @ pred_optimal_coverage
            true_value = p @ pred_optimal_coverage + 0.5 * pred_optimal_coverage @ Q_regularized @ pred_optimal_coverage # - jac @ pred_optimal_coverage + 0.5 * pred_optimal_coverage @ Q_sym @ pred_optimal_coverage
            print("QP value: {}".format(QP_value))
            print("true value: {}".format(true_value))

        else:
            break

    return pred_defender_utility, coverage_qp_solution

    

# def testModel(dataset, net2, learning_model, omega=4, defender_utility_computation=True):
#     
#     #return 1.0, 1.0
#     
#     time1=time.time()
#     
#     # COMPUTE TEST LOSS
#     total_loss=0.0    
# 
#     total_pred_defender_utility   = 0
#     total_ideal_defender_utility  = 0
#     total_random_defender_utility = 0
# 
#     for iter_n, single_data in enumerate(dataset):
#         G, Fv, coverage_prob, phi_true, unbiased_probs_true, path_list, log_prob = single_data
#         
#         U, initial_distribution, budget = G.graph['U'], G.graph['initial_distribution'], G.graph['budget']
#         
#         Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
#         edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
#         phi_pred=net2(Fv_torch, edge_index).view(-1)
#         
#         biased_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
#         unbiased_probs_pred = prob2unbiased(G, coverage_prob, biased_probs_pred, omega)
#         
#         loss=torch.zeros(1)
#         for path in path_list: 
#             for e in path:
#                 loss -= torch.log(biased_probs_pred[e[0]][e[1]])
#         loss /= len(path_list)
#         loss = loss - log_prob
# 
#         # ---------------- computing defender utility ------------------------
#         unbiased_probs_pred   = phi2prob(G, phi_pred)
#         unbiased_probs_true   = phi2prob(G, phi_true)
#         random_coverage_prob=np.random.rand(nx.number_of_edges(G))
#         random_coverage_prob=budget*(random_coverage_prob/np.sum(random_coverage_prob))
# 
#         total_pred_defender_utility   += getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, verbose=False)[0]
#         total_ideal_defender_utility  += getDefUtility(single_data, unbiased_probs_true, learning_model, omega=omega, verbose=False)[0]
#         total_random_defender_utility += -(objective_function_matrix_form(random_coverage_prob, G, unbiased_probs_true, U, initial_distribution, omega=omega))        
#         total_loss+=loss
# 
#     testing_loss=total_loss/len(dataset)
#     total_pred_defender_utility   = total_pred_defender_utility   / len(dataset)
#     total_ideal_defender_utility  = total_ideal_defender_utility  / len(dataset)
#     total_random_defender_utility = total_random_defender_utility / len(dataset)
#     print("Testing loss per sample:", testing_loss)
#   
#     time2=time.time()
#     if time_analysis:
#         cprint (("TESTING TIME: ", time2-time1),'red')
#     
#     cprint(("Defender utility: ideal/model/random: ", total_ideal_defender_utility, total_pred_defender_utility, total_random_defender_utility), color='blue') 
#     defender_utility={'entire testing': total_pred_defender_utility}
# 
#     return defender_utility, testing_loss


if __name__=='__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--fixed-graph', type=int, default=0, help='0 -> randomly generated, 1 -> first fixed graph, 2 -> second fixed graph')
    parser.add_argument('--prob', type=float, default=0.2, help='input the probability used as input of random graph generator')

    parser.add_argument('--learning-rate', type=float, default=0.005, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--feature-size', type=int, default=5, help='feature size of each node')
    parser.add_argument('--omega', type=float, default=4, help='risk aversion of the attacker')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')

    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')
    parser.add_argument('--number-graphs', type=int, default=1, help='number of different graphs in the dataset')
    parser.add_argument('--number-samples', type=int, default=100, help='number of samples per graph')
    parser.add_argument('--number-sources', type=int, default=2, help='number of randomly generated sources')
    parser.add_argument('--number-targets', type=int, default=2, help='number of randomly generated targets')

    parser.add_argument('--distribution', type=int, default=0, help='0 -> random walk distribution, 1 -> empirical distribution')
    parser.add_argument('--method', type=int, default=0, help='0 -> two-stage, 1 -> decision-focused')
    
    args = parser.parse_args()

    ############################# Parameters and settings:
    time1 =time.time()
    
    time_analysis=True

    plot_everything=True
    learning_mode = args.distribution
    learning_model_type = 'random_walk_distribution' if learning_mode == 0 else 'empirical_distribution'
    training_mode = args.method
    training_method = 'two-stage' if training_mode == 0 else 'decision-focused' # 'two-stage' or 'decision-focused'

    feature_size = args.feature_size
    OMEGA = args.omega

    GRAPH_N_LOW  = args.number_nodes
    GRAPH_N_HIGH = GRAPH_N_LOW + 1
    GRAPH_E_PROB_LOW  = args.prob
    GRAPH_E_PROB_HIGH = GRAPH_E_PROB_LOW
    
    NUMBER_OF_GRAPHS  = args.number_graphs
    SAMPLES_PER_GRAPH = args.number_samples
    EMPIRICAL_SAMPLES_PER_INSTANCE = 100
    NUMBER_OF_SOURCES = args.number_sources
    NUMBER_OF_TARGETS = args.number_targets
    
    N_EPOCHS = args.epochs
    LR = args.learning_rate # roughly 0.005 ~ 0.01 for two-stage; N/A for decision-focused
    BATCH_SIZE = 1
    OPTIMIZER = 'adam'
    DEFENDER_BUDGET = args.budget # This means the budget (sum of coverage prob) is <= DEFENDER_BUDGET*Number_of_edges 
    FIXED_GRAPH = args.fixed_graph
    GRAPH_TYPE = "random_graph" if FIXED_GRAPH == 0 else "fixed_graph"
    SEED = args.seed
    if SEED == 0:
        SEED = np.random.randint(1, 100000)

    ###############################
    date = "0629-1900"
    if FIXED_GRAPH == 0:
        filepath_data = "results/random/{}_{}_n{}_p{}_b{}.csv".format(date, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET)
        filepath_figure = "figures/random/{}_{}_n{}_p{}_b{}".format(date, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET)
    else:
        filepath_data = "results/fixed/{}_{}_test.csv".format(date, training_method)
        filepath_figure = "figures/fixed/{}_{}_test".format(date, training_method)

    f_save = open(filepath_data, 'a')
      
    ############################### Data genaration:
    train_data, test_data=generateSyntheticData(feature_size, path_type=learning_model_type, 
                        n_graphs=NUMBER_OF_GRAPHS, samples_per_graph=SAMPLES_PER_GRAPH, empirical_samples_per_instance=EMPIRICAL_SAMPLES_PER_INSTANCE,
                        fixed_graph=FIXED_GRAPH, omega=OMEGA,
                        N_low=GRAPH_N_LOW, N_high=GRAPH_N_HIGH, e_low=GRAPH_E_PROB_LOW, e_high=GRAPH_E_PROB_HIGH,
                        budget=DEFENDER_BUDGET, n_sources=NUMBER_OF_SOURCES, n_targets=NUMBER_OF_TARGETS,
                        random_seed=SEED)
    
    time2 =time.time()
    if time_analysis:
        cprint (("DATA GENERATION: ", time2-time1), 'red')
    np.random.shuffle(train_data)
    np.random.shuffle(train_data)

    print ("Training method: {}".format(training_method))
    print ("Data length train/test:", len(train_data), len(test_data))

    ############################## Training the ML models:    
    time3=time.time()
    # Learn the neural networks:
    net2, tr_loss, test_loss, training_graph_def_u, testing_graph_def_u=learnEdgeProbs_simple(
                                train_data, test_data, f_save,
                                learning_model=learning_model_type,
                                lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE, 
                                optimizer=OPTIMIZER, omega=OMEGA, training_method=training_method)

    time4=time.time()
    if time_analysis:
        cprint (("TOTAL TRAINING+TESTING TIME: ", time4-time3), 'red')

    f_save.close()

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
                "Data size (#graphs, #samples)": (NUMBER_OF_GRAPHS, SAMPLES_PER_GRAPH),
                "Running time": time4-time3, 
                "Entire graph defender utility:":(testing_graph_def_u[0],testing_graph_def_u[-1]),
                "Test Loss:": test_loss} 
    
    cprint (all_params, 'green')
    #############################            
    if plot_everything:
        plotEverything(all_params,tr_loss, test_loss, training_graph_def_u, testing_graph_def_u, filepath=filepath_figure)
        
