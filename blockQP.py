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
import qpth

from gcn import GCNPredictionNet2
from graphData import *
from derivative import *
# from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form

def learnEdgeProbs_simple(train_data, validate_data, test_data, f_save, f_time, f_summary, lr=0.1, learning_model='random_walk_distribution'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage', max_norm=0.1):
    
    net2= GCNPredictionNet2(feature_size)
    net2.train()
    if optimizer=='adam':
        optimizer=optim.Adam(net2.parameters(), lr=lr)
    elif optimizer=='sgd':
        optimizer=optim.SGD(net2.parameters(), lr=lr)
    elif optimizer=='adamax':
        optimizer=optim.Adamax(net2.parameters(), lr=lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8)
   
    training_loss_list, validating_loss_list, testing_loss_list = [], [], []
    training_defender_utility_list, validating_defender_utility_list, testing_defender_utility_list = [], [], []
    
    print ("Training...")
    beginning_time = time.time()
    time3 = time.time()

    f_save.write("mode, epoch, average loss, defender utility, simulated defender utility\n")

    pretrain_epochs = n_epochs
    for epoch in range(-1, n_epochs):
        for mode in ["training", "validating", "testing"]:
            if mode == "training":
                dataset = train_data
                epoch_loss_list = training_loss_list
                epoch_def_list  = training_defender_utility_list
                if epoch > 0:
                    net2.train()
                else:
                    net2.eval()
            elif mode == "validating":
                dataset = validate_data
                epoch_loss_list = validating_loss_list
                epoch_def_list  = validating_defender_utility_list
                net2.eval()
            elif mode == "testing":
                dataset = test_data
                epoch_loss_list = testing_loss_list
                epoch_def_list  = testing_defender_utility_list
                net2.eval()
            else:
                raise TypeError("Not valid mode: {}".format(mode))

            loss_list, def_obj_list, simulated_def_obj_list = [], [], [] 
            for iter_n in tqdm.trange(len(dataset)):
                random_value = np.random.random()
                ################################### Gather data based on learning model
                G, Fv, coverage_prob, phi_true, path_list, cut, log_prob, unbiased_probs_true = dataset[iter_n]
                
                ################################### Compute edge probabilities
                Fv_torch   = torch.as_tensor(Fv, dtype=torch.float)
                edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
                phi_pred   = net2(Fv_torch, edge_index).view(-1) if epoch >= 0 else phi_true # when epoch < 0, testing the optimal loss and defender utility
                # phi_pred.require_grad = True

                unbiased_probs_pred = phi2prob(G, phi_pred)
                biased_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
                
                ################################### Compute loss
                # loss = torch.norm((unbiased_probs_true - unbiased_probs_pred) * torch.Tensor(nx.adjacency_matrix(G).toarray()))
                log_prob_pred = torch.zeros(1)
                for path in path_list:
                    for e in path: 
                        log_prob_pred -= torch.log(biased_probs_pred[e[0]][e[1]])
                log_prob_pred /= len(path_list)
                loss = log_prob_pred - log_prob

                # COMPUTE DEFENDER UTILITY 
                single_data = dataset[iter_n]

                if mode == 'testing' or mode == "validating" or epoch <= 0: # or training_method == "two-stage" or epoch <= 0:
                    def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, verbose=False, training_mode=False, training_method=training_method) # feed forward only
                else:
                    if training_method == "two-stage":
                        def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, verbose=False, training_mode=False, training_method=training_method) # most time-consuming part
                    else:
                        def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, verbose=False, training_mode=True,  training_method=training_method) # most time-consuming part
                        
                        # =============== checking gradient manually ===============
                        # dopt_dphi = torch.Tensor(len(def_coverage), len(phi_pred))
                        # for i in range(len(def_coverage)):
                        #     grad_def_obj, grad_def_coverage, _ = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, verbose=False, training_mode=True,  training_method=training_method) # most time-consuming part
                        #     dopt_dphi[i] = torch.autograd.grad(grad_def_coverage[i], phi_pred, retain_graph=True)[0] # ith dimension
                        #     step_size = 0.01

                        # estimated_dopt_dphi = torch.Tensor(len(def_coverage), len(phi_pred))
                        # for i in range(len(phi_pred)):
                        #     new_phi_pred = phi_pred.clone()
                        #     new_phi_pred[i] += step_size
        
                        #     # validating
                        #     new_unbiased_probs_pred = phi2prob(G, new_phi_pred)
                        #     new_def_obj, new_def_coverage, _ = getDefUtility(single_data, new_unbiased_probs_pred, learning_model, omega=omega, verbose=False, training_mode=False,  training_method=training_method) # most time-consuming part
                        #     estimated_dopt_dphi[:,i] = (new_def_coverage - grad_def_coverage) / step_size

                        # print('dopt_dphi abs sum: {}, estimated abs sum: {}, difference sum: {}, difference number: {}'.format(
                        #     torch.sum(torch.abs(dopt_dphi)),
                        #     torch.sum(torch.abs(estimated_dopt_dphi)),
                        #     torch.sum(torch.abs(dopt_dphi - estimated_dopt_dphi)),
                        #     torch.sum(torch.abs(torch.sign(dopt_dphi) - torch.sign(estimated_dopt_dphi))/2)))
                        #     # torch.max(dopt_dphi - estimated_dopt_dphi)))
                        # print('difference:', dopt_dphi - estimated_dopt_dphi)
                        # ==========================================================

                U = torch.Tensor(G.graph['U'])
                initial_distribution = torch.Tensor(G.graph['initial_distribution'])
                new_def_obj  = -(objective_function_matrix_form(def_coverage, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))
                def_obj_list.append(new_def_obj.item())
                simulated_def_obj_list.append(simulated_def_obj)

                loss_list.append(loss.item())

                if (iter_n%batch_size == (batch_size-1)) and (epoch > 0) and (mode == "training"):
                    optimizer.zero_grad()
                    try:
                        if training_method == "two-stage":
                            loss[0].backward()
                        elif training_method == "decision-focused" or training_method == "block-decision-focused" or training_method == 'corrected-block-decision-focused':
                            (-def_obj).backward()
                        elif training_method == "hybrid":
                            loss[0].backward(retain_graph=True)
                            ts_grad = [parameter.grad.clone() for parameter in net2.parameters()]
                            optimizer.zero_grad()
                            (-def_obj).backward()
                            df_grad = [parameter.grad for parameter in net2.parameters()]
                            cos = nn.CosineSimilarity(dim=0)
                            for (ts_grad_i, df_grad_i) in zip(ts_grad, df_grad):
                                cosine_similarity = cos(ts_grad_i.reshape(-1), df_grad_i.reshape(-1))
                                df_grad_i = df_grad_i + ts_grad_i * max(0, cosine_similarity)
                        else:
                            raise TypeError("Not Implemented Method")
                        torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=max_norm) # gradient clipping
                        # print(torch.norm(net2.gcn1.weight.grad))
                        # print(torch.norm(net2.gcn2.weight.grad))
                        # print(torch.norm(net2.fc1.weight.grad))
                        optimizer.step()
                    except:
                        print("no grad is backpropagated...")

            if (epoch > 0) and (mode == "validating"):
                if training_method == "two-stage":
                    scheduler.step(np.mean(loss_list))
                elif training_method == "decision-focused" or training_method == "block-decision-focused" or training_method == 'corrected-block-decision-focused' or training_method == 'hybrid':
                    scheduler.step(-np.mean(def_obj_list))
                else:
                    raise TypeError("Not Implemented Method")

            # Storing loss and defender utility
            if epoch >= 0:
                epoch_loss_list.append(np.mean(loss_list))
                epoch_def_list.append(np.mean(def_obj_list))

            ################################### Print stuff after every epoch 
            np.random.shuffle(dataset)
            print("Mode: {}/ Epoch number: {}/ Loss: {}/ DefU: {}/ Simulated DefU: {}".format(
                  mode, epoch, np.mean(loss_list), np.mean(def_obj_list), np.mean(simulated_def_obj_list)))

            f_save.write("{}, {}, {}, {}, {}\n".format(mode, epoch, np.mean(loss_list), np.mean(def_obj_list), np.mean(simulated_def_obj_list)))
        
        time4 = time.time()
        cprint (("TIME FOR THIS EPOCH:", time4-time3),'red')
        time3 = time4

    average_nodes = np.mean([x[0].number_of_nodes() for x in train_data] + [x[0].number_of_nodes() for x in validate_data] + [x[0].number_of_nodes() for x in test_data])
    average_edges = np.mean([x[0].number_of_edges() for x in train_data] + [x[0].number_of_edges() for x in validate_data] + [x[0].number_of_edges() for x in test_data])

    # ============== using results on validation set to choose solution ===============
    # if training_method == "two-stage":
    #     max_epoch = np.argmax(validating_loss_list[-validation_window*2:-validation_window]) # choosing based on loss
    #     final_loss = testing_loss_list[-validation_window*2 + max_epoch]
    #     final_defender_utility = testing_defender_utility_list[-validation_window*2 + max_epoch]
    # else:
    #     max_epoch = np.argmax(validating_defender_utility_list[-validation_window*2:-validation_window]) # choosing based on defender utility
    #     final_loss = testing_loss_list[-validation_window*2 + max_epoch]
    #     final_defender_utility = testing_defender_utility_list[-validation_window*2 + max_epoch]

    f_time.write("nodes, {}, edges, {}, epochs, {}, time, {}\n".format(average_nodes, average_edges, epoch, time.time() - beginning_time))
            
    return net2 ,training_loss_list, testing_loss_list, training_defender_utility_list, testing_defender_utility_list
    

def getDefUtility(single_data, unbiased_probs_pred, path_model, omega=4, verbose=False, initial_coverage_prob=None, training_mode=True, training_method='two-stage'):
    G, Fv, coverage_prob, phi_true, path_list, min_cut, log_prob, unbiased_probs_true = single_data
    
    n, m = G.number_of_nodes(), G.number_of_edges()
    budget = G.graph['budget']
    U = torch.Tensor(G.graph['U'])
    initial_distribution = torch.Tensor(G.graph['initial_distribution'])
    options = {"maxiter": 100, "disp": verbose}
    tol = None
    method = "SLSQP"

    edges = G.edges()
    edge2index = {}
    for idx, edge in enumerate(edges):
        edge2index[edge] = idx
        edge2index[(edge[1], edge[0])] = idx

    # full forward path, the decision variables are the entire set of variables
    # initial_coverage_prob = np.zeros(m)
    initial_coverage_prob = np.random.rand(m) # somehow this is very influential...
    # initial_coverage_prob = np.ones(m) # somehow this is very influential...
    initial_coverage_prob = initial_coverage_prob / np.sum(initial_coverage_prob) * budget

    pred_optimal_res = get_optimal_coverage_prob(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options, method=method, initial_coverage_prob=initial_coverage_prob, tol=tol) # scipy version
    pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
    if not pred_optimal_res['success']:
        print(pred_optimal_res)
        print('optimization fails...')
    # pred_optimal_coverage = torch.Tensor(get_optimal_coverage_prob_frank_wolfe(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, num_iterations=100, initial_coverage_prob=initial_coverage_prob, tol=tol)) # Frank Wolfe version

    # ======================== edge set choice =====================
    # sample_distribution = pred_optimal_coverage.detach().numpy() + 0.1
    # sample_distribution /= sum(sample_distribution)
    sample_distribution = np.ones(m) / m
    if training_method == 'block-decision-focused' or training_method == 'hybrid' or training_method == 'corrected-block-decision-focused':
        cut_size = n // 2 # heuristic
        while True:
            edge_set = np.array(sorted(np.random.choice(range(m), size=cut_size, replace=False, p=sample_distribution)))
            if sum(pred_optimal_coverage[edge_set]) > 0:
                break
    else:
        cut_size = m
        edge_set = list(range(m))
    # ========================== QP part ===========================

    # A_matrix, b_matrix = torch.Tensor(), torch.Tensor()
    # G_matrix = torch.cat((-torch.eye(cut_size), torch.eye(cut_size), torch.ones(1, cut_size)))
    # h_matrix = torch.cat((torch.zeros(cut_size), torch.ones(cut_size), torch.Tensor([sum(pred_optimal_coverage[edge_set])])))
    A_matrix, b_matrix = torch.ones(1, cut_size), torch.Tensor([sum(pred_optimal_coverage[edge_set])])
    G_matrix = torch.cat((-torch.eye(cut_size), torch.eye(cut_size)))
    h_matrix = torch.cat((torch.zeros(cut_size), torch.ones(cut_size)))

    if training_mode and pred_optimal_res['success']: # and sum(pred_optimal_coverage[edge_set]) > 0.1:
        
        solver_option = 'default'
        # I seriously don't know wherether to use 'default' or 'gurobi' now...
        # Gurobi performs well when there is no noise but default performs well when there is noise
        # But theoretically they should perform roughly the same...

        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega)
        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega, lib=torch)

        Q_sym = (Q + Q.t()) / 2
    
        eigenvalues, eigenvectors = np.linalg.eig(Q_sym)
        eigenvalues = [x.real for x in eigenvalues]
        reg_const = max(0, -min(eigenvalues) + 1)
        # Q_regularized = torch.eye(len(edge_set)) * 1
        Q_regularized = Q_sym + torch.eye(len(edge_set)) * reg_const
        # Q_regularized = (Q_sym + torch.eye(len(edge_set)) * max(0, -min(eigenvalues) + reg_const))
        # new_eigenvalues, new_eigenvectors = np.linalg.eig(Q_regularized)
        
        p = jac.view(1, -1) - pred_optimal_coverage[edge_set] @ Q_regularized
  
        qp_solver = qpth.qp.QPFunction()
        coverage_qp_solution = qp_solver(Q_regularized, p, G_matrix, h_matrix, A_matrix, b_matrix)[0]       # Default version takes 1/2 x^T Q x + x^T p; not 1/2 x^T Q x + x^T p

        full_coverage_qp_solution = pred_optimal_coverage.clone()
        full_coverage_qp_solution[edge_set] = coverage_qp_solution
        old_pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))

        if training_method == 'corrected-block-decision-focused':
            # computing the correction terms
            indices = np.arange(cut_size)
            np.random.shuffle(indices)
            indices1, indices2 = np.array_split(indices, 2)
            edge_set1, edge_set2 = edge_set[indices1], edge_set[indices2]
            Q1, Q2 = Q_regularized[indices1][:,indices1], Q_regularized[indices2][:,indices2]
            p1 = jac[indices1].view(1,-1) - pred_optimal_coverage[edge_set1] @ Q1
            p2 = jac[indices2].view(1,-1) - pred_optimal_coverage[edge_set2] @ Q2
            cut_size1, cut_size2 = len(indices1), len(indices2)

            # Q1 part
            A_matrix1, b_matrix1 = torch.ones(1, cut_size1), torch.Tensor([sum(pred_optimal_coverage[edge_set1])])
            G_matrix1 = torch.cat((-torch.eye(cut_size1), torch.eye(cut_size1)))
            h_matrix1 = torch.cat((torch.zeros(cut_size1), torch.ones(cut_size1)))
            qp_solver1 = qpth.qp.QPFunction()
            coverage_qp_solution1 = qp_solver(Q1, p1, G_matrix1, h_matrix1, A_matrix1, b_matrix1)[0]       # Default version takes 1/2 x^T Q x + x^T p; not 1/2 x^T Q x + x^T p
            full_coverage_qp_solution1 = pred_optimal_coverage.clone()
            full_coverage_qp_solution1[edge_set1] = coverage_qp_solution1

            pred_defender_utility1  = -(objective_function_matrix_form(full_coverage_qp_solution1, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))

            # Q2 part
            A_matrix2, b_matrix2 = torch.ones(1, cut_size2), torch.Tensor([sum(pred_optimal_coverage[edge_set2])])
            G_matrix2 = torch.cat((-torch.eye(cut_size2), torch.eye(cut_size2)))
            h_matrix2 = torch.cat((torch.zeros(cut_size2), torch.ones(cut_size2)))
            qp_solver2 = qpth.qp.QPFunction()
            coverage_qp_solution2 = qp_solver(Q2, p2, G_matrix2, h_matrix2, A_matrix2, b_matrix2)[0]       # Default version takes 1/2 x^T Q x + x^T p; not 1/2 x^T Q x + x^T p
            full_coverage_qp_solution2 = pred_optimal_coverage.clone()
            full_coverage_qp_solution2[edge_set2] = coverage_qp_solution2

            pred_defender_utility2  = -(objective_function_matrix_form(full_coverage_qp_solution2, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))

            # correction ratio
            c = (m - cut_size) / ((m - cut_size/2))
            pred_defender_utility = old_pred_defender_utility / (1-c) - c/(1-c) * (pred_defender_utility1 + pred_defender_utility2)
        else:
            pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))


    else:
        full_coverage_qp_solution = pred_optimal_coverage.clone()
        pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))

    # ========================= Error message =========================
    if (torch.norm(pred_optimal_coverage - full_coverage_qp_solution) > 0.01): # or 0.01 for GUROBI, 0.1 for qpth
        print('QP solution and scipy solution differ {} too much..., not backpropagating this instance'.format(torch.norm(pred_optimal_coverage - full_coverage_qp_solution)))
        print("objective value (SLSQP): {}".format(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
        print(pred_optimal_coverage)
        print("objective value (QP): {}".format(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
        print(full_coverage_qp_solution)
        full_coverage_qp_solution = pred_optimal_coverage.clone()

    # ==================== Actual Defender Utility ====================
    # Running simulations to check the actual defender utility
    # _, simulated_defender_utility = attackerOracle(G, full_optimal_coverage, phi_true, omega=omega, num_paths=100)
    simulated_defender_utility = 0

    return pred_defender_utility, full_coverage_qp_solution, simulated_defender_utility

if __name__=='__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--filename', type=str, help='filename under folder results')

    parser.add_argument('--fixed-graph', type=int, default=0, help='0 -> randomly generated, 1 -> first fixed graph, 2 -> second fixed graph')
    parser.add_argument('--prob', type=float, default=0.2, help='input the probability used as input of random graph generator')

    parser.add_argument('--learning-rate', type=float, default=0.005, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--feature-size', type=int, default=5, help='feature size of each node')
    parser.add_argument('--noise', type=float, default=0, help='noise level of the normalized features (in variance)')
    parser.add_argument('--omega', type=float, default=4, help='risk aversion of the attacker')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')

    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')
    parser.add_argument('--number-graphs', type=int, default=1, help='number of different graphs in the dataset')
    parser.add_argument('--number-samples', type=int, default=100, help='number of samples per graph')
    parser.add_argument('--number-sources', type=int, default=2, help='number of randomly generated sources')
    parser.add_argument('--number-targets', type=int, default=2, help='number of randomly generated targets')

    parser.add_argument('--distribution', type=int, default=1, help='0 -> random walk distribution, 1 -> empirical distribution')
    parser.add_argument('--method', type=int, default=0, help='0 -> two-stage, 1 -> decision-focused, 2 -> block df, 3 -> corrected df, 4 -> hybrid')
    
    args = parser.parse_args()

    ############################# Parameters and settings:
    time1 =time.time()
    
    learning_mode = args.distribution
    learning_model_type = 'random_walk_distribution' if learning_mode == 0 else 'empirical_distribution'

    training_mode = args.method
    method_dict = {0: 'two-stage', 1: 'decision-focused', 2: 'block-decision-focused', 3: 'corrected-block-decision-focused', 4: 'hybrid'} # 3 is reserved for new-block-df
    training_method = method_dict[training_mode]

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
    NOISE_LEVEL = args.noise
    if SEED == 0:
        SEED = np.random.randint(1, 100000)

    ###############################
    filename = args.filename
    if FIXED_GRAPH == 0:
        filepath_data    =      "results/random/{}_{}_n{}_p{}_b{}_noise{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
        filepath_figure  =      "figures/random/{}_{}_n{}_p{}_b{}_noise{}.png".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
        filepath_time    = "results/time/random/{}_{}_n{}_p{}_b{}_noise{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
        filepath_summary =     "results/summary/{}_{}_n{}_p{}_b{}_noise{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    else:
        filepath_data    = "results/fixed/{}_{}_test.csv"               .format(filename, training_method)
        filepath_figure  = "figures/fixed/{}_{}_test.png"               .format(filename, training_method)
        filepath_time    = "results/time/fixed/{}_{}_b{}.csv"           .format(filename, training_method, DEFENDER_BUDGET)
        filepath_summary = "results/summary/fixed/{}_{}_n{}_p{}_b{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET)

    f_save = open(filepath_data, 'a')
    f_time = open(filepath_time, 'a')
    f_summary = open(filepath_summary, 'a')

    f_save.write('Random seed, {}\n'.format(SEED))
    f_time.write('Random seed, {}\n'.format(SEED))
    f_summary.write('Random seed, {}\n'.format(SEED))
      
    ############################### Data genaration:
    train_data, validate_data, test_data = generateSyntheticData(feature_size, path_type=learning_model_type,
            n_graphs=NUMBER_OF_GRAPHS, samples_per_graph=SAMPLES_PER_GRAPH, empirical_samples_per_instance=EMPIRICAL_SAMPLES_PER_INSTANCE,
            fixed_graph=FIXED_GRAPH, omega=OMEGA,
            N_low=GRAPH_N_LOW, N_high=GRAPH_N_HIGH, e_low=GRAPH_E_PROB_LOW, e_high=GRAPH_E_PROB_HIGH,
            budget=DEFENDER_BUDGET, n_sources=NUMBER_OF_SOURCES, n_targets=NUMBER_OF_TARGETS,
            random_seed=SEED, noise_level=NOISE_LEVEL)
    
    time2 = time.time()
    cprint (("DATA GENERATION: ", time2-time1), 'red')

    np.random.shuffle(train_data)

    print("Training method: {}".format(training_method))
    print('Noise level: {}'.format(NOISE_LEVEL))
    print('Node size: {}, p={}, budget: {}'.format(GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET))
    print('Sample graph size: {}, sample size: {}'.format(NUMBER_OF_GRAPHS, SAMPLES_PER_GRAPH))
    print('omega: {}'.format(OMEGA))
    print("Data length train/test:", len(train_data), len(test_data))

    ############################## Training the ML models:    
    time3=time.time()
    # Learn the neural networks:
    net2, train_loss, test_loss, training_graph_def_u, testing_graph_def_u=learnEdgeProbs_simple(
                                train_data, validate_data, test_data, f_save, f_time, f_summary,
                                learning_model=learning_model_type,
                                lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE, 
                                optimizer=OPTIMIZER, omega=OMEGA, training_method=training_method)

    time4=time.time()
    cprint (("TOTAL TRAINING+TESTING TIME: ", time4-time3), 'red')

    f_save.close()
    f_time.close()
    f_summary.close()

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
        
