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

import qpthnew

def learnEdgeProbs_simple(train_data, validate_data, test_data, f_save, f_time, lr=0.1, learning_model='random_walk_distribution'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage', max_norm=0.1, block_cut_size=0.5):
    
    net2= GCNPredictionNet2(feature_size)
    net2.train()
    if optimizer=='adam':
        optimizer=optim.Adam(net2.parameters(), lr=lr)
    elif optimizer=='sgd':
        optimizer=optim.SGD(net2.parameters(), lr=lr)
    elif optimizer=='adamax':
        optimizer=optim.Adamax(net2.parameters(), lr=lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)
   
    training_loss_list, validating_loss_list, testing_loss_list = [], [], []
    training_defender_utility_list, validating_defender_utility_list, testing_defender_utility_list = [], [], []
    
    print ("Training...")
    training_time = 0

    f_save.write("mode, epoch, average loss, defender utility, simulated defender utility\n")

    pretrain_epochs = 0
    decay_rate = 0.95
    for epoch in range(-1, n_epochs):
        epoch_training_time = 0
        if epoch <= pretrain_epochs:
            ts_weight = 1
        else:
            ts_weight = decay_rate ** (epoch - pretrain_epochs)
        df_weight = 1 - ts_weight
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
                time1 = time.time()
                random_value = np.random.random()
                ################################### Gather data based on learning model
                G, Fv, coverage_prob, phi_true, path_list, cut, log_prob, unbiased_probs_true, previous_gradient = dataset[iter_n]
                n, m = G.number_of_nodes(), G.number_of_edges()
                
                ################################### Compute edge probabilities
                Fv_torch   = torch.as_tensor(Fv, dtype=torch.float)
                edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
                phi_pred   = net2(Fv_torch, edge_index).view(-1) if epoch >= 0 else phi_true # when epoch < 0, testing the optimal loss and defender utility
                # phi_pred.require_grad = True

                unbiased_probs_pred = phi2prob(G, phi_pred) if epoch >= 0 else unbiased_probs_true
                biased_probs_pred = prob2unbiased(G, -coverage_prob,  unbiased_probs_pred, omega=omega) # feeding negative coverage to be biased
                # biased_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
                
                ################################### Compute loss
                # loss = torch.norm((unbiased_probs_true - unbiased_probs_pred)) # * torch.Tensor(nx.adjacency_matrix(G).toarray()))
                log_prob_pred = torch.zeros(1)
                for path in path_list:
                    for e in path: 
                        log_prob_pred -= torch.log(biased_probs_pred[e[0]][e[1]])
                log_prob_pred /= len(path_list)
                loss = (log_prob_pred - log_prob)[0]

                # COMPUTE DEFENDER UTILITY 
                single_data = dataset[iter_n]
                epoch_training_time += time.time() - time1

                if mode == 'testing' or mode == "validating" or epoch <= 0: # or training_method == "two-stage" or epoch <= 0:
                    cut_size = m
                    def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=False, training_method=training_method) # feed forward only
                else:
                    if training_method == "two-stage" or epoch <= pretrain_epochs:
                        cut_size = m
                        def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=False, training_method=training_method) # most time-consuming part
                        # ignore the time of computing defender utility
                    else:
                        time2 = time.time() # including the time of computing defender utility
                        if training_method == 'decision-focused':
                            cut_size = m
                        elif training_method == 'block-decision-focused' or training_method == 'hybrid' or training_method == 'corrected-block-decision-focused':
                            if type(block_cut_size) == str and block_cut_size[-1] == 'n':
                                cut_size = int(n * float(block_cut_size[:-1]))
                            elif block_cut_size <= 1:
                                cut_size = int(m * block_cut_size)
                            else:
                                cut_size = block_cut_size
                        else:
                            raise TypeError('Not defined method')

                        def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=True,  training_method=training_method) # most time-consuming part
                        epoch_training_time += time.time() - time2
                        
                        # =============== checking gradient manually ===============
                        # dopt_dphi = torch.Tensor(len(def_coverage), len(phi_pred))
                        # for i in range(len(def_coverage)):
                        #     grad_def_obj, grad_def_coverage, _ = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=True,  training_method=training_method) # most time-consuming part
                        #     dopt_dphi[i] = torch.autograd.grad(grad_def_coverage[i], phi_pred, retain_graph=True)[0] # ith dimension
                        #     step_size = 0.01

                        # estimated_dopt_dphi = torch.Tensor(len(def_coverage), len(phi_pred))
                        # for i in range(len(phi_pred)):
                        #     new_phi_pred = phi_pred.clone()
                        #     new_phi_pred[i] += step_size
        
                        #     # validating
                        #     new_unbiased_probs_pred = phi2prob(G, new_phi_pred)
                        #     _, new_def_coverage, _ = getDefUtility(single_data, new_unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=False,  training_method=training_method) # most time-consuming part
                        #     estimated_dopt_dphi[:,i] = (new_def_coverage - grad_def_coverage) / step_size

                        # print('dopt_dphi abs sum: {}, estimated abs sum: {}, difference sum: {}, difference number: {}'.format(
                        #     torch.sum(torch.abs(dopt_dphi)),
                        #     torch.sum(torch.abs(estimated_dopt_dphi)),
                        #     torch.sum(torch.abs(dopt_dphi - estimated_dopt_dphi)),
                        #     torch.sum(torch.abs(torch.sign(dopt_dphi) - torch.sign(estimated_dopt_dphi))/2)))
                        #     # torch.max(dopt_dphi - estimated_dopt_dphi)))
                        # cos = nn.CosineSimilarity(dim=0)
                        # print('cosine similarity:', cos(dopt_dphi.reshape(-1), estimated_dopt_dphi.reshape(-1)))
                        # ==========================================================

                U = torch.Tensor(G.graph['U'])
                initial_distribution = torch.Tensor(G.graph['initial_distribution'])
                new_def_obj  = -(objective_function_matrix_form(def_coverage, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))
                def_obj_list.append(new_def_obj.item())
                simulated_def_obj_list.append(simulated_def_obj)

                loss_list.append(loss.item())

                if (iter_n%batch_size == (batch_size-1)) and (epoch > 0) and (mode == "training"):
                    time3 = time.time()
                    optimizer.zero_grad()
                    try:
                        if training_method == "two-stage" or epoch <= pretrain_epochs:
                            loss.backward()
                        elif training_method == "decision-focused" or training_method == "block-decision-focused" or training_method == 'corrected-block-decision-focused':
                            # (-def_obj).backward()
                            (-def_obj * m / cut_size).backward()
                        elif training_method == "hybrid":
                            # ((-def_obj) * df_weight + loss[0] * ts_weight).backward()
                            ((-def_obj * m / cut_size) * df_weight + loss * ts_weight).backward()

                            # loss[0].backward(retain_graph=True)
                            # rs_grad = [parameter.grad.clone() for parameter in net2.parameters()]
                            # optimizer.zero_grad()
                            # (-def_obj * m / cut_size).backward()
                            # df_grad = [parameter.grad for parameter in net2.parameters()]
                            # cos = nn.CosineSimilarity(dim=0)
                            # for (ts_grad_i, df_grad_i) in zip(ts_grad, df_grad):
                            #     cosine_similarity = cos(ts_grad_i.reshape(-1), df_grad_i.reshape(-1))
                            #     df_grad_i = df_grad_i + ts_grad_i * max(0, cosine_similarity) # cosine similarity method
                        else:
                            raise TypeError("Not Implemented Method")
                        torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=max_norm) # gradient clipping
                        # print(torch.norm(net2.gcn1.weight.grad))
                        # print(torch.norm(net2.gcn2.weight.grad))
                        # print(torch.norm(net2.fc1.weight.grad))
                        optimizer.step()
                    except:
                        print("no grad is backpropagated...")
                    epoch_training_time += time.time() - time3

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
        print('Training time for this epoch: {}'.format(epoch_training_time))
        training_time += epoch_training_time
        

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

    print('Total training time: {}'.format(training_time))
    f_time.write("nodes, {}, edges, {}, epochs, {}, time, {}\n".format(average_nodes, average_edges, epoch, training_time))
            
    return net2 ,training_loss_list, testing_loss_list, training_defender_utility_list, testing_defender_utility_list
    

def getDefUtility(single_data, unbiased_probs_pred, path_model, cut_size, omega=4, verbose=False, initial_coverage_prob=None, training_mode=True, training_method='two-stage'):
    G, Fv, coverage_prob, phi_true, path_list, min_cut, log_prob, unbiased_probs_true, previous_gradient = single_data
    
    n, m = G.number_of_nodes(), G.number_of_edges()
    budget = G.graph['budget']
    U = torch.Tensor(G.graph['U'])
    initial_distribution = torch.Tensor(G.graph['initial_distribution'])
    options = {"maxiter": 100, "disp": verbose}
    tol = None
    method = "SLSQP"

    edges = G.edges()

    # full forward path, the decision variables are the entire set of variables
    # initial_coverage_prob = np.zeros(m)
    # initial_coverage_prob = np.random.rand(m) # somehow this is very influential...
    initial_coverage_prob = np.ones(m) # somehow this is very influential...
    initial_coverage_prob = initial_coverage_prob / np.sum(initial_coverage_prob) * budget

    pred_optimal_res = get_optimal_coverage_prob(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options, method=method, initial_coverage_prob=initial_coverage_prob, tol=tol) # scipy version
    pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
    if not pred_optimal_res['success']:
        print(pred_optimal_res)
        print('optimization fails...')
    # pred_optimal_coverage = torch.Tensor(get_optimal_coverage_prob_frank_wolfe(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, num_iterations=100, initial_coverage_prob=initial_coverage_prob, tol=tol)) # Frank Wolfe version

    # ======================== edge set choice =====================
    first_order_derivative = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, np.arange(m), omega=omega, lib=torch)
    sample_distribution = np.abs(first_order_derivative.detach().numpy()) + 1e-3
    # sample_distribution = pred_optimal_coverage.detach().numpy() + 1e-3
    # sample_distribution = np.ones(m)
    sample_distribution /= sum(sample_distribution)
    if training_method == 'block-decision-focused' or training_method == 'hybrid':
        min_sum = 1e-2
        while True:
            edge_set = np.array(sorted(np.random.choice(range(m), size=cut_size, replace=False, p=sample_distribution)))
            if sum(pred_optimal_coverage[edge_set]) > min_sum:
                break
    elif training_method == 'corrected-block-decision-focused':
        min_sum = 1e-2
        while True:
            edge_set = np.array(sorted(np.random.choice(range(m), size=cut_size, replace=False, p=sample_distribution)))
            indices = np.arange(cut_size)
            np.random.shuffle(indices)
            indices1, indices2 = np.array_split(indices, 2)
            indices1, indices2 = sorted(indices1), sorted(indices2)
            edge_set1, edge_set2 = edge_set[indices1], edge_set[indices2]
            if sum(pred_optimal_coverage[edge_set1]) > min_sum / 10 and sum(pred_optimal_coverage[edge_set2]) > min_sum / 10:
                break
    else:
        edge_set = list(range(m))
    off_edge_set = sorted(list(set(range(m)) - set(edge_set)))
    # ========================== QP part ===========================

    # A_matrix, b_matrix = torch.Tensor(), torch.Tensor()
    # G_matrix = torch.cat((-torch.eye(cut_size), torch.eye(cut_size), torch.ones(1, cut_size)))
    # h_matrix = torch.cat((torch.zeros(cut_size), torch.ones(cut_size), torch.Tensor([sum(pred_optimal_coverage[edge_set])])))
    scale_constant = 1 # cut_size
    A_matrix, b_matrix = torch.ones(1, cut_size)/scale_constant, torch.Tensor([sum(pred_optimal_coverage[edge_set])])/scale_constant
    G_matrix = torch.cat((-torch.eye(cut_size), torch.eye(cut_size)))
    h_matrix = torch.cat((torch.zeros(cut_size), torch.ones(cut_size)))

    if training_mode and pred_optimal_res['success']: # and sum(pred_optimal_coverage[edge_set]) > 0.1:
        
        solver_option = 'default'
        # I seriously don't know wherether to use 'default' or 'gurobi' now...
        # Gurobi performs well when there is no noise but default performs well when there is noise
        # But theoretically they should perform roughly the same...

        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega)
        # Q = Q_full[:,edge_set]
        # Q_bar = Q_full[:,off_edge_set]

        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega, lib=torch)

        Q_sym = (Q + Q.t()) / 2
    
        # ------------------ regularization -----------------------
        Q_regularized = Q_sym.clone()
        reg_const = 0.1
        while True:
            # ------------------ eigen regularization -----------------------
            # Q_regularized = Q_sym + torch.eye(len(edge_set)) * max(0, -min(eigenvalues) + reg_const)
            # ----------------- diagonal regularization ---------------------
            Q_regularized[range(cut_size), range(cut_size)] = torch.clamp(torch.diag(Q_sym), min=reg_const)
            try:
                L = torch.cholesky(Q_regularized)
                break
            except:
                reg_const *= 2

        p = jac.view(1, -1) - Q_regularized @ pred_optimal_coverage[edge_set]
 
        try:
            # qp_solver = qpthnew.qp.QPFunction(correction_term, previous_gradient)
            qp_solver = qpth.qp.QPFunction()
            coverage_qp_solution = qp_solver(Q_regularized, p, G_matrix, h_matrix, A_matrix, b_matrix)[0]       # Default version takes 1/2 x^T Q x + x^T p; not 1/2 x^T Q x + x^T p
            full_coverage_qp_solution = pred_optimal_coverage.clone()
            full_coverage_qp_solution[edge_set] = coverage_qp_solution
        except:
            print("QP solver fails... Usually because Q is not PSD")
            full_coverage_qp_solution = pred_optimal_coverage.clone()

        if training_method == 'corrected-block-decision-focused':
            # computing the correction terms
            Q1, Q2 = Q_regularized[indices1][:,indices1], Q_regularized[indices2][:,indices2]
            p1 = jac[indices1].view(1,-1) - pred_optimal_coverage[edge_set1] @ Q1
            p2 = jac[indices2].view(1,-1) - pred_optimal_coverage[edge_set2] @ Q2
            cut_size1, cut_size2 = len(indices1), len(indices2)

            # Q1 part
            A_matrix1, b_matrix1 = torch.ones(1, cut_size1), torch.Tensor([sum(pred_optimal_coverage[edge_set1])])
            G_matrix1 = torch.cat((-torch.eye(cut_size1), torch.eye(cut_size1)))
            h_matrix1 = torch.cat((torch.zeros(cut_size1), torch.ones(cut_size1)))
            qp_solver1 = qpth.qp.QPFunction()
            try:
                coverage_qp_solution1 = qp_solver1(Q1, p1, G_matrix1, h_matrix1, A_matrix1, b_matrix1)[0]
                full_coverage_qp_solution1 = pred_optimal_coverage.clone()
                full_coverage_qp_solution1[edge_set1] = coverage_qp_solution1
            except:
                print("QP solver 1 fails... Usually because Q is not PSD")
                full_coverage_qp_solution1 = pred_optimal_coverage.clone()

            pred_defender_utility1  = -(objective_function_matrix_form(full_coverage_qp_solution1, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))
            if (torch.norm(pred_optimal_coverage - full_coverage_qp_solution1) > 0.01):
                pred_defender_utility1  = -(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))

            # Q2 part
            A_matrix2, b_matrix2 = torch.ones(1, cut_size2), torch.Tensor([sum(pred_optimal_coverage[edge_set2])])
            G_matrix2 = torch.cat((-torch.eye(cut_size2), torch.eye(cut_size2)))
            h_matrix2 = torch.cat((torch.zeros(cut_size2), torch.ones(cut_size2)))
            try:
                qp_solver2 = qpth.qp.QPFunction()
                coverage_qp_solution2 = qp_solver2(Q2, p2, G_matrix2, h_matrix2, A_matrix2, b_matrix2)[0]
                full_coverage_qp_solution2 = pred_optimal_coverage.clone()
                full_coverage_qp_solution2[edge_set2] = coverage_qp_solution2
            except:
                print("QP solver 2 fails... Usually because Q is not PSD")
                full_coverage_qp_solution2 = pred_optimal_coverage.clone()

            old_pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))
            pred_defender_utility2  = -(objective_function_matrix_form(full_coverage_qp_solution2, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))
            if (torch.norm(pred_optimal_coverage - full_coverage_qp_solution2) > 0.01):
                pred_defender_utility2  = -(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), None, omega=omega))

            # correction ratio
            c = (m - cut_size) / ((m - cut_size/2))
            pred_defender_utility = old_pred_defender_utility / (1-c) - c/(1-c) * (pred_defender_utility1 + pred_defender_utility2)
        else:
            pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))


    else:
        full_coverage_qp_solution = pred_optimal_coverage.clone()
        pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))

    # ========================= Error message =========================
    if (torch.norm(pred_optimal_coverage - full_coverage_qp_solution) > 0.1): # or 0.01 for GUROBI, 0.1 for qpth
        print('QP solution and scipy solution differ {} too much..., not backpropagating this instance'.format(torch.norm(pred_optimal_coverage - full_coverage_qp_solution)))
        print("objective value (SLSQP): {}".format(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
        print(pred_optimal_coverage)
        print("objective value (QP): {}".format(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
        print(full_coverage_qp_solution)
        full_coverage_qp_solution = pred_optimal_coverage.clone()
        pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))

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
    parser.add_argument('--cut-size', type=str, default='n/2', help='number of the defender budget')

    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')
    parser.add_argument('--number-graphs', type=int, default=1, help='number of different graphs in the dataset')
    parser.add_argument('--number-samples', type=int, default=100, help='number of samples per graph')
    parser.add_argument('--number-sources', type=int, default=2, help='number of randomly generated sources')
    parser.add_argument('--number-targets', type=int, default=2, help='number of randomly generated targets')

    parser.add_argument('--distribution', type=int, default=1, help='0 -> random walk distribution, 1 -> empirical distribution')
    parser.add_argument('--method', type=int, default=0, help='0 -> two-stage, 1 -> decision-focused, 2 -> block df, 3 -> corrected df, 4 -> hybrid')
    
    args = parser.parse_args()

    ############################# Parameters and settings:
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
    CUT_SIZE = args.cut_size
    if CUT_SIZE[-1] != 'n':
        CUT_SIZE = float(CUT_SIZE)
    GRAPH_TYPE = "random_graph" if FIXED_GRAPH == 0 else "fixed_graph"
    SEED = args.seed
    NOISE_LEVEL = args.noise
    if SEED == 0:
        SEED = np.random.randint(1, 100000)

    ###############################
    filename = args.filename
    if FIXED_GRAPH == 0:
        filepath_data    =      "results/random/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        filepath_time    = "results/time/random/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
    else:
        filepath_data    = "results/fixed/{}_{}_test.csv"               .format(filename, training_method)
        filepath_time    = "results/time/fixed/{}_{}_b{}.csv"           .format(filename, training_method, DEFENDER_BUDGET)

    f_save = open(filepath_data, 'a')
    f_time = open(filepath_time, 'a')

    f_save.write('Random seed, {}\n'.format(SEED))
    f_time.write('Random seed, {}\n'.format(SEED))
      
    ############################### Data genaration:
    train_data, validate_data, test_data = generateSyntheticData(feature_size, path_type=learning_model_type,
            n_graphs=NUMBER_OF_GRAPHS, samples_per_graph=SAMPLES_PER_GRAPH, empirical_samples_per_instance=EMPIRICAL_SAMPLES_PER_INSTANCE,
            fixed_graph=FIXED_GRAPH, omega=OMEGA,
            N_low=GRAPH_N_LOW, N_high=GRAPH_N_HIGH, e_low=GRAPH_E_PROB_LOW, e_high=GRAPH_E_PROB_HIGH,
            budget=DEFENDER_BUDGET, n_sources=NUMBER_OF_SOURCES, n_targets=NUMBER_OF_TARGETS,
            random_seed=SEED, noise_level=NOISE_LEVEL)
    
    np.random.shuffle(train_data)

    print("Training method: {}".format(training_method))
    print('Noise level: {}'.format(NOISE_LEVEL))
    print('Node size: {}, p={}, budget: {}'.format(GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET))
    print('Block size: {}'.format(CUT_SIZE))
    print('Sample graph size: {}, sample size: {}'.format(NUMBER_OF_GRAPHS, SAMPLES_PER_GRAPH))
    print('omega: {}'.format(OMEGA))
    print("Data length train/test:", len(train_data), len(test_data))

    ############################## Training the ML models:    
    # Learn the neural networks:
    net2, train_loss, test_loss, training_graph_def_u, testing_graph_def_u=learnEdgeProbs_simple(
                                train_data, validate_data, test_data, f_save, f_time,
                                learning_model=learning_model_type,
                                lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE, 
                                optimizer=OPTIMIZER, omega=OMEGA, training_method=training_method, block_cut_size=CUT_SIZE)

    f_save.close()
    f_time.close()

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
                "Entire graph defender utility:":(testing_graph_def_u[0],testing_graph_def_u[-1]),
                "Test Loss:": test_loss} 
    
    cprint (all_params, 'green')
        
