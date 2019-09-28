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
from coverageProbability import *
# from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form
import qpthlocal

def mincut_coverage_to_full(mincut_coverage, cut, number_of_edges):
    full_coverage = np.zeros(number_of_edges)
    for idx, edge in enumerate(cut):
        full_coverage[edge] = mincut_coverage[idx].item()

    return full_coverage

def learnEdgeProbs_simple(train_data, validate_data, test_data, f_save, f_time, f_summary, lr=0.1, learning_model='random_walk_distribution'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage', max_norm=0.1, restrict_mincut=True):
    
    net2= GCNPredictionNet2(feature_size)
    net2.train()
    if optimizer=='adam':
        optimizer=optim.Adam(net2.parameters(), lr=lr)
    elif optimizer=='sgd':
        optimizer=optim.SGD(net2.parameters(), lr=lr)
    elif optimizer=='adamax':
        optimizer=optim.Adamax(net2.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, 'min')    
   
    training_loss_list, validating_loss_list, testing_loss_list = [], [], []
    training_defender_utility_list, validating_defender_utility_list, testing_defender_utility_list = [], [], []
    
    print ("Training...") 

    ######################################################
    #                   Pre-train Loop
    ######################################################
    # for epoch in range(0, 10):
    #     net2.train()
    #     dataset = train_data
    #     batch_loss = 0
    #     loss_list = []
    #     for iter_n in tqdm.trange(len(dataset)):
    #         ################################### Gather data based on learning model
    #         G, Fv, coverage_prob, phi_true, path_list, cut, log_prob, unbiased_probs_true = dataset[iter_n]
    #         
    #         ################################### Compute edge probabilities
    #         Fv_torch   = torch.as_tensor(Fv, dtype=torch.float)
    #         edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
    #         phi_pred   = net2(Fv_torch, edge_index).view(-1) if epoch >= 0 else phi_true # when epoch < 0, testing the optimal loss and defender utility

    #         unbiased_probs_pred = phi2prob(G, phi_pred)
    #         biased_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
    #         
    #         ################################### Compute loss
    #         log_prob_pred = torch.zeros(1)
    #         for path in path_list:
    #             for e in path: 
    #                 log_prob_pred -= torch.log(biased_probs_pred[e[0]][e[1]])
    #         log_prob_pred /= len(path_list)
    #         loss = log_prob_pred - log_prob

    #         single_data = dataset[iter_n]
    #         
    #         batch_loss += loss
    #         loss_list.append(loss.item())
    #         if torch.isnan(loss):
    #             print(phi_pred)
    #             print(unbiased_probs_pred)
    #             print(biased_probs_pred)
    #             raise ValueError('loss is nan!')

    #         if (iter_n%batch_size == (batch_size-1)):
    #             try:
    #                 optimizer.zero_grad()
    #                 batch_loss.backward()
    #                 # print(np.mean([parameter.grad.norm(2).item() for parameter in net2.parameters()]))
    #                 optimizer.step()
    #             except:
    #                 print("no grad is backpropagated...")
    #             batch_loss = 0

    #     print("Mode: {}/ loss: {}".format("Initialization", np.mean(loss_list)))

    ######################################################
    #                   TRAINING LOOP
    ######################################################
    beginning_time = time.time()
    time3 = time.time()

    f_save.write("mode, epoch, average loss, defender utility, simulated defender utility, fast defender utility\n")

    pretrain_epochs = 0
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

            loss_list, def_obj_list, simulated_def_obj_list, fast_obj_list = [], [], [], [] # fast obj list only used in testing time
            batch_loss = 0
            for iter_n in tqdm.trange(len(dataset)):
                ################################### Gather data based on learning model
                start_time = time.time()
                G, Fv, coverage_prob, phi_true, path_list, cut, log_prob, unbiased_probs_true = dataset[iter_n]
                
                ################################### Compute edge probabilities
                Fv_torch   = torch.as_tensor(Fv, dtype=torch.float)
                edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
                phi_pred   = net2(Fv_torch, edge_index).view(-1) if epoch >= 0 else phi_true # when epoch < 0, testing the optimal loss and defender utility
                # print(phi_pred)
                # print(phi_true)

                unbiased_probs_pred = phi2prob(G, phi_pred)
                biased_probs_pred = generate_EdgeProbs_from_Attractiveness(G, coverage_prob,  phi_pred, omega=omega)
                
                ################################### Compute loss
                log_prob_pred = torch.zeros(1)
                for path in path_list:
                    for e in path: 
                        log_prob_pred -= torch.log(biased_probs_pred[e[0]][e[1]])
                log_prob_pred /= len(path_list)
                loss = log_prob_pred - log_prob

                # print('running time for prediction:', time.time() - start_time)
                start_time = time.time()

                # COMPUTE DEFENDER UTILITY 
                single_data = dataset[iter_n]

                start_iteration = - 1 # -1: disable
                if mode == 'testing' or mode == "validating":
                    if epoch < start_iteration:
                        def_obj, simulated_def_obj, fast_def_obj = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                    else:
                        def_obj, def_coverage, simulated_def_obj                = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=False, verbose=False, training_mode=False)
                        fast_def_obj, fast_def_coverage, fast_simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=True,  verbose=False, training_mode=False)
                        fast_def_obj = fast_def_obj.item()

                else:
                    if training_method == 'decision-focused' and epoch > pretrain_epochs:
                        def_obj, def_coverage, simulated_def_obj                = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=restrict_mincut, verbose=False, training_mode=True, approximate=False) # most time-consuming part
                        fast_def_obj, fast_def_coverage, fast_simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=True,  verbose=False, training_mode=False)
                    else:
                        def_obj, simulated_def_obj = torch.zeros(1), torch.zeros(1)
                        fast_def_obj, fast_def_coverage, fast_simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=True,  verbose=False, training_mode=False)
                        # fast_def_obj = 0

                def_obj_list.append(def_obj.item())
                simulated_def_obj_list.append(simulated_def_obj)
                fast_obj_list.append(fast_def_obj)

                loss_list.append(loss.item())

                # backpropagate using loss when training two-stage and using -defender utility when training end-to-end
                if training_method == "two-stage" or epoch <= pretrain_epochs:
                    batch_loss += loss
                    if torch.isnan(loss):
                        print(phi_pred)
                        print(unbiased_probs_pred)
                        print(biased_probs_pred)
                        raise ValueError('loss is nan!')
                        # print(phi_pred)
                        # print(log_prob_pred)
                        # print(biased_probs_pred)
                elif training_method == "decision-focused":
                    batch_loss += (-def_obj)
                else:
                    raise TypeError("Not Implemented Method")

                # print(batch_loss)
                if (iter_n%batch_size == (batch_size-1)) and (epoch > 0) and (mode == "training"):
                    optimizer.zero_grad()
                    try:
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=max_norm) # gradient clipping
                        # print(torch.norm(net2.gcn1.weight.grad))
                        # print(torch.norm(net2.gcn2.weight.grad))
                        # print(torch.norm(net2.fc1.weight.grad))
                        optimizer.step()
                    except:
                        print("no grad is backpropagated...")
                    batch_loss = 0

                # print('running time for optimization:', time.time() - start_time)

            if (epoch > 0) and (mode == "validating"):
                if training_method == "two-stage" or epoch <= pretrain_epochs:
                    scheduler.step(np.mean(loss_list))
                elif training_method == "decision-focused":
                    scheduler.step(-np.mean(def_obj_list))
                else:
                    raise TypeError("Not Implemented Method")

            # Storing loss and defender utility
            epoch_loss_list.append(np.mean(loss_list))
            epoch_def_list.append(np.mean(def_obj_list))

            ################################### Print stuff after every epoch 
            np.random.shuffle(dataset)
            print("Mode: {}/ Epoch number: {}/ Loss: {}/ DefU: {}/ Simulated DefU: {}/ Fast DefU: {}".format(
                  mode, epoch, np.mean(loss_list), np.mean(def_obj_list), np.mean(simulated_def_obj_list), np.mean(fast_obj_list)))

            f_save.write("{}, {}, {}, {}, {}, {}\n".format(mode, epoch, np.mean(loss_list), np.mean(def_obj_list), np.mean(simulated_def_obj_list), np.mean(fast_obj_list)))
        
        if time_analysis:
            time4 = time.time()
            cprint (("TIME FOR THIS EPOCH:", time4-time3),'red')
            time3 = time4
        average_nodes = np.mean([x[0].number_of_nodes() for x in train_data] + [x[0].number_of_nodes() for x in validate_data] + [x[0].number_of_nodes() for x in test_data])
        average_edges = np.mean([x[0].number_of_edges() for x in train_data] + [x[0].number_of_edges() for x in validate_data] + [x[0].number_of_edges() for x in test_data])

    # ============== using results on validation set to choose solution ===============
    if training_method == "two-stage":
        max_epoch = np.argmax(validating_loss_list) # choosing based on loss
        final_loss = testing_loss_list[max_epoch]
        final_defender_utility = testing_defender_utility_list[max_epoch]
    else:
        max_epoch = np.argmax(validating_defender_utility_list) # choosing based on defender utility
        final_loss = testing_loss_list[max_epoch]
        final_defender_utility = testing_defender_utility_list[max_epoch]

    f_time.write("nodes, {}, edges, {}, epochs, {}, time, {}\n".format(average_nodes, average_edges, epoch, time.time() - beginning_time))
    f_summary.write("final loss, {}, final defender utility, {}\n".format(final_loss, final_defender_utility))
            
    return net2 ,training_loss_list, testing_loss_list, training_defender_utility_list, testing_defender_utility_list
    


def getDefUtility(single_data, unbiased_probs_pred, path_model, omega=4, restrict_mincut=True, verbose=False, initial_coverage_prob=None, training_mode=True, adding_edge=False, approximate=False):
    G, Fv, coverage_prob, phi_true, path_list, min_cut, log_prob, unbiased_probs_true = single_data
    
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

    # ========================== QP part ===========================
    if restrict_mincut:
        m = len(min_cut)
        edge_set = set(min_cut)
    else:
        m = G.number_of_edges()
        edge_set = set(range(m))

    A_matrix, b_matrix = torch.ones(1,m), torch.Tensor([budget])
    G_matrix = torch.cat((-torch.eye(m), torch.eye(m)))
    h_matrix = torch.cat((torch.zeros(m), torch.ones(m)))

    initial_coverage_prob = np.random.rand(len(edge_set))
    initial_coverage_prob = initial_coverage_prob / np.sum(initial_coverage_prob) * budget
    # if initial_coverage_prob is None:
    #     initial_coverage_prob = np.zeros(len(edge_set))

    pred_optimal_res = get_optimal_coverage_prob(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options, method=method, initial_coverage_prob=initial_coverage_prob, tol=tol, edge_set=edge_set) # scipy version
    pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
    # pred_optimal_coverage = torch.Tensor(get_optimal_coverage_prob_frank_wolfe(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, num_iterations=100, initial_coverage_prob=initial_coverage_prob, tol=tol, edge_set=edge_set)) # Frank Wolfe version

    if training_mode:
        solver_option = 'default'
        if solver_option == 'default':
            qp_solver = qpthlocal.qp.QPFunction(zhats=None, slacks=None, nus=None, lams=None)
        else:
            qp_solver = qpthlocal.qp.QPFunction(verbose=verbose, solver=qpthlocal.qp.QPSolvers.GUROBI,
                                           zhats=None, slacks=None, nus=None, lams=None)

        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega, approximate=approximate)
        Q_sym = (Q + Q.t()) / 2
    
        eigenvalues, eigenvectors = np.linalg.eig(Q_sym)
        eigenvalues = [x.real for x in eigenvalues]
        Q_regularized = Q_sym + torch.eye(len(edge_set)) * max(0, -min(eigenvalues)+1)
        
        is_symmetric = np.allclose(Q_sym.numpy(), Q_sym.numpy().T)
        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega, lib=torch, approximate=approximate)
        p = jac.view(1, -1) - pred_optimal_coverage @ Q_regularized
    
        if solver_option == 'default':
            coverage_qp_solution = qp_solver(Q_regularized, p, G_matrix, h_matrix, A_matrix, b_matrix)[0]       # Default version takes 1/2 x^T Q x + x^T p; not 1/2 x^T Q x + x^T p
        else:
            coverage_qp_solution = qp_solver(0.5 * Q_regularized, p, G_matrix, h_matrix, A_matrix, b_matrix)[0] # GUROBI version takes x^T Q x + x^T p; not 1/2 x^T Q x + x^T p

    else:
        coverage_qp_solution = pred_optimal_coverage

    pred_obj_value = objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)
    if pred_obj_value < -0.1: # unknown ERROR # TODO
        print("unknown behavior happened...")
        print("objective value (SLSQP): {}".format(pred_obj_value))
        coverage_qp_solution = torch.Tensor(initial_coverage_prob)

    # ========================= Error message =========================
    if (torch.norm(pred_optimal_coverage - coverage_qp_solution) > 1): # or 0.01 for GUROBI, 0.1 for qpth
        print('QP solution and scipy solution differ {} too much..., not backpropagating this instance'.format(torch.norm(pred_optimal_coverage - coverage_qp_solution)))
        print("objective value (SLSQP): {}".format(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
        print(pred_optimal_coverage)
        print("objective value (QP): {}".format(objective_function_matrix_form(coverage_qp_solution, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
        print(coverage_qp_solution)
        coverage_qp_solution = pred_optimal_coverage


    # ================== Evaluation on the ground truth ===============
    # ======================= Defender Utility ========================
    pred_defender_utility  = -(objective_function_matrix_form(coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))
    
    # ==================== Actual Defender Utility ====================
    # Running simulations to check the actual defender utility
    full_optimal_coverage = torch.zeros(G.number_of_edges())
    for idx, edge in enumerate(edge_set):
        full_optimal_coverage[edge] = coverage_qp_solution[idx].item()
    
    # _, simulated_defender_utility = attackerOracle(G, full_optimal_coverage, phi_true, omega=omega, num_paths=100)
    simulated_defender_utility = 0

    return pred_defender_utility, coverage_qp_solution, simulated_defender_utility

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
    parser.add_argument('--omega', type=float, default=1, help='risk aversion of the attacker')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')

    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')
    parser.add_argument('--number-graphs', type=int, default=1, help='number of different graphs in the dataset')
    parser.add_argument('--number-samples', type=int, default=100, help='number of samples per graph')
    parser.add_argument('--number-sources', type=int, default=2, help='number of randomly generated sources')
    parser.add_argument('--number-targets', type=int, default=2, help='number of randomly generated targets')

    parser.add_argument('--distribution', type=int, default=1, help='0 -> random walk distribution, 1 -> empirical distribution')
    parser.add_argument('--mincut', type=int, default=0, help='0 -> choose from all edges, 1 -> choose from a min-cut')
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
    restrict_mincut = True if args.mincut == 1 else False
    print('restrict mincut:', restrict_mincut)

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
    BATCH_SIZE = 5
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
    mincut_name = 'mincut' if restrict_mincut else 'global'
    if FIXED_GRAPH == 0:
        filepath_data    =      "results/random/{}_{}_n{}_p{}_b{}_noise{}_{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL, mincut_name)
        filepath_figure  =      "figures/random/{}_{}_n{}_p{}_b{}_noise{}_{}.png".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL, mincut_name)
        filepath_time    = "results/time/random/{}_{}_n{}_p{}_b{}_noise{}_{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL, mincut_name)
        filepath_summary =     "results/summary/{}_{}_n{}_p{}_b{}_noise{}_{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL, mincut_name)
    else:
        filepath_data    = "results/fixed/{}_{}_{}_test.csv"               .format(filename, training_method, mincut_name)
        filepath_figure  = "figures/fixed/{}_{}_{}_test.png"               .format(filename, training_method, mincut_name)
        filepath_time    = "results/time/fixed/{}_{}_b{}_{}.csv"           .format(filename, training_method, DEFENDER_BUDGET, mincut_name)
        filepath_summary = "results/summary/fixed/{}_{}_n{}_p{}_b{}_{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, mincut_name)

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
    
    time2 =time.time()
    if time_analysis:
        cprint (("DATA GENERATION: ", time2-time1), 'red')

    np.random.shuffle(train_data)

    print ("Training method: {}".format(training_method))
    print ("Data length train/test:", len(train_data), len(test_data))

    ############################## Training the ML models:    
    time3=time.time()
    # Learn the neural networks:
    net2, train_loss, test_loss, training_graph_def_u, testing_graph_def_u=learnEdgeProbs_simple(
                                train_data, validate_data, test_data, f_save, f_time, f_summary,
                                learning_model=learning_model_type,
                                lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE, 
                                optimizer=OPTIMIZER, omega=OMEGA, training_method=training_method, restrict_mincut=restrict_mincut)

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
        
