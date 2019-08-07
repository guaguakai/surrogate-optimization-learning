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
from obsoleteCode import *
from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form
import qpthlocal

def plotEverything(all_params,train_loss, test_loss, training_graph_def_u, testing_graph_def_u, filepath="figure/"):
    learning_model=all_params['learning model']

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    fig.text(0.5, 0.04, '# epochs', ha='center', va='center')
    fig.text(0.06, 0.75, 'KL-divergence loss', ha='center', va='center', rotation='vertical')
    fig.text(0.06, 0.25, 'Defender utility', ha='center', va='center', rotation='vertical')

    epochs = len(train_loss) - 1
    x=range(-1, epochs)

    # Training loss
    ax1.plot(x, train_loss, label='Training loss')
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
    

def learnEdgeProbs_simple(train_data, validate_data, test_data, f_save, f_time, lr=0.1, learning_model='random_walk_distribution'
                          ,n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage', restrict_mincut=True):

    
    time1=time.time()
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
    time2=time.time()
    
    ######################################################
    #                   TRAINING LOOP
    ######################################################
    time3 = time.time()

    f_save.write("mode, epoch, average loss, defender utility\n")

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

                #if (training_method == 'decision-focused' and (mode!="testing")) or (epoch == n_epochs - 1) or (not time_analysis):
                def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=True, verbose=False)
		if (not (restrict_mincut)):
			G=single_data[0]
			mincut=single_data[-3]
			E=len(list(G.edges()))
			intial_coverage=[def_coverage[mincut.index(i)] if i in mincut else 0 for i in range(E)]
			 
	                def_obj, def_coverage, simulated_def_obj = getDefUtility(single_data, unbiased_probs_pred, learning_model, omega=omega, restrict_mincut=restrict_mincut, verbose=False, initial_coverage_prob=initial_coverage)
                
		def_obj_list.append(def_obj.item())
                simulated_def_obj_list.append(simulated_def_obj)

                loss_list.append(loss.item())

                # backpropagate using loss when training two-stage and using -defender utility when training end-to-end
                if training_method == "two-stage":
                    batch_loss += loss
                elif training_method == "decision-focused":
                    batch_loss += (-def_obj)
                else:
                    raise TypeError("Not Implemented Method")

                # print(batch_loss)
                if (iter_n%batch_size == (batch_size-1)) and (epoch > 0) and (mode == "training"):
                    optimizer.zero_grad()
                    batch_loss.backward()
                    # print(torch.norm(net2.gcn1.weight.grad))
                    # print(torch.norm(net2.gcn2.weight.grad))
                    # print(torch.norm(net2.fc1.weight.grad))
                    optimizer.step()
                    batch_loss = 0

                # print('running time for optimization:', time.time() - start_time)

            if (epoch > 0) and (mode == "validating"):
                if training_method == "two-stage":
                    scheduler.step(np.sum(loss_list))
                elif training_method == "decision-focused":
                    scheduler.step(np.sum(def_obj_list))
                else:
                    raise TypeError("Not Implemented Method")

            # Storing loss and defender utility
            epoch_loss_list.append(np.mean(loss_list))
            epoch_def_list.append(np.mean(def_obj_list))

            ################################### Print stuff after every epoch 
            np.random.shuffle(dataset)
            print("Mode: {}/ Epoch number: {}/ Average training loss: {}/ Average defender objective: {}/ Simulated defender objective: {}".format(
                  mode, epoch, np.mean(loss_list), np.mean(def_obj_list), np.mean(simulated_def_obj_list)))

            f_save.write("{}, {}, {}, {}, {}\n".format(mode, epoch, np.mean(loss_list), np.mean(def_obj_list), np.mean(simulated_def_obj_list)))
        
        time4=time.time()
        if time_analysis:
            cprint (("TIME FOR THIS EPOCH:", time4-time3),'red')
        average_nodes = np.mean([x[0].number_of_nodes() for x in train_data] + [x[0].number_of_nodes() for x in validate_data] + [x[0].number_of_nodes() for x in test_data])
        average_edges = np.mean([x[0].number_of_edges() for x in train_data] + [x[0].number_of_edges() for x in validate_data] + [x[0].number_of_edges() for x in test_data])

        f_time.write("nodes, {}, edges, {}, epochs, {}, time, {}\n".format(average_nodes, average_edges, epoch, time4-time3))
        time3=time4
            
    return net2 ,training_loss_list, testing_loss_list, training_defender_utility_list, testing_defender_utility_list
    


def getDefUtility(single_data, unbiased_probs_pred, path_model, omega=4, restrict_mincut=True, verbose=False, initial_coverage_prob=None):
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
    for tmp_iter in range(1): # no retry now
        if restrict_mincut:
            m = len(min_cut)
            edge_set = set(min_cut)
        else:
            m = G.number_of_edges()
            edge_set = set(range(m))

        A_matrix, b_matrix = torch.Tensor(), torch.Tensor()
        G_matrix = torch.cat((-torch.eye(m), torch.eye(m), torch.ones(1,m)))
        h_matrix = torch.cat((torch.zeros(m), torch.ones(m), torch.Tensor([budget])))

        # initial_coverage_prob = np.random.rand(nx.number_of_edges(G))
        # initial_coverage_prob = initial_coverage_prob / np.sum(initial_coverage_prob) * budget * 0.1
	if initial_coverage_prob==None:
	        initial_coverage_prob = np.zeros(len(edge_set))

        pred_optimal_res = get_optimal_coverage_prob(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options, method=method, initial_coverage_prob=initial_coverage_prob, tol=tol, edge_set=edge_set)

        pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
        qp_solver = qpthlocal.qp.QPFunction(zhats=None, slacks=None, nus=None, lams=None)
        # qp_solver = qpthlocal.qp.QPFunction(verbose=verbose, solver=qpthlocal.qp.QPSolvers.GUROBI,
        #                                zhats=None, slacks=None, nus=None, lams=None)

        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega)
        Q_sym = Q #+ Q.t()) / 2

        eigenvalues, eigenvectors = np.linalg.eig(Q_sym)
        eigenvalues = [x.real for x in eigenvalues]
        Q_regularized = Q_sym - torch.eye(len(edge_set)) * min(0, min(eigenvalues)-10)
        
        is_symmetric = np.allclose(Q_sym.numpy(), Q_sym.numpy().T)
        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega, lib=torch)
        p = jac.view(1, -1) - pred_optimal_coverage @ Q_regularized

        coverage_qp_solution = qp_solver(0.5 * Q_regularized, p, G_matrix, h_matrix, A_matrix, b_matrix)[0] # GUROBI version takes x^T Q x + x^T p; not 1/2 x^T Q x + x^T p

        # ======================= Defender Utility ========================
        pred_defender_utility  = -(objective_function_matrix_form(coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))

        # ==================== Actual Defender Utility ====================
        # Running simulations to check the actual defender utility
        full_optimal_coverage = torch.zeros(G.number_of_edges())
        for idx, edge in enumerate(edge_set):
            full_optimal_coverage[edge] = coverage_qp_solution[idx].item()

        # _, simulated_defender_utility = attackerOracle(G, full_optimal_coverage, phi_true, omega=omega, num_paths=100)
        simulated_defender_utility = 0

        # ========================= Error message =========================
        if (torch.norm(pred_optimal_coverage - coverage_qp_solution) > 0.5): # or 0.01 for GUROBI, 0.1 for qpth
            print('QP solution and scipy solution differ too much...', torch.norm(pred_optimal_coverage - coverage_qp_solution))
            if verbose:
                print(pred_optimal_res)
                print("Minimum Eigenvalue: {}".format(min(eigenvalues)))
                print("Hessian: {}".format(Q_sym))
                print("Gradient: {}".format(jac))
                print("Eigen decomposition: {}".format(np.linalg.eig(Q_sym.detach().numpy())[0]))
                print("Eigen decomposition: {}".format(np.linalg.eig(Q_regularized.detach().numpy())[0]))
                print("objective value (SLSQP): {}".format(objective_function_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
                print("objective value (QP): {}".format(objective_function_matrix_form(coverage_qp_solution, G, unbiased_probs_pred, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega)))
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
    parser.add_argument('--omega', type=float, default=4, help='risk aversion of the attacker')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')

    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')
    parser.add_argument('--number-graphs', type=int, default=1, help='number of different graphs in the dataset')
    parser.add_argument('--number-samples', type=int, default=100, help='number of samples per graph')
    parser.add_argument('--number-sources', type=int, default=2, help='number of randomly generated sources')
    parser.add_argument('--number-targets', type=int, default=2, help='number of randomly generated targets')

    parser.add_argument('--distribution', type=int, default=1, help='0 -> random walk distribution, 1 -> empirical distribution')
    parser.add_argument('--mincut', type=int, default=1, help='0 -> choose from all edges, 1 -> choose from a min-cut')
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
    EMPIRICAL_SAMPLES_PER_INSTANCE = 50
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
    filename = args.filename
    mincut_name = 'mincut' if restrict_mincut else 'global'
    if FIXED_GRAPH == 0:
        filepath_data   = "results/random/{}_{}_n{}_p{}_b{}_{}.csv".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, mincut_name)
        filepath_figure = "figures/random/{}_{}_n{}_p{}_b{}_{}.png".format(filename, training_method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, mincut_name)
        filepath_time   = "results/time/random/{}_{}_b{}_{}.csv".format(filename, training_method, DEFENDER_BUDGET, mincut_name)
    else:
        filepath_data   = "results/fixed/{}_{}_{}_test.csv".format(filename, training_method, mincut_name)
        filepath_figure = "figures/fixed/{}_{}_{}_test.png".format(filename, training_method, mincut_name)
        filepath_time   = "results/time/fixed/{}_{}_b{}_{}.csv".format(filename, training_method, DEFENDER_BUDGET, mincut_name)

    f_save = open(filepath_data, 'a')
    f_time = open(filepath_time, 'a')
      
    ############################### Data genaration:
    train_data, validate_data, test_data = generateSyntheticData(feature_size, path_type=learning_model_type,
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
    net2, train_loss, test_loss, training_graph_def_u, testing_graph_def_u=learnEdgeProbs_simple(
                                train_data, validate_data, test_data, f_save, f_time,
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
    #############################            
    if plot_everything:
        plotEverything(all_params,train_loss, test_loss, training_graph_def_u, testing_graph_def_u, filepath=filepath_figure)
        
