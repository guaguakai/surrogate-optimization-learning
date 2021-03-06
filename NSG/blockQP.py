# -*- coding: utf-8 -*-

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
from utils import phi2prob, prob2unbiased
# from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form

def train_model(train_data, validate_data, test_data, lr=0.1, learning_model='random_walk_distribution', block_selection='coverage',
                          n_epochs=150, batch_size=100, optimizer='adam', omega=4, training_method='two-stage', max_norm=0.1, block_cut_size=0.5):
    
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
    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    evaluate = False if training_method == 'two-stage' else True # for two-stage only
    pretrain_epochs = 0
    decay_rate = 0.95
    for epoch in range(-1, n_epochs):
        if epoch == n_epochs-1:
            evaluate = True

        epoch_forward_time, epoch_qp_time, epoch_backward_time, epoch_inference_time = 0, 0, 0, 0
        if epoch <= pretrain_epochs:
            ts_weight = 1
            df_weight = 0
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

            loss_list, def_obj_list = [], []
            for iter_n in tqdm.trange(len(dataset)):
                forward_start_time = time.time()
                G, Fv, coverage_prob, phi_true, path_list, cut, log_prob, unbiased_probs_true, previous_gradient = dataset[iter_n]
                n, m = G.number_of_nodes(), G.number_of_edges()
                
                # =============== Compute edge probabilities ===========
                Fv_torch   = torch.as_tensor(Fv, dtype=torch.float)
                edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
                phi_pred   = net2(Fv_torch, edge_index).view(-1) if epoch >= 0 else phi_true # when epoch < 0, testing the optimal loss and defender utility

                unbiased_probs_pred = phi2prob(G, phi_pred) if epoch >= 0 else unbiased_probs_true
                biased_probs_pred = prob2unbiased(G, -coverage_prob,  unbiased_probs_pred, omega=omega) # feeding negative coverage to be biased
                
                # =================== Compute loss =====================
                log_prob_pred = torch.zeros(1)
                for path in path_list:
                    for e in path: 
                        log_prob_pred -= torch.log(biased_probs_pred[e[0]][e[1]])
                log_prob_pred /= len(path_list)
                loss = (log_prob_pred - log_prob)[0]

                single_forward_time = time.time() - forward_start_time
                single_qp_time     = 0

                # ============== COMPUTE DEFENDER UTILITY ==============
                single_data = dataset[iter_n]

                if mode == 'testing' or mode == "validating" or epoch <= 0: # or training_method == "two-stage" or epoch <= 0:
                    cut_size = m
                    if evaluate or epoch <= 0:
                        def_obj, def_coverage, (single_inference_time, single_qp_time) = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=False, training_method=training_method, block_selection=block_selection) # feed forward only
                    else:
                        def_obj, def_coverage = torch.Tensor([-float('Inf')]), None
                else:
                    if training_method == "two-stage" or epoch <= pretrain_epochs:
                        cut_size = m
                        if evaluate:
                            def_obj, def_coverage, (single_inference_time, single_qp_time) = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=False, training_method=training_method, block_selection=block_selection) # most time-consuming part
                        else:
                            def_obj, def_coverage = torch.Tensor([-float('Inf')]), None
                            single_inference_time, single_qp_time = 0, 0
                            # ignore the time of computing defender utility
                    else:
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

                        def_obj, def_coverage, (single_inference_time, single_qp_time) = getDefUtility(single_data, unbiased_probs_pred, learning_model, cut_size=cut_size, omega=omega, verbose=False, training_mode=True,  training_method=training_method, block_selection=block_selection) # most time-consuming part
                
                if epoch > 0 and mode == "training":
                    epoch_forward_time   += single_forward_time
                    epoch_inference_time += single_inference_time
                    epoch_qp_time        += single_qp_time

                def_obj_list.append(def_obj.item())
                loss_list.append(loss.item())

                if (iter_n%batch_size == (batch_size-1)) and (epoch > 0) and (mode == "training"):
                    backward_start_time = time.time()
                    optimizer.zero_grad()
                    try:
                        if training_method == "two-stage" or epoch <= pretrain_epochs:
                            loss.backward()
                        elif training_method == "decision-focused" or training_method == "block-decision-focused" or training_method == 'corrected-block-decision-focused':
                            (-def_obj * m / cut_size).backward()
                        elif training_method == "hybrid":
                            ((-def_obj * m / cut_size) * df_weight + loss * ts_weight).backward()
                        else:
                            raise TypeError("Not Implemented Method")
                        torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=max_norm) # gradient clipping
                        optimizer.step()
                    except:
                        print("no grad is backpropagated...")
                    epoch_backward_time += time.time() - backward_start_time

            if (epoch > 0) and (mode == "validating"):
                if training_method == "two-stage":
                    scheduler.step(np.mean(loss_list))
                elif training_method == "decision-focused" or training_method == "block-decision-focused" or training_method == 'corrected-block-decision-focused' or training_method == 'hybrid':
                    scheduler.step(-np.mean(def_obj_list))
                else:
                    raise TypeError("Not Implemented Method")

            # Storing loss and defender utility
            epoch_loss_list.append(np.mean(loss_list))
            epoch_def_list.append(np.mean(def_obj_list))

            ################################### Print stuff after every epoch 
            np.random.shuffle(dataset)
            print("Mode: {}/ Epoch number: {}/ Loss: {}/ DefU: {}".format(
                  mode, epoch, np.mean(loss_list), np.mean(def_obj_list)))

        print('Forward time for this epoch: {}'.format(epoch_forward_time))
        print('QP time for this epoch: {}'.format(epoch_qp_time))
        print('Backward time for this epoch: {}'.format(epoch_backward_time))
        if epoch >= 0:
            forward_time   += epoch_forward_time
            inference_time += epoch_inference_time
            qp_time        += epoch_qp_time
            backward_time  += epoch_backward_time

        # ============= early stopping criteria =============
        kk = 3
        if epoch >= kk*2 -1:
            if training_method == 'two-stage':
                if evaluate:
                    break
                GE_counts = np.sum(np.array(validating_loss_list[1:][-kk:]) >= np.array(validating_loss_list[1:][-2*kk:-kk]) - 1e-4)
                print('Generalization error increases counts: {}'.format(GE_counts))
                if GE_counts == kk:
                    evaluate = True
            else: # surrogate or decision-focused
                GE_counts = np.sum(np.array(validating_defender_utility_list[1:][-kk:]) <= np.array(validating_defender_utility_list[1:][-2*kk:-kk]) + 1e-4)
                print('Generalization error increases counts: {}'.format(GE_counts))
                if GE_counts == kk:
                    break

    average_nodes = np.mean([x[0].number_of_nodes() for x in train_data] + [x[0].number_of_nodes() for x in validate_data] + [x[0].number_of_nodes() for x in test_data])
    average_edges = np.mean([x[0].number_of_edges() for x in train_data] + [x[0].number_of_edges() for x in validate_data] + [x[0].number_of_edges() for x in test_data])
    print('Total forward time: {}'.format(forward_time))
    print('Total inference time: {}'.format(inference_time))
    print('Total qp time: {}'.format(qp_time))
    print('Total backward time: {}'.format(backward_time))
            
    return net2, training_loss_list, validating_loss_list, testing_loss_list, training_defender_utility_list, validating_defender_utility_list, testing_defender_utility_list, (forward_time, inference_time, qp_time, backward_time), epoch
    

def getDefUtility(single_data, unbiased_probs_pred, path_model, cut_size, omega=4, verbose=False, initial_coverage_prob=None, training_mode=True, training_method='two-stage', block_selection='coverage'):
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

    forward_start_time = time.time()
    pred_optimal_res = get_optimal_coverage_prob(G, unbiased_probs_pred.detach(), U, initial_distribution, budget, omega=omega, options=options, method=method, initial_coverage_prob=initial_coverage_prob, tol=tol) # scipy version
    pred_optimal_coverage = torch.Tensor(pred_optimal_res['x'])
    if not pred_optimal_res['success']:
        print(pred_optimal_res)
        print('optimization fails...')
    inference_time = time.time() - forward_start_time

    # ======================== edge set choice =====================
    qp_start_time = time.time()
    first_order_derivative = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, np.arange(m), omega=omega, lib=torch)

    if block_selection == 'derivative':
        sample_distribution = np.abs(first_order_derivative.detach().numpy()) + 1e-3
    elif block_selection == 'coverage':
        sample_distribution = pred_optimal_coverage.detach().numpy() + 1e-3
    elif block_selection == 'uniform':
        sample_distribution = np.ones(m)
    elif block_selection == 'slack':
        sample_distribution = np.exp(-np.abs(pred_optimal_coverage.detach().numpy() - 0.5) * 5)
    else:
        raise ValueError('Not Implemented Block Selection')
    sample_distribution /= sum(sample_distribution)
    if training_method == 'block-decision-focused' or training_method == 'hybrid':
        # min_sum = 1e-2
        while True:
            edge_set = np.array(sorted(np.random.choice(range(m), size=cut_size, replace=False, p=sample_distribution)))
            # if sum(pred_optimal_coverage[edge_set]) > min_sum:
            break
    elif training_method == 'corrected-block-decision-focused':
        # min_sum = 1e-2
        while True:
            edge_set = np.array(sorted(np.random.choice(range(m), size=cut_size, replace=False, p=sample_distribution)))
            indices = np.arange(cut_size)
            np.random.shuffle(indices)
            indices1, indices2 = np.array_split(indices, 2)
            indices1, indices2 = sorted(indices1), sorted(indices2)
            edge_set1, edge_set2 = edge_set[indices1], edge_set[indices2]
            # if sum(pred_optimal_coverage[edge_set1]) > min_sum / 10 and sum(pred_optimal_coverage[edge_set2]) > min_sum / 10:
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

        hessian_start_time = time.time()
        Q = obj_hessian_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega)
        jac = dobj_dx_matrix_form(pred_optimal_coverage, G, unbiased_probs_pred, U, initial_distribution, edge_set, omega=omega, lib=torch)
        Q_sym = (Q + Q.t()) / 2
        hessian_time = time.time() - hessian_start_time
    
        # ------------------ regularization -----------------------
        Q_regularized = Q_sym.clone()
        reg_const = 0.1
        while True:
            # ------------------ eigen regularization -----------------------
            Q_regularized[range(cut_size), range(cut_size)] = torch.clamp(torch.diag(Q_sym), min=reg_const)
            try:
                L = torch.cholesky(Q_regularized)
                break
            except:
                reg_const *= 2

        p = jac.view(1, -1) - Q_regularized @ pred_optimal_coverage[edge_set]
 
        try:
            qp_solver = qpth.qp.QPFunction()
            coverage_qp_solution = qp_solver(Q_regularized, p, G_matrix, h_matrix, A_matrix, b_matrix)[0]       # Default version takes 1/2 x^T Q x + x^T p; not 1/2 x^T Q x + x^T p
            full_coverage_qp_solution = pred_optimal_coverage.clone()
            full_coverage_qp_solution[edge_set] = coverage_qp_solution
        except:
            print("QP solver fails... Usually because Q is not PSD")
            full_coverage_qp_solution = pred_optimal_coverage.clone()

        pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))

    else:
        full_coverage_qp_solution = pred_optimal_coverage.clone()
        pred_defender_utility  = -(objective_function_matrix_form(full_coverage_qp_solution, G, unbiased_probs_true, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega=omega))
    qp_time = time.time() - qp_start_time

    return pred_defender_utility, full_coverage_qp_solution, (inference_time, qp_time)

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
    parser.add_argument('--cut-size', type=str, default='0.5n', help='number of the defender budget')
    parser.add_argument('--block-selection', type=str, default='coverage', help='block selection')

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
    block_selection = args.block_selection

    training_mode = args.method
    method_dict = {0: 'two-stage', 1: 'decision-focused', 2: 'block-decision-focused', 4: 'hybrid'}
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
        CUT_SIZE = int(CUT_SIZE)
    GRAPH_TYPE = "random_graph" if FIXED_GRAPH == 0 else "fixed_graph"
    SEED = args.seed
    NOISE_LEVEL = args.noise
    if SEED == 0:
        SEED = np.random.randint(1, 100000)

    ###############################
    filename = args.filename
    if FIXED_GRAPH == 0:
        filepath_data    =      "results/performance/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, training_method, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        filepath_time    = "results/time/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, training_method, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
    else:
        raise NotImplementedError

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
    print('Block selection:', block_selection)

    ############################## Training the ML models:
    net2, training_loss, validating_loss, testing_loss, training_defu, validating_defu, testing_defu, (forward_time, inference_time, qp_time, backward_time), epoch = train_model(
                                train_data, validate_data, test_data,
                                learning_model=learning_model_type, block_selection=block_selection,
                                lr=LR, n_epochs=N_EPOCHS,batch_size=BATCH_SIZE,
                                optimizer=OPTIMIZER, omega=OMEGA, training_method=training_method, block_cut_size=CUT_SIZE)

    f_save = open(filepath_data, 'a')
    f_time = open(filepath_time, 'a')

    # ============== recording the important information only ================
    validating_loss = np.array(validating_loss)
    validating_defu = np.array(validating_defu)
    validating_loss[np.isnan(validating_loss)] = np.inf
    validating_defu[np.isnan(validating_defu)] = -np.inf

    if training_method == 'two-stage': # using the last index
        selected_idx = -1 # np.argmin(validating_loss[1:])
    else:
        selected_idx = np.argmax(validating_defu[1:])

    f_time.write('Random seed, {}, forward time, {}, inference time, {}, qp time, {}, backward_time, {}, epoch, {}\n'.format(SEED, forward_time, inference_time, qp_time, backward_time, epoch))
    f_save.write('Random seed, {},'.format(SEED) +
            'training loss, {}, training defu, {}, training opt, {}, '.format(training_loss[1:][selected_idx], training_defu[1:][selected_idx], training_defu[0]) +
            'validating loss, {}, validating defu, {}, validating opt, {},'.format(validating_loss[1:][selected_idx], validating_defu[1:][selected_idx], validating_defu[0]) +
            'testing loss, {}, testing defu, {}, testing opt, {}\n'.format(testing_loss[1:][selected_idx], testing_defu[1:][selected_idx], testing_defu[0])
            )

    f_save.close()
    f_time.close()

    ############################# Print the summary:
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
                "Entire graph defender utility:":(testing_defu[0],testing_defu[-1]),
                "Test Loss:": testing_loss} 
    
    cprint (all_params, 'green')
        
