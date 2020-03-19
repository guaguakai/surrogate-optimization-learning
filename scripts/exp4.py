import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from generate_bar import read_file, generateBarChart
from generate_fig4 import return_yaxis, generateLineChart
from generate_time import load_time

if __name__=='__main__':

    labels = ['block-decision-focused', 'hybrid']
    print(labels)
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--filename', type=str, help='filename under folder results')
    parser.add_argument('--prob', type=float, default=0.2, help='input the probability used as input of random graph generator')
    parser.add_argument('--noise', type=float, default=0, help='noise level of the normalized features (in variance)')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')
    parser.add_argument('--cut-size', type=str, default='0.5n', help='block size')
    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')

    args = parser.parse_args()

    ############################# Parameters and settings:
    GRAPH_N_LOW  = args.number_nodes
    GRAPH_E_PROB_LOW  = args.prob

    DEFENDER_BUDGET = args.budget # This means the budget (sum of coverage prob) is <= DEFENDER_BUDGET*Number_of_edges 
    CUT_SIZE = args.cut_size
    ###############################
    filename = args.filename
    noise = args.noise

    block_selection_list = ['uniform', 'derivative', 'coverage', 'slack']

    for i, label in enumerate(labels):
        print(label)
        testing_defu = np.zeros((len(block_selection_list), 101))
        testing_defu_std = np.zeros((len(block_selection_list), 101))
        for j, block_selection in enumerate(block_selection_list):
            filepath = "results/random/exp4/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, noise)
            key_list = None
            (tr_loss, te_loss, tr_defu, te_defu, x1), (tr_loss_std, te_loss_std, tr_defu_std, te_defu_std, x1) = return_yaxis(filepath)
            testing_defu[j] = -np.array(te_defu)
            testing_defu_std[j] = te_defu_std

        xy_list = []
        xy_list.append((x1, testing_defu, block_selection_list, label, 'defender utility'))
        xy_list.append((x1, testing_defu + testing_defu_std, block_selection_list, label, 'defender utility')) # just because generate_fig4 is not configured correctly
        print(xy_list)
        np.savetxt('experiments/exp4_{}.csv'.format(label), testing_defu, delimiter=',')
        np.savetxt('experiments/exp4_std_{}.csv'.format(label), testing_defu_std, delimiter=',')

        linechart_save_filename = "results/excel/comparison/exp4/linechart_{}_{}_n{}_p{}_b{}_noise{}.png".format(filename, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, noise)

        generateLineChart(xy_list, linechart_save_filename)









