import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from generate_bar import read_file, generateBarChart
from generate_fig4 import return_yaxis, generateLineChart
from generate_time import load_time

if __name__=='__main__':

    # labels = ['block-decision-focused', 'hybrid']
    labels = ['two-stage', 'decision-focused', 'block-decision-focused', 'hybrid']
    # labels = ['two-stage', 'block-decision-focused', 'hybrid']
    print(labels)
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--filename', type=str, help='filename under folder results')
    parser.add_argument('--prob', type=float, default=0.2, help='input the probability used as input of random graph generator')
    parser.add_argument('--noise', type=float, default=0, help='noise level of the normalized features (in variance)')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')
    parser.add_argument('--cut-size', type=str, default='0.5n', help='block size')
    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')
    parser.add_argument('--block-selection', type=str, default='coverage', help='block selection')

    args = parser.parse_args()

    ############################# Parameters and settings:
    GRAPH_N_LOW  = args.number_nodes
    GRAPH_E_PROB_LOW  = args.prob

    DEFENDER_BUDGET = args.budget # This means the budget (sum of coverage prob) is <= DEFENDER_BUDGET*Number_of_edges 
    CUT_SIZE = args.cut_size
    ###############################
    filename = args.filename
    block_selection = args.block_selection

    noise_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    defu_list = np.zeros((len(labels), len(noise_list)))
    for i, label in enumerate(labels):
        print(label)
        for j, noise in enumerate(noise_list):
            filepath = "results/random/exp3/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, noise)

            key_list = None
            if label == 'two-stage':
                loss, defu, opt_loss, opt_defu, init_loss, init_defu = read_file(filepath, 'two-stage', key_list)
            else:
                loss, defu, opt_loss, opt_defu, init_loss, init_defu = read_file(filepath, 'decision-focused', key_list)

            loss_mean   = np.mean(-loss + opt_loss)
            defu_mean   = np.mean(-defu + opt_defu)

            defu_list[i,j] = defu_mean

    for label, defu in zip(labels, defu_list):
        print(label + ',' + ','.join([str(x) for x in defu]))







