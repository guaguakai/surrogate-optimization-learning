import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from generate_bar import read_file, generateBarChart
from generate_fig4 import return_yaxis, generateLineChart
from generate_time import load_time

if __name__ == '__main__':
    
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--filename', type=str, help='filename under folder results')
    parser.add_argument('--prob', type=float, default=0.2, help='input the probability used as input of random graph generator')
    parser.add_argument('--noise', type=float, default=0, help='noise level of the normalized features (in variance)')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')
    parser.add_argument('--number-nodes', type=int, default=10, help='input node size for randomly generated graph')

    args = parser.parse_args()

    ############################# Parameters and settings:
    GRAPH_N_LOW  = args.number_nodes
    GRAPH_E_PROB_LOW  = args.prob

    DEFENDER_BUDGET = args.budget # This means the budget (sum of coverage prob) is <= DEFENDER_BUDGET*Number_of_edges
    NOISE_LEVEL = args.noise
    ###############################
    filename = args.filename

    cut_size_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [str(x) for x in cut_size_list]
    
    method_list = ['block-decision-focused', 'hybrid']
    key_list = None # list(set(range(1,31)) - set([21, 25]))
    for method in method_list:
        testing_defu = np.zeros((len(cut_size_list), 101))
        bar_list = np.zeros((2, len(cut_size_list)))
        for i, cut_size in enumerate(cut_size_list):
            filepath = "results/random/exp2/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, cut_size, NOISE_LEVEL)
            loss_list, defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list = read_file(filepath, 'decision-focused', key_list)
    
            print('Opt mean:', np.mean(opt_loss_list), np.mean(opt_defu_list))
            loss_median = np.median(loss_list - opt_loss_list)
            defu_median = np.median(opt_defu_list - defu_list)
    
            loss_mean   = np.mean(loss_list - opt_loss_list)
            defu_mean   = np.mean(opt_defu_list - defu_list)
    
            bar_list[0,i] = defu_mean
            bar_list[1,i] = defu_median


        tr_loss, te_loss = [[]] * len(labels), [[]] * len(labels)
        tr_defu, te_defu = [[]] * len(labels), [[]] * len(labels)
        training_time_list = np.zeros(len(cut_size_list))
        optimizing_time_list = np.zeros(len(cut_size_list))
        for i, cut_size in enumerate(cut_size_list):
            filepath = "results/random/exp2/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, cut_size, NOISE_LEVEL)
            (tr_loss[i], te_loss[i], tr_defu[i], te_defu[i], x1), _ = return_yaxis(filepath)
            testing_defu[i] = - np.array(te_defu[i])

            time_filepath = "results/time/random/exp2/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, cut_size, NOISE_LEVEL)
            training_time, optimizing_time = load_time(time_filepath)
            training_time_list[i] = training_time
            optimizing_time_list[i] = optimizing_time

        xy_list = []
        xy_list.append((x1, np.array(tr_loss), labels, "Training Loss", 'KL Divergence'))
        xy_list.append((x1, np.array(te_loss), labels, "Testing Loss",  'KL Divergence'))
        xy_list.append((x1, np.array(tr_defu), labels, "Training Defender Utility", 'Defender utility'))
        xy_list.append((x1, np.array(te_defu), labels, "Testing Defender Utility",  'Defender utility'))

        init_defu_median = np.median(opt_defu_list - init_defu_list)
        init_defu_mean   = np.mean(opt_defu_list - init_defu_list)
    
        print('mean  :', ','.join([str(x) for x in bar_list[0]]) + ',' + str(init_defu_mean))
        print('training time:', training_time_list)
        print('optimizing time:', optimizing_time_list)
        np.savetxt('experiments/exp2_{}.csv'.format(method), testing_defu, delimiter=',')
    
        barchart_save_filename = "results/excel/comparison/exp2/barchart_{}_{}_n{}_p{}_b{}_noise{}.png".format(filename, method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
        linechart_save_filename = "results/excel/comparison/exp2/linechart_{}_{}_n{}_p{}_b{}_noise{}.png".format(filename, method, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
        generateBarChart(bar_list, labels, barchart_save_filename)
        generateLineChart(xy_list, linechart_save_filename)

