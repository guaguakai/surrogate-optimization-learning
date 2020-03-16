import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from generate_bar import read_file, generateBarChart
from generate_fig4 import return_yaxis, generateLineChart
from generate_time import load_time

if __name__=='__main__':

    # labels = ['two-stage']
    labels = ['two-stage', 'hybrid']
    # labels = ['two-stage', 'decision-focused', 'block-decision-focused', 'hybrid']
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
    NOISE_LEVEL = args.noise
    CUT_SIZE = args.cut_size
    block_selection = args.block_selection
    ###############################
    filename = args.filename

    # ============================== generate bar chart =====================================
    defu_list_list = []
    bar_list = np.zeros((4, len(labels)))
    for i, label in enumerate(labels):
        filepath = "results/random/exp1/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        key_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50] #None
        if label == 'two-stage':
            loss_list, defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list = read_file(filepath, 'two-stage', key_list)
        else:
            loss_list, defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list = read_file(filepath, 'decision-focused', key_list)

        defu_list_list.append(defu_list)

        print('Opt mean:', np.mean(opt_loss_list), np.mean(opt_defu_list))
        loss_median = np.median(-loss_list + opt_loss_list)
        defu_median = np.median(-defu_list + opt_defu_list)

        loss_mean   = np.mean(-loss_list   + opt_loss_list)
        defu_mean   = np.mean(-defu_list   + opt_defu_list)

        bar_list[0,i] = defu_mean
        bar_list[1,i] = defu_median

        time_filepath = "results/time/random/exp1/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        training_time, optimizing_time = load_time(time_filepath)
        bar_list[2,i] = training_time
        bar_list[3,i] = optimizing_time

    from scipy import stats
    x = np.sum(defu_list_list[0] > defu_list_list[1])
    n = len(defu_list_list[0])
    print(x, n)
    print('p value:', stats.binom_test(x,n))

    init_defu_median = np.median(-init_defu_list + opt_defu_list)
    init_defu_mean   = np.mean(  -init_defu_list + opt_defu_list)
    # bar_list[0,-1] = init_defu_mean
    # bar_list[1,-1] = init_defu_median

    print('mean (ts, df, bdf, hb, init):',   ','.join([str(x) for x in bar_list[0]]) + ',' + str(init_defu_mean))
    print('training time (ts, df, bdf, hb):', ','.join([str(x) for x in bar_list[2]]))
    print('optimizing time (ts, df, bdf, hb):', ','.join([str(x) for x in bar_list[3]]))

    save_filename = "results/excel/comparison/exp1/barchart_{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generateBarChart(bar_list, labels, save_filename)

    # ============================== generate line chart =====================================
    tr_loss, te_loss = [[]] * len(labels), [[]] * len(labels)
    tr_defu, te_defu = [[]] * len(labels), [[]] * len(labels)
    for i, label in enumerate(labels):
        filepath = "results/random/exp1/{}_{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, block_selection, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        print(label)
        (tr_loss[i], te_loss[i], tr_defu[i], te_defu[i], x1), _ = return_yaxis(filepath)

    xy_list = []
    xy_list.append((x1, np.array(tr_loss), labels, "Training Loss", 'KL Divergence'))
    xy_list.append((x1, np.array(te_loss), labels, "Testing Loss",  'KL Divergence'))
    xy_list.append((x1, np.array(tr_defu), labels, "Training Defender Utility", 'Defender utility'))
    xy_list.append((x1, np.array(te_defu), labels, "Testing Defender Utility",  'Defender utility'))

    save_filename = "results/excel/comparison/exp1/{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generateLineChart(xy_list, save_filename)

