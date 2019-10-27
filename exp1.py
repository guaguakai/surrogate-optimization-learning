import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from generate_bar import read_file, generateBarChart
from generate_fig4 import return_yaxis, generateLineChart

if __name__=='__main__':

    # labels = ['two-stage', 'block-decision-focused']
    labels = ['two-stage', 'block-decision-focused', 'hybrid']
    # labels = ['two-stage', 'block-decision-focused', 'corrected-block-decision-focused', 'hybrid']
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
    NOISE_LEVEL = args.noise
    CUT_SIZE = args.cut_size
    ###############################
    filename = args.filename

    # ============================== generate bar chart =====================================
    bar_list = np.zeros((2, len(labels)))
    for i, label in enumerate(labels):
        filepath = "results/random/exp1/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        key_list = None
        if label == 'two-stage':
            loss_list, defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list = read_file(filepath, 'two-stage', key_list)
        else:
            loss_list, defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list = read_file(filepath, 'decision-focused', key_list)

        print('Opt mean:', np.mean(opt_loss_list), np.mean(opt_defu_list))
        loss_median = np.median(loss_list - opt_loss_list)
        defu_median = np.median(opt_defu_list - defu_list)

        loss_mean   = np.mean(loss_list - opt_loss_list)
        defu_mean   = np.mean(opt_defu_list - defu_list)

        bar_list[0,i] = defu_mean
        bar_list[1,i] = defu_median

    init_defu_median = np.median(opt_defu_list - init_defu_list)
    init_defu_mean   = np.mean(opt_defu_list - init_defu_list)
    # bar_list[0,-1] = init_defu_mean
    # bar_list[1,-1] = init_defu_median

    print('mean (ts, bdf, cbdf, hb, init):',   ','.join([str(x) for x in bar_list[0]]) + ',' + str(init_defu_mean))
    print('median (ts, bdf, cbdf, hb, init):', ','.join([str(x) for x in bar_list[1]]) + ',' + str(init_defu_median))

    save_filename = "results/excel/comparison/exp1/barchart_{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generateBarChart(bar_list, labels, save_filename)

    # ============================== generate line chart =====================================
    tr_loss, te_loss = [[]] * len(labels), [[]] * len(labels)
    tr_defu, te_defu = [[]] * len(labels), [[]] * len(labels)
    for i, label in enumerate(labels):
        filepath = "results/random/exp1/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        print(label)
        tr_loss[i], te_loss[i], tr_defu[i], te_defu[i], x1 = return_yaxis(filepath)

    xy_list = []
    xy_list.append((x1, np.array(tr_loss), labels, "Training Loss", 'KL Divergence'))
    xy_list.append((x1, np.array(te_loss), labels, "Testing Loss",  'KL Divergence'))
    xy_list.append((x1, np.array(tr_defu), labels, "Training Defender Utility", 'Defender utility'))
    xy_list.append((x1, np.array(te_defu), labels, "Testing Defender Utility",  'Defender utility'))

    save_filename = "results/excel/comparison/exp1/{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generateLineChart(xy_list, save_filename)

