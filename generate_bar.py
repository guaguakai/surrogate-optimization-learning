# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:06:15 2019

@author: Aditya
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_file(filename, method, key_list=None):
    f = open(filename, 'r')
    data = {}
    
    #d={'lol':[[] for i in range(max_epochs+1)]}
    
    final_loss_list = []
    final_defu_list = []
    opt_defu_list = []
    opt_loss_list = []
    print(filename)
    while True:
        item = f.readline()
        if not item: break
        item = item[:-1].split(',')
        if item[0] == 'Random seed':
            seed = int(item[1])
            data[seed] = {'tr_loss': [], 'val_loss':[], 'te_loss': [], 'tr_defu': [], 'val_defu':[], 'te_defu': [], 'opt_defu': None, 'opt_loss': None}
        elif item[0] == 'mode':
            continue
        else:
            epoch = int(item[1])
            if epoch == -1:
                if item[0]=='training':
                    opt_tr_loss = float(item[2])
                    opt_tr_defu = float(item[3])
                elif item[0]=='validating':
                    opt_val_loss = float(item[2])
                    opt_val_defu = float(item[3])
                elif item[0]=='testing':
                    opt_te_loss = float(item[2])
                    opt_te_defu = float(item[3])
                    data[seed]['opt_loss'] = opt_te_loss
                    data[seed]['opt_defu'] = opt_te_defu
            else:
                if item[0]=='training':
                    data[seed]['tr_loss'].append(float(item[2]))
                    data[seed]['tr_defu'].append(float(item[3]))
                elif item[0]=='validating':
                    data[seed]['val_loss'].append(float(item[2]))
                    data[seed]['val_defu'].append(float(item[3]))
                elif item[0]=='testing':
                    data[seed]['te_loss'].append(float(item[2]))
                    data[seed]['te_defu'].append(float(item[3]))

    if key_list is None:
        key_list = data.keys()
    for key in key_list:
        tmp_opt_loss = data[key]['opt_loss']
        tmp_opt_defu = data[key]['opt_defu']
        if method == 'two-stage':
            tmp_idx = np.argmin(data[key]['val_loss'])
            tmp_loss = data[key]['te_loss'][tmp_idx]
            tmp_defu = data[key]['te_defu'][tmp_idx]
        elif method == 'decision-focused' or method == 'hybrid':
            tmp_idx = np.argmax(data[key]['val_defu'])
            tmp_loss = data[key]['te_loss'][tmp_idx]
            tmp_defu = data[key]['te_defu'][tmp_idx]
        else:
            raise ValueError
        final_loss_list.append(tmp_loss)
        final_defu_list.append(tmp_defu)
        opt_loss_list.append(tmp_opt_loss)
        opt_defu_list.append(tmp_opt_defu)

    print(method, len(data.keys()))
    print(list(key_list))

    final_loss_list = np.array(final_loss_list)
    final_defu_list = np.array(final_defu_list)
    opt_loss_list = np.array(opt_loss_list)
    opt_defu_list = np.array(opt_defu_list)

    init_loss_list = np.array([data[key]['te_loss'][0] for key in key_list])
    init_defu_list = np.array([data[key]['te_defu'][0] for key in key_list])
    f.close()
        
    return final_loss_list, final_defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list

def generateBarChart(bar_list, labels, filename):
    
    fig, axs = plt.subplots(1, len(bar_list))

    for i in range(len(bar_list)):
        axs[i].bar(labels, bar_list[i], width=0.15)
        # axs[len(xy_list) + i].yticks(np.arange(2), (label_df, label_2d))

    axs[0].title.set_text('Mean Regret')
    axs[1].title.set_text('Median Regret')

    #epochs = len(train_loss) - 1
    #x=range(-1, epochs)
    plt.autoscale()
    plt.savefig(filename)
    plt.show()

if __name__=='__main__':
    
    # labels = ['two-stage', 'block-decision-focused']
    labels = ['two-stage', 'block-decision-focused', 'corrected-block-decision-focused', 'hybrid']
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

    bar_list = np.zeros((2, len(labels)))
    for i, label in enumerate(labels):
        filepath = "results/random/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        key_list = None
        if label == 'two-stage':
            # loss_list, defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list = read_file(filepath, 'decision-focused', key_list)
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
    
    save_filename = "results/excel/comparison/barchart_{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generateBarChart(bar_list, labels, save_filename)
    

    
    #print(len(to_plot['tr_loss'][0]))
