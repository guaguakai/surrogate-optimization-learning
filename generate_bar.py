# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:06:15 2019

@author: Aditya
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_file(filename, method):
    f = open(filename, 'r')
    data = {}
    
    #d={'lol':[[] for i in range(max_epochs+1)]}
    
    final_loss_list = []
    final_defu_list = []
    opt_defu_list = []
    opt_loss_list = []
    while True:
        item = f.readline()
        if not item: break
        item = item[:-1].split(',')
        if item[0] == 'Random seed':
            seed = int(item[1])
            data[seed] = {'tr_loss': [], 'val_loss':[], 'te_loss': [], 'tr_defu': [], 'val_defu':[], 'te_defu': []}
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
                    opt_defu_list.append(opt_te_defu)
                    opt_loss_list.append(opt_te_loss)
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

    for key in data:
        if method == 'two-stage':
            tmp_idx = np.argmin(data[key]['val_loss'])
            tmp_loss = data[key]['te_loss'][tmp_idx]
            tmp_defu = data[key]['te_defu'][tmp_idx]
        elif method == 'decision-focused':
            tmp_idx = np.argmax(data[key]['val_defu'])
            tmp_loss = data[key]['te_loss'][tmp_idx]
            tmp_defu = data[key]['te_defu'][tmp_idx]
        else:
            raise ValueError
        final_loss_list.append(tmp_loss)
        final_defu_list.append(tmp_defu)

    print(method, len(data.keys()))
    print(sorted(data.keys()))

    final_loss = np.mean(final_loss_list)
    final_defu = np.mean(final_defu_list)
    opt_loss = np.mean(opt_loss_list)
    opt_defu = np.mean(opt_defu_list)
    init_loss = np.mean([data[key]['te_loss'][0] for key in data])
    init_defu = np.mean([data[key]['te_defu'][0] for key in data])
        
    return final_loss, final_defu, opt_loss, opt_defu, init_loss, init_defu

def generatePlot(bar_list, labels, filename):
    
    fig, axs = plt.subplots(1, len(bar_list))

    for i in range(len(bar_list)):
        data_opt, data_df, data_2s, data_init = bar_list[i]
        axs[i].bar(labels, [data_opt, data_df, data_2s, data_init])
        # axs[len(xy_list) + i].yticks(np.arange(2), (label_df, label_2d))

    #epochs = len(train_loss) - 1
    #x=range(-1, epochs)
    plt.autoscale()
    plt.savefig("./results/excel/comparison/"+filename)
    plt.show()

if __name__=='__main__':
    
    labels = ['OPT', 'DF-block', '2S', 'Initial']
    print(labels)
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

    #f="./results/0808normto10_linearphi_dist_tenruns_fullINIT_decision-focused_n20_p0.3_b2.0_global - Copy.csv"
    file1 = "results/random/{}_{}_n{}_p{}_b{}_noise{}.csv".format(filename, 'block-decision-focused', GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    file2 = "results/random/{}_{}_n{}_p{}_b{}_noise{}.csv".format(filename, 'two-stage', GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)

    #to_plot,x,epochs,=return_yaxis(f)
    df_loss, df_defu, opt_loss, opt_defu, init_loss, init_defu = read_file(file1, 'decision-focused')
    ts_loss, ts_defu, _, _, _, _ = read_file(file2, 'two-stage')
    
    bar_list = [(opt_loss, df_loss, ts_loss, init_loss), (opt_defu, df_defu, ts_defu, init_defu)]
    print('loss:', bar_list[0])
    print('defu:', bar_list[1])
    
    save_filename = "{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generatePlot(bar_list, labels, save_filename)
    

    
    #print(len(to_plot['tr_loss'][0]))
