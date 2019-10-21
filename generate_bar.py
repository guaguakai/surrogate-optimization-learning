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

    if key_list is None:
        key_list = data.keys()
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
    print(key_list)

    final_loss_list = np.array(final_loss_list)
    final_defu_list = np.array(final_defu_list)
    opt_loss_list = np.array(opt_loss_list)
    opt_defu_list = np.array(opt_defu_list)

    init_loss_list = np.array([data[key]['te_loss'][0] for key in data])
    init_defu_list = np.array([data[key]['te_defu'][0] for key in data])
        
    return final_loss_list, final_defu_list, opt_loss_list, opt_defu_list, init_loss_list, init_defu_list

def generatePlot(bar_list, labels, filename):
    
    fig, axs = plt.subplots(1, len(bar_list))

    for i in range(len(bar_list)):
        data_df, data_2s, data_hb, data_init = bar_list[i]
        axs[i].bar(labels, [data_df, data_2s, data_hb, data_init])
        # axs[len(xy_list) + i].yticks(np.arange(2), (label_df, label_2d))

    axs[0].title.set_text('Mean Regret')
    axs[1].title.set_text('Median Regret')


    #epochs = len(train_loss) - 1
    #x=range(-1, epochs)
    plt.autoscale()
    plt.savefig("./results/excel/comparison/"+filename)
    plt.show()

if __name__=='__main__':
    
    labels = ['DF-block', '2S', 'Hybrid', 'Initial']
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
    file3 = "results/random/{}_{}_n{}_p{}_b{}_noise{}.csv".format(filename, 'hybrid', GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)

    #to_plot,x,epochs,=return_yaxis(f)
    # key_list = [4, 6, 7, 8, 1, 5]
    key_list = None
    df_loss_list, df_defu_list, df_opt_loss_list, df_opt_defu_list, df_init_loss_list, df_init_defu_list = read_file(file1, 'decision-focused', key_list)
    ts_loss_list, ts_defu_list, ts_opt_loss_list, ts_opt_defu_list, ts_init_loss_list, ts_init_defu_list = read_file(file2, 'two-stage', key_list)
    hb_loss_list, hb_defu_list, hb_opt_loss_list, hb_opt_defu_list, hb_init_loss_list, hb_init_defu_list = read_file(file3, 'two-stage', key_list)

    print('Opt mean:', np.mean(df_opt_loss_list), np.mean(df_opt_defu_list))
    df_loss_median = np.median(df_loss_list - df_opt_loss_list)
    df_defu_median = np.median(df_defu_list - df_opt_defu_list)
    init_loss_median = np.median(df_init_loss_list - df_opt_loss_list)
    init_defu_median = np.median(df_init_defu_list - df_opt_defu_list)
    ts_loss_median = np.median(ts_loss_list - ts_opt_loss_list)
    ts_defu_median = np.median(ts_defu_list - ts_opt_defu_list)
    hb_loss_median = np.median(hb_loss_list - hb_opt_loss_list)
    hb_defu_median = np.median(hb_defu_list - hb_opt_defu_list)

    df_loss_mean   = np.mean(df_loss_list - df_opt_loss_list)
    df_defu_mean   = np.mean(df_defu_list - df_opt_defu_list)
    init_loss_mean = np.mean(df_init_loss_list - df_opt_loss_list)
    init_defu_mean = np.mean(df_init_defu_list - df_opt_defu_list)
    ts_loss_mean   = np.mean(ts_loss_list - ts_opt_loss_list)
    ts_defu_mean   = np.mean(ts_defu_list - ts_opt_defu_list)
    hb_loss_mean   = np.mean(hb_loss_list - hb_opt_loss_list)
    hb_defu_mean   = np.mean(hb_defu_list - hb_opt_defu_list)
    
    bar_list = [(df_defu_mean,   ts_defu_mean,   hb_defu_mean,   init_defu_mean),
                (df_defu_median, ts_defu_median, hb_defu_median, init_defu_median)]
    print('mean (df, ts, hb, init):', bar_list[0])
    print('median (df, ts, hb, init):', bar_list[1])
    
    save_filename = "barchart_{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generatePlot(bar_list, labels, save_filename)
    

    
    #print(len(to_plot['tr_loss'][0]))
