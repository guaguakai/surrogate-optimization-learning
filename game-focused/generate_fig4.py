# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:06:15 2019

@author: Aditya
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse

def return_yaxis(filename):
    f = open(filename, 'r')
    max_epochs = 100
    to_plot={'tr_loss': [[] for _ in range(max_epochs+1)],
             'val_loss':[[] for _ in range(max_epochs+1)],
             'te_loss': [[] for _ in range(max_epochs+1)],
             'opt_loss': [[] for _ in range(max_epochs+1)],
             'tr_defu': [[] for _ in range(max_epochs+1)],
             'val_defu':[[] for _ in range(max_epochs+1)],
             'te_defu': [[] for _ in range(max_epochs+1)],
             'opt_defu': [[] for _ in range(max_epochs+1)]}
    
    #d={'lol':[[] for i in range(max_epochs+1)]}
    
    while True:
        item = f.readline()
        if not item: break
        item = item[:-1].split(',')
        if len(item) <= 3 or item[0] == 'mode':
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
            else:
                if item[0]=='training':
                    to_plot['tr_loss'][epoch].append(float(item[2]) - opt_tr_loss)
                    to_plot['tr_defu'][epoch].append(float(item[3]) - opt_tr_defu)
                elif item[0]=='validating':
                    to_plot['val_loss'][epoch].append(float(item[2]) - opt_val_loss)
                    to_plot['val_defu'][epoch].append(float(item[3]) - opt_val_defu)
                elif item[0]=='testing':
                    to_plot['te_loss'][epoch].append(float(item[2]) - opt_te_loss)
                    to_plot['te_defu'][epoch].append(float(item[3]) - opt_te_defu)

    for key in to_plot:
        to_plot[key] = np.array(to_plot[key])
    # _, num_samples = to_plot['tr_loss'].shape

    # MEAN
    tr_loss = [np.average(to_plot['tr_loss'][i]) for i in range(max_epochs+1)]
    te_loss = [np.average(to_plot['te_loss'][i]) for i in range(max_epochs+1)]
    tr_defu = [np.average(to_plot['tr_defu'][i]) for i in range(max_epochs+1)]
    te_defu = [np.average(to_plot['te_defu'][i]) for i in range(max_epochs+1)]

    # STD
    tr_loss_std = [np.std(to_plot['tr_loss'][i]) for i in range(max_epochs+1)]
    te_loss_std = [np.std(to_plot['te_loss'][i]) for i in range(max_epochs+1)]
    tr_defu_std = [np.std(to_plot['tr_defu'][i]) for i in range(max_epochs+1)]
    te_defu_std = [np.std(to_plot['te_defu'][i]) for i in range(max_epochs+1)]
    x=range(max_epochs+1)
    
    #return (to_plot,x,max_epochs+1)
    return (tr_loss, te_loss, tr_defu, te_defu, x), (tr_loss_std, te_loss_std, tr_defu_std, te_defu_std, x)

def generateLineChart(xy_list, filename):
    
    fig, axs = plt.subplots(1, len(xy_list), figsize=(12,6))

    for i in range(len(xy_list)):
        x, ys, labels, title, ytitle = xy_list[i]
        for y, label in zip(ys, labels):
            print(len(y), label)
            axs[i].plot(x, y, label=label, markersize=1)
        
        if i in [2,3]:
            start = 10
            bottom = min(min(ys[0][start:]), min(ys[1][start:]), min(ys[2][start:]))
            top    = max(max(ys[0][start:]), max(ys[1][start:]), max(ys[2][start:]), max(ys[3][start:]))
            axs[i].set_ylim(bottom, top)
        if i == 0:
            axs[i].legend()

    # plt.autoscale()
    plt.savefig(filename)
    plt.show()

if __name__=='__main__':
    
    labels = ['two-stage','block-decision-focused', 'corrected-block-decision-focused', 'hybrid']
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--filename', type=str, help='filename under folder results')
    parser.add_argument('--prob', type=float, default=0.2, help='input the probability used as input of random graph generator')
    parser.add_argument('--noise', type=float, default=0, help='noise level of the normalized features (in variance)')
    parser.add_argument('--budget', type=float, default=1, help='number of the defender budget')
    parser.add_argument('--cut-size', type=float, default=0.5, help='cut size')
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

    #f="./results/0808normto10_linearphi_dist_tenruns_fullINIT_decision-focused_n20_p0.3_b2.0_global - Copy.csv"
    tr_loss, te_loss = [[]] * len(labels), [[]] * len(labels)
    tr_defu, te_defu = [[]] * len(labels), [[]] * len(labels)
    for i, label in enumerate(labels):
        filepath = "results/random/{}_{}_n{}_p{}_b{}_cut{}_noise{}.csv".format(filename, label, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, CUT_SIZE, NOISE_LEVEL)
        print(label)
        tr_loss[i], te_loss[i], tr_defu[i], te_defu[i], x1 = return_yaxis(filepath)
    
    xy_list = []
    xy_list.append((x1, np.array(tr_loss), labels, "Training Loss", 'KL Divergence'))
    xy_list.append((x1, np.array(te_loss), labels, "Testing Loss",  'KL Divergence'))
    xy_list.append((x1, np.array(tr_defu), labels, "Training Defender Utility", 'Defender utility'))
    xy_list.append((x1, np.array(te_defu), labels, "Testing Defender Utility",  'Defender utility'))

    save_filename = "{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generateLineChart(xy_list, save_filename)
    

    
    #print(len(to_plot['tr_loss'][0]))
