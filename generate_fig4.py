# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:06:15 2019

@author: Aditya
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse

def return_yaxis(filename, method):
    f = open(filename, 'r')
    max_epochs = 200
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
            if epoch>-1:
                if item[0]=='training':
                    to_plot['tr_loss'][epoch].append(float(item[2]))
                    to_plot['tr_defu'][epoch].append(float(item[3]))
                
                elif item[0]=='validating':
                    to_plot['val_loss'][epoch].append(float(item[2]))
                    to_plot['val_defu'][epoch].append(float(item[3]))

                elif item[0]=='testing':
                    to_plot['te_loss'][epoch].append(float(item[2]))
                    to_plot['te_defu'][epoch].append(float(item[3]))

    for key in to_plot:
        to_plot[key] = np.array(to_plot[key])
    _, num_samples = to_plot['tr_loss'].shape

    final_loss_list = []
    final_defu_list = []
    if method == 'two-stage':
        for i in range(num_samples):
            tmp_idx = np.argmin(to_plot['val_loss'][:,i])
            tmp_loss = to_plot['te_loss'][tmp_idx,i]
            tmp_defu = to_plot['te_defu'][tmp_idx,i]
            final_loss_list.append(tmp_loss)
            final_defu_list.append(tmp_defu)
    elif method == 'decision-focused' or method == 'hybrid':
        for i in range(num_samples):
            tmp_idx = np.argmax(to_plot['val_defu'][:,i])
            tmp_loss = to_plot['te_loss'][tmp_idx,i]
            tmp_defu = to_plot['te_defu'][tmp_idx,i]
            final_loss_list.append(tmp_loss)
            final_defu_list.append(tmp_defu)

    final_loss = np.mean(final_loss_list)
    final_defu = np.mean(final_defu_list)
        
    tr_loss=[np.average(to_plot['tr_loss'][i]) for i in range(max_epochs+1)]
    te_loss=[np.average(to_plot['te_loss'][i]) for i in range(max_epochs+1)]
    tr_defu=[np.average(to_plot['tr_defu'][i]) for i in range(max_epochs+1)]
    te_defu=[np.average(to_plot['te_defu'][i]) for i in range(max_epochs+1)]
    x=range(max_epochs+1)
    
    #return (to_plot,x,max_epochs+1)
    return (tr_loss, te_loss, tr_defu, te_defu, final_loss, final_defu, x)

def generatePlot(xy_list, filename):
    
    fig, axs = plt.subplots(1, len(xy_list))

    #ax1 = fig.add_subplot(221)
    #ax2 = fig.add_subplot(222)
    #ax3 = fig.add_subplot(223)
    #ax4 = fig.add_subplot(224)
    for i in range(len(xy_list)):
        x, y1, label1, y2, label2, y3, label3, title, ytitle = xy_list[i]
        axs[i].plot(x, y1,'bo-' ,label=label1, markersize=1)
        axs[i].plot(x, y2, 'gs-' ,label=label2, markersize=1)
        axs[i].plot(x, y3, 'r-' ,label=label3, markersize=1)
        
        axs[i].legend()
        # axs[i].title(title)
        # axs[i].xlabel('Epochs')
        # axs[i].ylabel(ytitle)

    #epochs = len(train_loss) - 1
    #x=range(-1, epochs)
    plt.autoscale()
    plt.savefig("./results/excel/comparison/"+filename)
    plt.show()

if __name__=='__main__':
    
    l=['DF-block','2S', 'Hybrid']
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
    tr_loss1, te_loss1, tr_defu1, te_defu1, final_loss1, final_defu1, x1 = return_yaxis(file1, 'decision-focused')
    tr_loss2, te_loss2, tr_defu2, te_defu2, final_loss2, final_defu2, x2 = return_yaxis(file2, 'two-stage')
    tr_loss3, te_loss3, tr_defu3, te_defu3, final_loss3, final_defu3, x3 = return_yaxis(file3, 'hybrid')
    
    # generatePlot(x1,tr_loss1,l[0],tr_loss2,l[1],tr_loss3,l[2],tr_loss4,l[3], title="Training Loss", ytitle='KL Divergence')
    # generatePlot(x1,te_loss1,l[0],te_loss2,l[1],te_loss3,l[2],te_loss4,l[3], title="Testing Loss", ytitle='KL Divergence')
    # generatePlot(x1,tr_defu1,l[0],tr_defu2,l[1],tr_defu3,l[2],tr_defu4,l[3],title="Training Defender Utility", ytitle='Defender utility')
    # generatePlot(x1,te_defu1,l[0],te_defu2,l[1],te_defu3,l[2],te_defu4,l[3],title="Testing Defender Utility", ytitle='Defender utility')
    
    xy_list = []
    xy_list.append((x1, tr_loss1, l[0], tr_loss2, l[1], tr_loss3, l[2], "Training Loss", 'KL Divergence'))
    xy_list.append((x1, te_loss1, l[0], te_loss2, l[1], te_loss3, l[2], "Testing Loss",  'KL Divergence'))
    xy_list.append((x1, tr_defu1, l[0], tr_defu2, l[1], tr_defu3, l[2], "Training Defender Utility", 'Defender utility'))
    xy_list.append((x1, te_defu1, l[0], te_defu2, l[1], te_defu3, l[2], "Testing Defender Utility",  'Defender utility'))

    save_filename = "{}_n{}_p{}_b{}_noise{}.png".format(filename, GRAPH_N_LOW, GRAPH_E_PROB_LOW, DEFENDER_BUDGET, NOISE_LEVEL)
    generatePlot(xy_list, save_filename)
    

    
    #print(len(to_plot['tr_loss'][0]))
