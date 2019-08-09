# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 01:06:15 2019

@author: Aditya
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

file="./results/random/0728adi_decision-focused_n20_p0.3_b2.0_global.csv"

def return_yaxis(file):
    df=pd.read_csv(file)
    
    columns=df.columns
    matrix=df.as_matrix()
    
    unique_epochs=[int(item) for item in (list(set(matrix[:,-4]))) if len(str(item))<=3]
    #print (set(matrix[:,-4]))
    print (unique_epochs)
    max_epochs=max(unique_epochs)
    print (max_epochs)
    to_plot={'tr_loss':[[] for _ in range(max_epochs+1)],
            'te_loss':[[] for _ in range(max_epochs+1)],
            'tr_defu':[[] for _ in range(max_epochs+1)],
            'te_defu':[[] for _ in range(max_epochs+1)]}
    
    #d={'lol':[[] for i in range(max_epochs+1)]}
    
    for item in matrix:
        print (item)
        if len(str(item[-4]))<=3:
            epoch=int(item[-4])
            if epoch>-1:
                if item[0]=='training':
                    to_plot['tr_loss'][epoch].append(float(item[2]))
                    to_plot['tr_defu'][epoch].append(float(item[3]))
                
                elif item[0]=='testing':
                    to_plot['te_loss'][epoch].append(float(item[2]))
                    to_plot['te_defu'][epoch].append(float(item[3]))
        
    tr_loss=[np.average(to_plot['tr_loss'][i]) for i in range(max_epochs+1)]
    te_loss=[np.average(to_plot['te_loss'][i]) for i in range(max_epochs+1)]
    tr_defu=[np.average(to_plot['tr_defu'][i]) for i in range(max_epochs+1)]
    te_defu=[np.average(to_plot['te_defu'][i]) for i in range(max_epochs+1)]
    x=range(max_epochs+1)
    
    return (to_plot,x,max_epochs+1)
    return (tr_loss, te_loss, tr_defu, te_defu, x)

def generatePlot(x,y1,label1, y2,label2, y3,label3, y4,label4, title="", ytitle=""):
    
    fig = plt.figure()
    #ax1 = fig.add_subplot(221)
    #ax2 = fig.add_subplot(222)
    #ax3 = fig.add_subplot(223)
    #ax4 = fig.add_subplot(224)
    plt.plot(x, y1,'bo-' ,label=label1)
    plt.plot(x, y2, 'gs-' ,label=label2)
    plt.plot(x, y3, 'r^-',label=label3)
    plt.plot(x, y4, 'yD-',label=label4)
    
    print ("x",x)
    print (label1, y1)
    print (label2, y2)
    print (label3, y3)
    print (label4, y4)
    plt.legend()
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ytitle)
    #fig.text(0.5, 0.04, '# epochs', ha='center', va='center')
    #fig.text(0.06, 0.75, 'KL-divergence loss', ha='center', va='center', rotation='vertical')
    #fig.text(0.06, 0.25, 'Defender utility', ha='center', va='center', rotation='vertical')

    #epochs = len(train_loss) - 1
    #x=range(-1, epochs)
    plt.savefig("./results/excel/comparison/"+title)
    plt.show()

if __name__=='__main__':
    
    l=['DF-full','DF-fast','2S-full','2S-fast']
    f="./results/0808normto10_linearphi_dist_tenruns_fullINIT_decision-focused_n20_p0.3_b2.0_global - Copy.csv"
    #file1="./results/0808normto10_linearphi_dist_tenruns_fullINIT_decision-focused_n20_p0.3_b2.0_global.csv"
    #file2="./results/0808normto10_linearphi_dist_tenruns_fullINIT_decision-focused_n20_p0.3_b2.0_mincut.csv"
    #file3="./results/0808normto10_linearphi_dist_tenruns_fullINIT_two-stage_n20_p0.3_b2.0_global.csv"
    #file4="./results/0808normto10_linearphi_dist_tenruns_fullINIT_two-stage_n20_p0.3_b2.0_mincut.csv"
    
    to_plot,x,epochs,=return_yaxis(f)
    #tr_loss2, te_loss2, tr_defu2, te_defu2, x2=return_yaxis(file2)
    #tr_loss3, te_loss3, tr_defu3, te_defu3, x3=return_yaxis(file3)
    #tr_loss4, te_loss4, tr_defu4, te_defu4, x4=return_yaxis(file4)
    
    n=3
    
    plt.plot(x1,[to_plot['tr_loss'][i][n] for i in range(epochs)],label=l[0])
    plt.plot(x1,[to_plot['te_loss'][i][n] for i in range(epochs)],label=l[0])
    plt.plot(x1,[to_plot['tr_defu'][i][n] for i in range(epochs)],label=l[0])
    plt.plot(x1,[to_plot['te_defu'][i][n] for i in range(epochs)],label=l[0])
    
    #generatePlot(x1,tr_loss1,l[0],tr_loss2,l[1],tr_loss3,l[2],tr_loss4,l[3], title="Training Loss", ytitle='KL Divergence')
    #generatePlot(x1,te_loss1,l[0],te_loss2,l[1],te_loss3,l[2],te_loss4,l[3], title="Testing Loss", ytitle='KL Divergence')
    #generatePlot(x1,tr_defu1,l[0],tr_defu2,l[1],tr_defu3,l[2],tr_defu4,l[3],title="Training Defender Utility", ytitle='Defender utility')
    #generatePlot(x1,te_defu1,l[0],te_defu2,l[1],te_defu3,l[2],te_defu4,l[3],title="Testing Defender Utility", ytitle='Defender utility')
    
    
    

    
    #print(len(to_plot['tr_loss'][0]))