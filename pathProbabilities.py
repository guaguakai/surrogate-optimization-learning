# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:49:05 2019

@author: Aditya
"""

from gcn import * 
import torch
import torch.optim as optim


A=torch.rand(10,10)
x=torch.rand(10,25)


net1= GCNDataGenerationNet(A, 25)
y=net1.forward(x).view(1,-1)
#print (y.size())
#print("Y:", y)


net2= GCNPredictionNet(A, 25)
optimizer=optim.SGD(net2.parameters(), lr=0.1)
#out=net2(x).view(1,-1)
#print("out:", out)
#print(out.size())
#print (len(list(net2.parameters())))
#print (list(net2.parameters())[5].size())
#loss=nn.MSELoss()
#print (loss(out, y))

for i in range(400):
    optimizer.zero_grad()
    out=net2(x).view(1,-1)
    loss_function=nn.MSELoss()
    loss=loss_function(out,y)
    print("Loss: ", loss)
    #net2.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
#diff=out-net2.forward(x)
#print (diff)