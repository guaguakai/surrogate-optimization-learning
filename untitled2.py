# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:48:04 2019

@author: Aditya
"""
from scipy.optimize import minimize

def trial():
    initial_a= [10,10]
    b=2
    c=3
    d=1
    def f (a, args=(b)):
        
        x=-(a[0]*b+a[1]*c+b*c+d)
        return x 
    budget=1.2
    bounds=[(0.0,1.0),(0.0,1.0)]
    constraints=[{'type':'ineq','fun':lambda x: budget-sum(x)}]
    final_a=minimize(f, initial_a, method='SLSQP', bounds=bounds, constraints=constraints)
    print (final_a)
    return 

if __name__=='__main__':
    trial()