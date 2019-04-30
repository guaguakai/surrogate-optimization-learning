# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:25:29 2019

@author: Aditya
"""
from scipy.optimize import minimize
import networkx as nx
import numpy as np
from numpy.linalg import *

def get_optimal_coverage_prob(G, phi, Fv, U):
    """
    Inputs: 
        G is the graph object with dictionaries G[node], G[graph] etc. 
        phi is the attractiveness function, for each of the N nodes, phi(v, Fv)
        Fv is the feature matrix of size N X feature_size
        
    """
    # initial_coverage_probs=#TO_DO
    # Randomly initialize this to some probability distribution
    initial_coverage_prob=np.random.rand(nx.number_of_edges(G))
    
    def objective_function(coverage_probs, args=(G, phi, Fv, U)):
        # TODO: Compute the objective value in terms of defender's coverage probability
        # TODO: Compute edge transition oprobabilities from initial_coverage_prob
        # compute P and Q 
        
        N= inv((np.identity(Q.shape[0])-Q))
        B= np.matmul(N, R)
        obj= np.matmul(np.matmul(initial_distribution, B),U)
        
        return obj
    
    
    
    coverage_prob_optimal= minimze(objective_function, initial_coverage_prob)        
    #TODO: Include constraint, include solver type




if __name__=='__main__':
    