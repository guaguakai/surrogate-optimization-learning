# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:25:29 2019

@author: Aditya
"""
from scipy.optimize import minimize 
import networkx as nx
import numpy as np
from numpy.linalg import *
from graphData import *
import torch

def objective_function(coverage_probs,G, phi, U, initial_distribution, omega=4):
    
    # Pre-compute the following parameters: Graph edges, targets, transient nodes
    N=nx.number_of_nodes(G)
    edges= list(G.edges)
    targets=G.graph['targets']
    transients=[node for node in list(G.nodes()) if not (node in targets)]
    n_targets=len(targets)      # Does NOT include the 'caught' state
    n_transients=N-n_targets              
    
    # Compute the objective function    
    R=np.zeros((n_transients, n_targets+1))
    Q=np.zeros((n_transients,n_transients))
        
    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=np.zeros((N,N))
    for i, e in enumerate(list(G.edges())):
        #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i]
        
    # EDGE TRANSITION PROBABILITY MATRIX     
    edge_probs=np.zeros((N,N))
    for i, u in enumerate(list(G.nodes())):
        neighbors=list(nx.all_neighbors(G,u))
        # Compute apriori neighbor transition probs (without considering coverage)
        neighbor_transition_probs=np.zeros(len(neighbors))
        for j,neighbor in enumerate(neighbors):
            edge=(u, neighbor)
            #pe= G.edge[node][neighbor]['coverage_prob']
            pe=coverage_prob_matrix[u][neighbor]
            neighbor_transition_probs[j]=np.exp(-omega*pe+phi[neighbor])            
        neighbor_transition_probs=neighbor_transition_probs*1.0/np.sum(neighbor_transition_probs)    
        for j,neighbor in enumerate(neighbors):
            edge_probs[u,neighbor]=neighbor_transition_probs[j]
        
        # POPULATE THE Q AND R MATRICES CORRECTLY 
        if u in transients:    
                        
            # Add the probability for the 'caught node'
            caught_probability=0.0
            for v in neighbors:    
                if v in transients:
                    # Add this entry to the Q matrix
                    Q[transients.index(u)][transients.index(v)]=edge_probs[u][v]*(1.0-coverage_prob_matrix[u][v])
                    caught_probability+=edge_probs[u][v]*(coverage_prob_matrix[u][v])
                else:
                    # Add this entry to the R matrix
                    R[transients.index(u)][targets.index(v)]= edge_probs[u][v]*(1.0-coverage_prob_matrix[u][v])
                    caught_probability+=edge_probs[u][v]*(coverage_prob_matrix[u][v])
            R[transients.index(u)][-1]=caught_probability        
    
    # Q and R are computed by now
    #print ("Not singular")
    N_matrix= np.linalg.inv((np.identity(Q.shape[0])-Q))
    B= np.matmul(N_matrix, R)
    obj=np.matmul(np.matmul(initial_distribution, B),U)
    return obj


def get_optimal_coverage_prob(G, phi, U, initial_distribution, budget, omega=4):
    """
    Inputs: 
        G is the graph object with dictionaries G[node], G[graph] etc. 
        phi is the attractiveness function, for each of the N nodes, phi(v, Fv)
        
    """    
    N=nx.number_of_nodes(G)
    # Randomly initialize coverage probability distribution
    initial_coverage_prob=np.random.rand(nx.number_of_edges(G))
    
    # Bounds and constraints
    bounds=[(0.0,1.0) for item in initial_coverage_prob]
    constraints=[{'type':'ineq','fun':lambda x: budget-sum(x)}]
    
    # Optimization step
    coverage_prob_optimal= minimize(objective_function_matrix_form,initial_coverage_prob,args=(G, phi, torch.Tensor(U), torch.Tensor(initial_distribution), omega, np), method='SLSQP', jac=dobj_dx_matrix_form, bounds=bounds, constraints=constraints)
    # coverage_prob_optimal= minimize(objective_function,initial_coverage_prob,args=(G, phi, U,initial_distribution,omega), method='SLSQP', bounds=bounds, constraints=constraints)
    
    return coverage_prob_optimal

def objective_function_matrix_form(coverage_probs, G, phi, U, initial_distribution, omega=4, lib=torch):
    n = len(G.nodes)
    targets = G.graph["targets"] + [n] # adding the caught node
    transient_vector = torch.Tensor([0 if v in targets else 1 for v in range(n+1)]).type(torch.uint8)

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        # coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i]

    adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
    exponential_term = torch.exp(- omega * coverage_prob_matrix) * torch.exp(phi) * adj
    marginal_prob = exponential_term / torch.sum(exponential_term, keepdim=True, dim=1)
    marginal_prob[torch.isnan(marginal_prob)] = 0

    state_prob = marginal_prob * (1 - coverage_prob_matrix)
    caught_prob = torch.sum(marginal_prob * coverage_prob_matrix, keepdim=True, dim=1)
    full_prob = torch.cat((state_prob, caught_prob), dim=1)
    Q = full_prob[transient_vector[:-1]][:,transient_vector]
    R = full_prob[transient_vector[:-1]][:,1 - transient_vector]
    N = (torch.eye(Q.shape[0]) - Q).inverse()
    B = N @ R
    obj = torch.Tensor(initial_distribution) @ B @ torch.Tensor(U)

    if lib == np:
        obj = obj.detach().numpy()

    return obj


def dobj_dx_matrix_form(coverage_probs, G, phi, U, initial_distribution, omega=4, lib=torch):
    n = len(G.nodes)
    targets = G.graph["targets"] + [n] # adding the caught node
    transient_vector = torch.Tensor([0 if v in targets else 1 for v in range(n+1)]).type(torch.uint8)

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        # G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        # coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i]

    adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
    exponential_term = torch.exp(- omega * coverage_prob_matrix) * torch.exp(phi) * adj
    marginal_prob = exponential_term / torch.sum(exponential_term, keepdim=True, dim=1)
    marginal_prob[torch.isnan(marginal_prob)] = 0

    state_prob = marginal_prob * (1 - coverage_prob_matrix)
    caught_prob = torch.sum(marginal_prob * coverage_prob_matrix, keepdim=True, dim=1)
    full_prob = torch.cat((state_prob, caught_prob), dim=1)
    Q = full_prob[transient_vector[:-1]][:,transient_vector]
    R = full_prob[transient_vector[:-1]][:,1 - transient_vector]
    N = (torch.eye(Q.shape[0]) - Q).inverse()
    B = N @ R

    dP_dx = torch.zeros((n,n,len(coverage_probs)))
    dstate_dx = torch.zeros((n,n,len(coverage_probs)))
    for i, edge_i in enumerate(list(G.edges)):
        (u, v) = edge_i
        for j, edge_j in enumerate(list(G.edges)):
            if edge_j[0] != u: # only proceed when edge_j = (u,w)
                continue
            (_, w) = edge_j
            if v == w:
                dP_dx[u,v,j] = omega * (-1 + marginal_prob[u,v])
                dstate_dx[u,v,j] = omega * (1 - coverage_prob_matrix[u,v]) * (-1 + marginal_prob[u,v]) - 1
            else:
                dP_dx[u,v,j] = omega * marginal_prob[u, w]
                dstate_dx[u,v,j] = omega * (1 - coverage_prob_matrix[u,v]) * marginal_prob[u, w]
        dP_dx[u,v,:] *= marginal_prob[u,v]
        dstate_dx[u,v,:] *= marginal_prob[u,v]

    dcaught_dx = -torch.sum(dstate_dx, keepdim=True, dim=1)
    dfull_dx = torch.cat((dstate_dx, dcaught_dx), dim=1)
    dQ_dx = dfull_dx[transient_vector[:-1]][:,transient_vector,:]
    dR_dx = dfull_dx[transient_vector[:-1]][:,1-transient_vector,:]
    # NdQ_dx = torch.einsum("ab,bcd->acd", N, dQ_dx)
    # NdQ_dxNR = torch.einsum("abc,bd->adc", NdQ_dx, (N @ R))
    # NdR_dx = torch.einsum("ab,bcd->acd", N, dR_dx)
    # dB_dx = NdQ_dxNR + NdR_dx  # TODO!! ERROR
    # dobj_dx = U @ torch.einsum("a,abc->bc", initial_distribution, dB_dx)

    distN = initial_distribution @ N
    distNdQ_dxNRU = distN @ torch.einsum("abc,b->ac", dQ_dx, (N @ (R @ U)))
    distNdR_dxU = distN @ (torch.einsum("abc,b->ac", dR_dx, U))
    dobj_dx = distNdQ_dxNRU + distNdR_dxU

    if lib == np:
        dobj_dx = dobj_dx.detach().numpy()

    return dobj_dx




if __name__=='__main__':
    
    # CODE BLOCK FOR GENERATING G, U, INITIAL_DISTRIBUTION, BUDGET
    G=returnGraph(fixed_graph=True)
    E=nx.number_of_edges(G)
    N=nx.number_of_nodes(G)
    nodes=list(G.nodes())
    sources=list(G.graph['sources'])
    targets=list(G.graph['targets'])
    transients=[node for node in nodes if not (node in G.graph['targets'])]
    initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
    
    U=[G.node[t]['utility'] for t in targets]
    U.append(-20)
    print ('U:', U)
    U=np.array(U)
    budget=0.5*E
    
    
    # CODE BLOCK FOR GENERATING PHI (GROUND TRUTH PHI GENERATED FOR NOW)
    node_feature_size=25
    net1= GCNDataGenerationNet(node_feature_size)     
    # Define node features for each of the n nodes
    for node in list(G.nodes()):
        node_features=np.random.randn(node_feature_size)
        G.node[node]['node_features']=node_features
    Fv=np.zeros((N,node_feature_size))
    for node in list(G.nodes()):
        Fv[node]=G.node[node]['node_features']
    Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
    # Generate attractiveness values for nodes
    A=nx.to_numpy_matrix(G)
    A_torch=torch.as_tensor(A, dtype=torch.float) 
    phi=(net1.forward(Fv_torch,A_torch).view(-1)).detach().numpy()
    

    optimal_coverage_probs=get_optimal_coverage_prob(G, phi, U, initial_distribution, budget)
    print ("Optimal coverage:\n", optimal_coverage_probs)
    print("Budget: ", budget)
    print ("Sum of coverage probabilities: ", sum(optimal_coverage_probs['x']))
