# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:25:29 2019

@author: Aditya
"""
from scipy.optimize import minimize 
import networkx as nx
# import numpy as np
import autograd.numpy as np
from numpy.linalg import *
from graphData import *
import torch
import autograd

def phi2prob(G, phi): # unbiased but no need to be normalized. It will be normalized later
    N=nx.number_of_nodes(G)
    adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
    unbiased_probs = adj * torch.exp(phi)

    unbiased_probs = unbiased_probs / torch.sum(unbiased_probs, keepdim=True, dim=1)
    unbiased_probs[torch.isnan(unbiased_probs)] = 0

    return unbiased_probs



def prob2unbiased(G, coverage_probs, biased_probs, omega): # no need to be normalized. It will be normalized later
    N=nx.number_of_nodes(G)
    coverage_prob_matrix=torch.zeros((N,N))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

    unbiased_probs = biased_probs * torch.exp(coverage_prob_matrix * omega) # removing the effect of coverage
    unbiased_probs = unbiased_probs / torch.sum(unbiased_probs, keepdim=True, dim=1)
    unbiased_probs[torch.isnan(unbiased_probs)] = 0

    return unbiased_probs

def get_optimal_coverage_prob(G, unbiased_probs, U, initial_distribution, budget, omega=4, options={}, method='SLSQP', initial_coverage_prob=None):
    """
    Inputs: 
        G is the graph object with dictionaries G[node], G[graph] etc. 
        phi is the attractiveness function, for each of the N nodes, phi(v, Fv)
    """    
    N=nx.number_of_nodes(G)
    E=nx.number_of_edges(G)
    # Randomly initialize coverage probability distribution
    if initial_coverage_prob is None:
        initial_coverage_prob=np.random.rand(nx.number_of_edges(G))
        initial_coverage_prob=budget*(initial_coverage_prob/np.sum(initial_coverage_prob))
        # initial_coverage_prob = np.zeros(nx.number_of_edges(G))
    
    # Bounds and constraints
    bounds=[(0.0,1.0) for item in initial_coverage_prob]
    ineq_fn = lambda x: budget - sum(x)
    constraints=[{'type':'ineq','fun': ineq_fn, 'jac': autograd.jacobian(ineq_fn)}]
    
    # Optimization step
    coverage_prob_optimal= minimize(objective_function_matrix_form,initial_coverage_prob,args=(G, unbiased_probs, torch.Tensor(U), torch.Tensor(initial_distribution), omega, np), method=method, jac=dobj_dx_matrix_form, bounds=bounds, constraints=constraints, options=options)
    
    return coverage_prob_optimal

def objective_function_matrix_form(coverage_probs, G, unbiased_probs, U, initial_distribution, omega=4, lib=torch):
    n = len(G.nodes)
    targets = G.graph["targets"] + [n] # adding the caught node
    transient_vector = torch.Tensor([0 if v in targets else 1 for v in range(n+1)]).type(torch.uint8)

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

    adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
    exponential_term = torch.exp(- omega * coverage_prob_matrix) * unbiased_probs * adj
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

def dobj_dx_matrix_form(coverage_probs, G, unbiased_probs, U, initial_distribution, omega=4, lib=torch):
    n = len(G.nodes)
    targets = G.graph["targets"] + [n] # adding the caught node
    transient_vector = torch.Tensor([0 if v in targets else 1 for v in range(n+1)]).type(torch.uint8)

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only


    adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
    exponential_term = torch.exp(- omega * coverage_prob_matrix) * unbiased_probs * adj
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
        for (u, v) in [(edge_i[0], edge_i[1]), (edge_i[1], edge_i[0])]:
            for j, edge_j in enumerate(list(G.edges)): 
                if edge_j[0] == u: # only proceed when edge_j = (u,w)
                    (_, w) = edge_j
                elif edge_j[1] == u:
                    (w, _) = edge_j
                else:
                    continue

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

def dobj_dx_matrix_form_np(coverage_probs, G, unbiased_probs, U, initial_distribution, omega=4):
    n = len(G.nodes)
    targets = G.graph["targets"] + [n] # adding the caught node
    transient_vector = np.array([0 if v in targets else 1 for v in range(n+1)])

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=np.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        # coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only


    adj = nx.adjacency_matrix(G).toarray()
    exponential_term = np.exp(- omega * coverage_prob_matrix) * unbiased_probs * adj
    marginal_prob = exponential_term / np.sum(exponential_term, keepdims=True, axis=1)
    marginal_prob[np.isnan(marginal_prob)] = 0

    state_prob = marginal_prob * (1 - coverage_prob_matrix)
    caught_prob = np.sum(marginal_prob * coverage_prob_matrix, keepdims=True, axis=1)
    full_prob = np.concatenate((state_prob, caught_prob), axis=1)
    Q = full_prob[transient_vector[:-1] == 1][:, transient_vector == 1]
    R = full_prob[transient_vector[:-1] == 1][:, transient_vector == 0]
    N = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    B = N @ R

    dP_dx = np.zeros((n,n,len(coverage_probs)))
    dstate_dx = np.zeros((n,n,len(coverage_probs)))

    for i, edge_i in enumerate(list(G.edges)):
        for (u, v) in [(edge_i[0], edge_i[1]), (edge_i[1], edge_i[0])]:
            for j, edge_j in enumerate(list(G.edges)): 
                if edge_j[0] == u: # only proceed when edge_j = (u,w)
                    (_, w) = edge_j
                elif edge_j[1] == u:
                    (w, _) = edge_j
                else:
                    continue

                if v == w:
                    dP_dx[u,v,j] = omega * (-1 + marginal_prob[u,v])
                    dstate_dx[u,v,j] = omega * (1 - coverage_prob_matrix[u,v]) * (-1 + marginal_prob[u,v]) - 1 
                else:
                    dP_dx[u,v,j] = omega * marginal_prob[u, w]
                    dstate_dx[u,v,j] = omega * (1 - coverage_prob_matrix[u,v]) * marginal_prob[u, w]

            dP_dx[u,v,:] *= marginal_prob[u,v]
            dstate_dx[u,v,:] *= marginal_prob[u,v]

    dcaught_dx = -np.sum(dstate_dx, keepdims=True, axis=1)
    dfull_dx = np.concatenate((dstate_dx, dcaught_dx), axis=1)
    dQ_dx = dfull_dx[transient_vector[:-1] == 1][:,transient_vector == 1,:]
    dR_dx = dfull_dx[transient_vector[:-1] == 1][:,transient_vector == 0,:]

    distN = initial_distribution @ N
    distNdQ_dxNRU = distN @ np.einsum("abc,b->ac", dQ_dx, (N @ (R @ U)))
    distNdR_dxU = distN @ (np.einsum("abc,b->ac", dR_dx, U))
    dobj_dx = distNdQ_dxNRU + distNdR_dxU

    return dobj_dx

def obj_hessian_matrix_form(coverage_probs, G, unbiased_probs, U, initial_distribution, omega=4, lib=torch):
    x = torch.autograd.Variable(coverage_probs.detach(), requires_grad=True)
    dobj_dx = dobj_dx_matrix_form(torch.Tensor(x), G, unbiased_probs, U, initial_distribution, omega=omega, lib=torch)
    m = len(x)
    obj_hessian = torch.zeros((m,m))
    for i in range(len(x)):
        obj_hessian[i] = torch.autograd.grad(dobj_dx[i], x, create_graph=False, retain_graph=True)[0]

    return obj_hessian


def obj_hessian_matrix_form_np(coverage_probs, G, unbiased_probs, U, initial_distribution, omega=4):
    def first_derivative(x):
        return dobj_dx_matrix_form_np(x, G, unbiased_probs, U, initial_distribution, omega=omega)

    obj_hessian = autograd.jacobian(first_derivative)(coverage_probs) # NOT WORKING...

    return obj_hessian


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
    phi=(net1.forward(Fv_torch,A_torch).view(-1)).detach()
    unbiased_probs = phi2prob(G, phi)
    

    optimal_coverage_probs=get_optimal_coverage_prob(G, unbiased_probs, U, initial_distribution, budget)
    print ("Optimal coverage:\n", optimal_coverage_probs)
    print("Budget: ", budget)
    print ("Sum of coverage probabilities: ", sum(optimal_coverage_probs['x']))
