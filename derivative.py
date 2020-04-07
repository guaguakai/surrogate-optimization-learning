# -*- coding: utf-8 -*-

from scipy.optimize import minimize 
import networkx as nx
import numpy as np
import time
# import autograd.numpy as np
from numpy.linalg import *
from graphData import *
import torch
import autograd
from gurobipy import *
from utils import phi2prob, prob2unbiased

REG = 0.01
MEAN_REG = 0.0

def get_optimal_coverage_prob(G, unbiased_probs, U, initial_distribution, budget, omega=4, options={}, method='SLSQP', initial_coverage_prob=None, tol=0.1):
    """
    Inputs: 
        G is the graph object with dictionaries G[node], G[graph] etc. 
        phi is the attractiveness function, for each of the N nodes, phi(v, Fv)
    """
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    # Randomly initialize coverage probability distribution
    if initial_coverage_prob is None:
        initial_coverage_prob=np.random.rand(m)
        initial_coverage_prob=budget*(initial_coverage_prob/np.sum(initial_coverage_prob))
    
    # Bounds and constraints
    bounds=[(0.0,1.0) for _ in range(m)]

    eq_fn = lambda x: budget - sum(x)
    constraints=[{'type': 'eq','fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}]
    edge_set = list(range(m))
    
    # Optimization step
    coverage_prob_optimal= minimize(objective_function_matrix_form,initial_coverage_prob,args=(G, unbiased_probs, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega, np), method=method, jac=dobj_dx_matrix_form, bounds=bounds, constraints=constraints, tol=tol, options=options)
    
    return coverage_prob_optimal

def get_optimal_coverage_prob_frank_wolfe(G, unbiased_probs, U, initial_distribution, budget, omega=4, num_iterations=100, initial_coverage_prob=None, tol=0.1):
    n=nx.number_of_nodes(G)
    m=nx.number_of_edges(G)

    # Randomly initialize coverage probability distribution
    if initial_coverage_prob is None:
        initial_coverage_prob = np.zeros(m)

    x = initial_coverage_prob
    edge_set = list(range(m))
    for k in range(num_iterations):
        gamma = 2 / (k + 2)
        dx = dobj_dx_matrix_form(x, G, unbiased_probs, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set, omega, np)

        model = Model()
        model.setParam('OutputFlag', False)
        coverage_prob = [model.addVar(lb=0.0, ub=1.0) for j in range(len(edge_set))]
        model.addConstr(sum(coverage_prob) <= budget)
        model.setObjective(np.dot(dx, coverage_prob))
        model.optimize()

        # print(model.ObjVal)
        s = np.array([var.x for var in coverage_prob])
        x = x + gamma * (s - x)
    
    return x

def objective_function_matrix_form(coverage_probs, G, unbiased_probs, U, initial_distribution, edge_set=None, omega=4, lib=torch): # edge_set is not used here
    n = len(G.nodes)
    m = len(G.nodes)
    targets = list(G.graph["targets"]) + [n] # adding the caught node
    transient_vector = list(set(range(n)) - set(targets))

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix = torch.zeros((n,n))
    edges = list(G.edges())
    for i, e in enumerate(edges):
        coverage_prob_matrix[e[0]][e[1]] = coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]] = coverage_probs[i] # for undirected graph only

    # adj = torch.Tensor(nx.adjacency_matrix(G, nodelist=range(n)).toarray())
    exponential_term = torch.exp(- omega * coverage_prob_matrix) * unbiased_probs # + MEAN_REG
    row_sum = torch.sum(exponential_term, dim=1)
    marginal_prob = torch.zeros_like(exponential_term)
    marginal_prob[row_sum != 0] = exponential_term[row_sum != 0] / torch.sum(exponential_term, keepdim=True, dim=1)[row_sum != 0]

    state_prob = marginal_prob * (1 - coverage_prob_matrix)
    caught_prob = 1 - torch.sum(state_prob, keepdim=True, dim=1)
    full_prob = torch.cat((state_prob, caught_prob), dim=1)

    Q = full_prob[transient_vector][:,transient_vector]
    R = full_prob[transient_vector][:,targets]

    N = (torch.eye(Q.shape[0]) * (1 + REG) - Q).inverse()
    B = N @ R
    obj = torch.Tensor(initial_distribution) @ B @ torch.Tensor(U)

    if lib == np:
        obj = obj.detach().numpy()

    return obj

def dobj_dx_matrix_form(coverage_probs, G, unbiased_probs, U, initial_distribution, edge_set, omega=4, lib=torch):
    n = len(G.nodes)
    m = len(G.edges)
    targets = list(G.graph["targets"]) + [n] # adding the caught node
    transient_vector = list(set(range(n)) - set(targets))

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    edges = list(G.edges())
    for i, e in enumerate(edges):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

    # adj = torch.Tensor(nx.adjacency_matrix(G, nodelist=range(n)).toarray())
    exponential_term = torch.exp(-omega * coverage_prob_matrix) * unbiased_probs # + MEAN_REG
    row_sum = torch.sum(exponential_term, dim=1)
    marginal_prob = torch.zeros_like(exponential_term)
    marginal_prob[row_sum != 0] = exponential_term[row_sum != 0] / torch.sum(exponential_term, keepdim=True, dim=1)[row_sum != 0]

    state_prob = marginal_prob * (1 - coverage_prob_matrix)
    caught_prob = torch.sum(marginal_prob * coverage_prob_matrix, keepdim=True, dim=1)
    full_prob = torch.cat((state_prob, caught_prob), dim=1)

    Q = full_prob[transient_vector][:,transient_vector]
    R = full_prob[transient_vector][:,targets]

    QQ = torch.eye(Q.shape[0]) * (1 + REG) - Q
    # N = (torch.eye(Q.shape[0]) * (1 + REG) - Q).inverse()
    # B = N @ R

    dstate_dx = torch.zeros((n,n,len(edge_set)))

    edges = list(G.edges())
    # =============== newer implementation of gradient computation ================ # speed up like 6 sec per instance
    for j, edge_j_idx in enumerate(edge_set):
        edge_j = edges[edge_j_idx]
        (v, w) = edge_j
        dstate_dx[v,list(G.neighbors(v)),j] = omega * (1 - coverage_prob_matrix[v,list(G.neighbors(v))]) * marginal_prob[v,w]
        dstate_dx[v,w,j] = dstate_dx[v,w,j] - omega * (1 - coverage_prob_matrix[v,w]) - 1

        dstate_dx[w,list(G.neighbors(w)),j] = omega * (1 - coverage_prob_matrix[w,list(G.neighbors(w))]) * marginal_prob[w,v]
        dstate_dx[w,v,j] = dstate_dx[w,v,j] - omega * (1 - coverage_prob_matrix[w,v]) - 1

    dstate_dx = torch.einsum('ij,ijk->ijk', marginal_prob, dstate_dx)

    dcaught_dx = -torch.sum(dstate_dx, keepdim=True, dim=1)
    dfull_dx = torch.cat((dstate_dx, dcaught_dx), dim=1)
    dQ_dx = dfull_dx[transient_vector][:,transient_vector,:]
    dR_dx = dfull_dx[transient_vector][:,targets,:]

    # distN = initial_distribution @ N
    distN, _ = torch.solve(initial_distribution[None,:,None], QQ.t()[None,:,:])
    distN = distN[0,:,0]
    # NRU = N @ (R @ U)
    NRU, _ = torch.solve((R @ torch.Tensor(U))[None,:,None], QQ[None,:,:])
    NRU = NRU[0,:,0]
    distNdQ_dxNRU = distN @ torch.einsum("abc,b->ac", dQ_dx, NRU)
    distNdR_dxU = distN @ (torch.einsum("abc,b->ac", dR_dx, U))
    dobj_dx = distNdQ_dxNRU + distNdR_dxU

    if lib == np:
        dobj_dx = dobj_dx.detach().numpy()

    return dobj_dx

def obj_hessian_matrix_form(coverage_probs, G, unbiased_probs, U, initial_distribution, edge_set, omega=4, lib=torch):
    if type(coverage_probs) == torch.Tensor:
        coverage_probs = coverage_probs.detach()
    else:
        coverage_probs = torch.Tensor(coverage_probs)

    full_coverage_probs = torch.Tensor(coverage_probs)

    x = torch.autograd.Variable(coverage_probs[list(edge_set)], requires_grad=True)
    full_coverage_probs[edge_set] = x
    dobj_dx = dobj_dx_matrix_form(torch.Tensor(full_coverage_probs), G, unbiased_probs, U, initial_distribution, edge_set, omega=omega, lib=torch)
    obj_hessian = torch.zeros((len(x),len(x)))
    for i in range(len(x)):
        if i < len(x) - 1:
            obj_hessian[i] = torch.autograd.grad(dobj_dx[i], x, create_graph=False, retain_graph=True)[0]
        else:
            obj_hessian[i] = torch.autograd.grad(dobj_dx[i], x, create_graph=False, retain_graph=False)[0]

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
