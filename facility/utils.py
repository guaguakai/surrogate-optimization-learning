from scipy.optimize import minimize
import networkx as nx
import numpy as np
import time
# import autograd.numpy as np
from numpy.linalg import *
# from graphData import *
import torch
import autograd
from gurobipy import *

def normalize_matrix(T):
    return T / torch.norm(T, dim=0)

def normalize_matrix_positive(T):
    pos_T = torch.clamp(T, min=0)
    # pos_T = T
    return pos_T / torch.norm(pos_T, dim=0)

def normalize_matrix_qr(T):
    Q, R = np.linalg.qr(T.detach().numpy())
    return torch.Tensor(Q)

def normalize_vector(s, max_value=1):
    s = torch.clamp(s, min=0)
    s_sum = torch.sum(s)
    if s_sum > max_value:
        s = s / s_sum * max_value
    return s

def computeCovariance(covariance_mat):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n = len(covariance_mat)
    cosine_matrix = torch.zeros((n,n))
    for i in range(n):
        cosine_matrix[i] = cos(covariance_mat, covariance_mat[i].repeat(n,1))
    return cosine_matrix

def point_projection(x, T): # project x to the hyperplane x = {Ty}_y
    y = torch.pinverse(T) @ x
    newx = T @ y
    return newx

def projection(P, A, b): # P is the reparameterization matrix
    p_variable_size, batch_size = P.shape # reparameterization matrix
    constraint_size, variable_size = A.shape
    b_constraint_size, b_size = b.shape
    assert(p_variable_size == variable_size and constraint_size == b_constraint_size and b_size == 1)
    AA_up   = torch.cat((A, torch.zeros(constraint_size, constraint_size)), dim=1)
    AA_down = torch.cat((-torch.eye(variable_size), A.t()), dim=1)
    AA = torch.cat((AA_up, AA_down))
    bb = torch.cat((b[None,:,:].repeat(batch_size,1,1), -P.t()[:,:,None]), dim=1)
    Plambda, _ = torch.solve(bb, AA)
    newP = Plambda[:,:variable_size,0].t()
    return newP

def normalize_projection(P, A, b):
    newP = projection(P, A, b)
    return newP

def phi2prob(G, phi): # unbiased but no need to be normalized. It will be normalized later
    N=nx.number_of_nodes(G)
    adj = torch.Tensor(nx.adjacency_matrix(G, nodelist=range(N)).toarray())
    exponential_term = torch.exp(adj * phi) * adj
    row_sum = torch.sum(exponential_term, dim=1)
    unbiased_probs = torch.zeros_like(exponential_term)
    unbiased_probs[row_sum != 0] = exponential_term[row_sum != 0] / torch.sum(exponential_term, keepdim=True, dim=1)[row_sum != 0]
    return unbiased_probs

# def generate_EdgeProbs_from_Attractiveness(G, coverage_probs, phi, omega=4):
#     N=nx.number_of_nodes(G)
#     coverage_prob_matrix=torch.zeros((N,N))
#     for i, e in enumerate(list(G.edges())):
#         #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
#         coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
#         coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only
#
#     # GENERATE EDGE PROBABILITIES
#     adj = torch.Tensor(nx.adjacency_matrix(G, nodelist=range(N)).toarray())
#     exponential_term = torch.exp(adj * phi - omega * coverage_prob_matrix) * adj
#     row_sum = torch.sum(exponential_term, dim=1)
#     transition_probs = torch.zeros_like(exponential_term)
#     transition_probs[row_sum != 0] = exponential_term[row_sum != 0] / torch.sum(exponential_term, keepdim=True, dim=1)[row_sum != 0]
#
#     # adj_phi = adj * phi - omega * coverage_prob_matrix - (1 - adj) * 100 # adding -100 to all the non-adjacent entries
#     # transition_probs = torch.nn.Softmax(dim=1)(adj_phi)
#
#     return transition_probs

def prob2unbiased(G, coverage_probs, biased_probs, omega): # no need to be normalized. It will be normalized later
    N=nx.number_of_nodes(G)
    coverage_prob_matrix=torch.zeros((N,N))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

    # adj = torch.Tensor(nx.adjacency_matrix(G, nodelist=range(N)).toarray())
    exponential_term = biased_probs * torch.exp(coverage_prob_matrix * omega) # removing the effect of coverage
    row_sum = torch.sum(exponential_term, dim=1)
    unbiased_probs = torch.zeros_like(exponential_term)
    unbiased_probs[row_sum != 0] = exponential_term[row_sum != 0] / torch.sum(exponential_term, keepdim=True, dim=1)[row_sum != 0] # + MEAN_REG

    return unbiased_probs
