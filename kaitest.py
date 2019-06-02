from graphData import generate_EdgeProbs_from_Attractiveness
import networkx as nx
import numpy as np
import torch
import time

from graphData import generateSyntheticData, returnGraph
from coverageProbability import objective_function
from gcn import GCNDataGenerationNet

def objective_function_matrix_form(coverage_probs, G, phi, U, initial_distribution, omega=4):
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

    return obj

def dobj_dx_matrix_form(coverage_probs, G, phi, U, initial_distribution, omega=4):
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

    print(dstate_dx.shape)
    dcaught_dx = -torch.sum(dstate_dx, keepdim=True, dim=1)
    print(dcaught_dx.shape)
    dfull_dx = torch.cat((dstate_dx, dcaught_dx), dim=1)
    print(dfull_dx.shape)
    dQ_dx = dfull_dx[transient_vector[:-1]][:,transient_vector,:]
    dR_dx = dfull_dx[transient_vector[:-1]][:,1-transient_vector,:]
    print(dQ_dx.shape)
    print(dR_dx.shape)
    print(N.shape)
    NdQ_dx = torch.einsum("ab,bcd->acd", N, dQ_dx)
    print(NdQ_dx.shape)
    print((N@R).shape)
    NdQ_dxNR = torch.einsum("abc,bd->adc", NdQ_dx, (N @ R))
    NdR_dx = torch.einsum("ab,bcd->acd", N, dR_dx)
    print(NdQ_dxNR.shape)
    dB_dx = NdQ_dxNR + NdR_dx  # TODO!! ERROR
    print(dB_dx.shape)
    dobj_dx = U @ torch.einsum("a,abc->bc", initial_distribution, dB_dx)
    print(dobj_dx.shape)

    return dobj_dx


if __name__ == "__main__":

    # CODE BLOCK FOR GENERATING G, U, INITIAL_DISTRIBUTION, BUDGET
    G=returnGraph(fixed_graph=True)
    E=nx.number_of_edges(G)
    N=nx.number_of_nodes(G)
    nodes=list(G.nodes())
    sources=list(G.graph['sources'])
    targets=list(G.graph['targets'])
    transients=[node for node in nodes if not (node in G.graph['targets'])]
    initial_distribution=torch.Tensor([1.0/len(sources) if n in sources else 0.0 for n in transients])

    U=[G.node[t]['utility'] for t in targets]
    U.append(-20)
    print ('U:', U)
    U=torch.Tensor(U)
    budget=0.5*E

    omega = 4


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

    initial_coverage_prob = torch.rand(nx.number_of_edges(G), requires_grad=True)
    # initial_coverage_prob.retain_grad()
    coverage_probs = initial_coverage_prob

    print("Time testing...")
    count = 1
    start_time = time.time()
    for i in range(count):
        obj = objective_function(initial_coverage_prob.detach().numpy(), G, phi.numpy(), U.numpy(),initial_distribution.numpy(), omega)
    print(time.time() - start_time)

    start_time = time.time()
    for i in range(count):
        obj_matrix_form = objective_function_matrix_form(initial_coverage_prob, G, torch.Tensor(phi), torch.Tensor(U), torch.Tensor(initial_distribution))
    print(time.time() - start_time)

    # print("obj: {}\nobj matrix form: {}\n".format(obj, obj_matrix_form))

    # derivatives...
    dobj_dx = dobj_dx_matrix_form(initial_coverage_prob, G, phi, U, initial_distribution, omega)

    torch_dobj_dx = torch.autograd.grad(obj_matrix_form, initial_coverage_prob)
