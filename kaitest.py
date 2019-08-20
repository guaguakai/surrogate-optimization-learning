from graphData import generate_EdgeProbs_from_Attractiveness
import networkx as nx
import numpy as np
import torch
import time
import autograd

from graphData import generateSyntheticData, returnGraph, generatePhi
from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, dobj_dx_matrix_form, obj_hessian_matrix_form, phi2prob

"""
def objective_function_matrix_form(coverage_probs, G, phi, U, initial_distribution, omega=4, lib=torch):
    n = len(G.nodes)
    targets = G.graph["targets"] + [n] # adding the caught node
    transient_vector = torch.Tensor([0 if v in targets else 1 for v in range(n+1)]).type(torch.bool)

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

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
    transient_vector = torch.Tensor([0 if v in targets else 1 for v in range(n+1)]).type(torch.bool)

    # COVERAGE PROBABILITY MATRIX
    coverage_prob_matrix=torch.zeros((n,n))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only


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
        for (u, v) in [(edge_i[0], edge_i[1]), (edge_i[1], edge_i[0])]:
            for j, edge_j in enumerate(list(G.edges)): # TODO!! change it to the undirected version
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

    print(dstate_dx.shape)
    dcaught_dx = -torch.sum(dstate_dx, keepdim=True, dim=1)
    print(dcaught_dx.shape)
    dfull_dx = torch.cat((dstate_dx, dcaught_dx), dim=1)
    print(dfull_dx.shape)
    dQ_dx = dfull_dx[transient_vector[:-1]][:,transient_vector,:]
    dR_dx = dfull_dx[transient_vector[:-1]][:,1-transient_vector,:]
    # print(dQ_dx.shape)
    # print(dR_dx.shape)
    # print(N.shape)
    # NdQ_dx = torch.einsum("ab,bcd->acd", N, dQ_dx)
    # print(NdQ_dx.shape)
    # print((N@R).shape)
    # NdQ_dxNR = torch.einsum("abc,bd->adc", NdQ_dx, (N @ R))
    # NdR_dx = torch.einsum("ab,bcd->acd", N, dR_dx)
    # print(NdQ_dxNR.shape)
    # dB_dx = NdQ_dxNR + NdR_dx  # TODO!! ERROR
    # print(dB_dx.shape)
    # dobj_dx = U @ torch.einsum("a,abc->bc", initial_distribution, dB_dx)

    distN = initial_distribution @ N
    distNdQ_dxNRU = distN @ torch.einsum("abc,b->ac", dQ_dx, (N @ (R @ U)))
    distNdR_dxU = distN @ (torch.einsum("abc,b->ac", dR_dx, U))
    dobj_dx = distNdQ_dxNRU + distNdR_dxU
    print(dobj_dx.shape)
    if lib == np:
        dobj_dx = dobj_dx.detach().numpy()

    return dobj_dx

def obj_hessian_matrix_form(coverage_probs, G, phi, U, initial_distribution, omega=4, lib=torch):
    x = torch.autograd.Variable(coverage_probs.detach(), requires_grad=True)
    dobj_dx = dobj_dx_matrix_form(torch.Tensor(x), G, phi, U, initial_distribution, omega=4, lib=torch)
    m = len(x)
    obj_hessian = torch.zeros((m,m))
    for i in range(len(x)):
        obj_hessian[i] = torch.autograd.grad(dobj_dx[i], x, create_graph=False, retain_graph=True)[0]

    return obj_hessian
# """

if __name__ == "__main__":

    # CODE BLOCK FOR GENERATING G, U, INITIAL_DISTRIBUTION, BUDGET
    G=returnGraph(fixed_graph=1)
    # G = nx.DiGraph(G)
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
    
    budget = 1
    omega = 4


    # CODE BLOCK FOR GENERATING PHI (GROUND TRUTH PHI GENERATED FOR NOW)
    node_feature_size=25
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
    phi=torch.Tensor(generatePhi(G))
    transition_probs = phi2prob(G, phi)

    # initial_coverage_prob = torch.rand(nx.number_of_edges(G), requires_grad=True) / 10
    initial_coverage_prob_res = get_optimal_coverage_prob(G, transition_probs, U, initial_distribution, budget, omega=omega)
    # initial_coverage_prob = torch.rand(E, requires_grad=True)
    # initial_coverage_prob = initial_coverage_prob / torch.sum(initial_coverage_prob) * budget
    # initial_coverage_prob = torch.autograd.Variable(torch.Tensor([9.55394184e-12, 1.05733076e-12, 8.09883103e-02, 2.90116910e-02,
    #    1.20345780e-12, 1.15105842e-12, 1.35015588e-12, 1.15172351e-11,
    #    1.12268636e-12, 1.18212753e-12, 1.22446281e-12]), requires_grad=True)
    initial_coverage_prob = torch.autograd.Variable(torch.Tensor(initial_coverage_prob_res['x']), requires_grad=True)

    # initial_coverage_prob = torch.zeros(nx.number_of_edges(G), requires_grad=True) / 10
    # initial_coverage_prob.retain_grad()
    coverage_probs = initial_coverage_prob.detach()

    obj_matrix_form = objective_function_matrix_form(initial_coverage_prob, G, transition_probs, torch.Tensor(U), torch.Tensor(initial_distribution), range(E), omega)

    # derivatives...
    dobj_dx = dobj_dx_matrix_form(initial_coverage_prob, G, transition_probs, U, initial_distribution, range(E), omega)

    torch_dobj_dx = torch.autograd.grad(obj_matrix_form, initial_coverage_prob, create_graph=True, retain_graph=True)[0]
    empirical_dobj_dx = torch.zeros(11)

    torch_obj_hessian = obj_hessian_matrix_form(coverage_probs, G, transition_probs, U, initial_distribution, edge_set=range(E), omega=omega)

    eigenvalues, eigenvectors = np.linalg.eig(torch_obj_hessian)
    indices = sorted(enumerate(eigenvalues), reverse=False, key = lambda x: x[1])
    i1, i2 = indices[0][0], indices[1][0]
    v1, v2 = torch.Tensor(eigenvectors[i1].real), torch.Tensor(eigenvectors[i2].real)
    print("Eigen decomposition:", eigenvalues)

    # graph plotting
    x_axis = np.linspace(-0.01, 0.01, 30)
    input_points_np = np.array([[x1, x2] for x1 in x_axis for x2 in x_axis])
    input_points_torch = [coverage_probs + v1 * x1 + v2 * x2 for (x1,x2) in input_points_np]

    optimal_obj = objective_function_matrix_form(coverage_probs, G, transition_probs, torch.Tensor(U), torch.Tensor(initial_distribution), range(E), omega=omega).item()
    labels = np.array([objective_function_matrix_form(x, G, transition_probs, torch.Tensor(U), torch.Tensor(initial_distribution), range(E), omega=omega).item() - (torch_dobj_dx @ (x - coverage_probs)).item() - optimal_obj for x in input_points_torch])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(input_points_np[:,0], input_points_np[:,1], labels)
    plt.show()

    print(np.linalg.eig(torch_obj_hessian))
