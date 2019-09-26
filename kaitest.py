from graphData import generate_EdgeProbs_from_Attractiveness
import networkx as nx
import numpy as np
import torch
import time
import copy
import autograd
import numdifftools as nd

from graphData import generateSyntheticData, returnGraph, generatePhi
from blockDerivative import get_optimal_coverage_prob, objective_function_matrix_form
from blockDerivative import dobj_dx_matrix_form, obj_hessian_matrix_form, phi2prob
# from coverageProbability import get_optimal_coverage_prob, objective_function_matrix_form, objective_function_matrix_form_np
# from coverageProbability import dobj_dx_matrix_form, dobj_dx_matrix_form_np, obj_hessian_matrix_form, obj_hessian_matrix_form_np, phi2prob

if __name__ == "__main__":

    # CODE BLOCK FOR GENERATING G, U, INITIAL_DISTRIBUTION, BUDGET
    budget = 2
    N_low = 10
    p = 0.3
    G=returnGraph(fixed_graph=0, n_sources=2, n_targets=2, N_low=N_low, N_high=N_low+1, e_low=p, e_high=p, budget=budget)
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
    
    omega = 4

    # PRECOMPUTE A MIN-CUT
    m = G.number_of_edges()
    edges = G.edges()
    edge2index = {}
    for idx, edge in enumerate(edges):
        edge2index[edge] = idx
        edge2index[(edge[1], edge[0])] = idx

    dummyG = copy.deepcopy(G)
    dummyG.add_nodes_from(['ds', 'dt']) # 1000 dummy source, 2000 dummy target
    for x in dummyG.graph['sources']:
        dummyG.add_edge('ds', x, capacity=100)
    for x in dummyG.graph['targets']:
        dummyG.add_edge(x, 'dt', capacity=100)

    value, partition = nx.minimum_cut(dummyG, 'ds', 'dt')
    print('cut size:', value)
    partition0, partition1 = set(partition[0]), set(partition[1])
    cut = []
    for idx, edge in enumerate(G.edges()):
        if edge[0] in partition0 and edge[1] in partition1:
            cut.append(idx)
        elif edge[0] in partition1 and edge[1] in partition0:
            cut.append(idx)

    cut_size = 5
    edge_set = np.random.choice(range(len(edges)), size=cut_size, replace=False)

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
    phi=torch.Tensor(generatePhi(G, fixed_phi=2))
    transition_probs = phi2prob(G, phi)
    print(edge_set)
    print([list(G.edges())[edge] for edge in edge_set])

    # initial_coverage_prob = torch.rand(nx.number_of_edges(G), requires_grad=True) / 10initial_coverage_prob_res = get_optimal_coverage_prob(G, transition_probs, U, initial_distribution, budget, omega=omega)
    initial_coverage_prob_res = get_optimal_coverage_prob(G, transition_probs, U, initial_distribution, budget, omega=omega)
    # initial_coverage_prob = initial_coverage_prob / torch.sum(initial_coverage_prob) * budget
    # initial_coverage_prob = torch.autograd.Variable(torch.Tensor([9.55394184e-12, 1.05733076e-12, 8.09883103e-02, 2.90116910e-02,
    #    1.20345780e-12, 1.15105842e-12, 1.35015588e-12, 1.15172351e-11,
    #    1.12268636e-12, 1.18212753e-12, 1.22446281e-12]), requires_grad=True)
    initial_coverage_prob = torch.autograd.Variable(torch.Tensor(initial_coverage_prob_res['x']), requires_grad=True)
    print(initial_coverage_prob)
    # initial_coverage_prob = torch.autograd.Variable(torch.Tensor([0] * len(edge_set)), requires_grad=True)
    # print(initial_coverage_prob)

    # initial_coverage_prob = torch.zeros(nx.number_of_edges(G), requires_grad=True) / 10
    # initial_coverage_prob.retain_grad()
    coverage_probs = initial_coverage_prob.detach()

    obj_matrix_form = objective_function_matrix_form(initial_coverage_prob, G, transition_probs, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set=edge_set, omega=omega)
    print('objective value:', obj_matrix_form)

    # derivatives...
    dobj_dx = dobj_dx_matrix_form(initial_coverage_prob, G, transition_probs, U, initial_distribution, edge_set=edge_set, omega=omega)
    print('torch version:', dobj_dx)

    torch_dobj_dx = torch.autograd.grad(obj_matrix_form, initial_coverage_prob, create_graph=True, retain_graph=True)[0][edge_set]
    print('autograd version:', torch_dobj_dx)

    print("autograd computing...")
    start_time = time.time()
    torch_obj_hessian = obj_hessian_matrix_form(coverage_probs, G, transition_probs, U, initial_distribution, edge_set=edge_set, omega=omega)
    print(time.time() - start_time)

    eigenvalues, eigenvectors = np.linalg.eig(torch_obj_hessian)
    indices = sorted(enumerate(eigenvalues), reverse=False, key = lambda x: x[1])
    i1, i2 = indices[0][0], indices[1][0]
    v1, v2 = torch.Tensor(eigenvectors[i1].real), torch.Tensor(eigenvectors[i2].real)
    print("Eigen decomposition:", eigenvalues)

    # graph plotting
    x_axis = np.linspace(-0.01, 0.01, 30)
    input_points_np = np.array([[x1, x2] for x1 in x_axis for x2 in x_axis])
    input_points_torch = [coverage_probs + v1 * x1 + v2 * x2 for (x1,x2) in input_points_np]

    optimal_obj = objective_function_matrix_form(coverage_probs, G, transition_probs, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set=edge_set, omega=omega).item()
    print('optimal obj: {}'.format(optimal_obj))
    labels = np.array([objective_function_matrix_form(x, G, transition_probs, torch.Tensor(U), torch.Tensor(initial_distribution), edge_set=edge_set, omega=omega).item() - (torch_dobj_dx @ (x - coverage_probs)).item() - optimal_obj for x in input_points_torch])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(input_points_np[:,0], input_points_np[:,1], labels)
    plt.show()
