# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:33:57 2019

@author: Aditya
"""
import networkx as nx 
import numpy as np
import torch
import torch.utils.data as utils
import random
import copy

from gcn import *
from coverageProbability import prob2unbiased, phi2prob

# Random Seed Initialization
# SEED = 1289 #  random.randint(0,10000)
# print("Random seed: {}".format(SEED))
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)


"""
Create the graph here. Define source and target. compute the adjacency matrix here. Compute all possible paths. 
Randomly also generate features (for now) for every node in the graph.  
Next, handle the data generation also here. So as part of this: 
    make a gcn object with suitable layers (more layers) for generation. 
    Pass the features through the gcn and obtain edge probabilities---> path probs for all paths
    Generate a dataset (train+test) by randomly sampling from the above prob distribution (can use one-hot vectors to denote paths)
    Split the data set into training and testing

"""

def getMarkovianWalk(G, edge_probs):
    '''
    Return a list of edges corresponding to the random walk
    '''
    sources=G.graph['sources']
    targets=G.graph['targets']
    #initial_distribution= TODO:Assumed to be uniform for now
    nodes=list(G.nodes())
    start_node= np.random.choice(sources)
    
    edge_list=[]
    current_node=start_node
    while (not(current_node in targets)):
        neighbors = list(G[current_node]) # works for directed graph
        transition_probs=np.array([(edge_probs[current_node][n]).detach().numpy() for n in neighbors])
        transition_probs/=(transition_probs).sum()
        next_node=np.random.choice(neighbors, p=transition_probs)
        edge_list.append((current_node, next_node))
        current_node=next_node
    
    return edge_list    


def generateFeatures(G, feature_length):
    
    N= nx.number_of_nodes(G)
    # Define node features for each of the n nodes
    #possible_feature_ranges=[(0,1),(4,6),(20,30)]
    possible_feature_ranges=[(0,1)]
    for node in list(G.nodes()):
        #node_features=np.random.randn(feature_length)
        # TODO: Use a better feature computation for a given node
        ranges=np.random.choice(len(possible_feature_ranges),feature_length)
        ranges=[possible_feature_ranges[r] for r in ranges]
        lower_bounds=[r[0] for r in ranges]
        upper_bounds=[r[1] for r in ranges]
        node_features=np.random.uniform(lower_bounds, upper_bounds)
        G.node[node]['node_features']=node_features
    
    # Generate features
    Fv=np.zeros((N,feature_length))
    for node in list(G.nodes()):
            Fv[node]=G.node[node]['node_features']
    return Fv        

def generatePhi(G, possible_ranges=[(0,0.5), (0.5,5), (5,8)], fixed_phi=0):
    
    N= nx.number_of_nodes(G)
    sources=G.graph['sources']
    targets=G.graph['targets']
    diameter=nx.diameter(G)
    if fixed_phi == 2:
        for node in list(G.nodes()):
            if node in sources:
                node_phi=0
            elif node in targets:
                node_phi=5 if node == 15 else 0
            else:
                node_phi=0
            G.node[node]['node_phi']=node_phi

    elif fixed_phi == 3:
        for i in range(3):
            for node in range(i*5, i*5+5):
                lower_bound = possible_ranges[i][0]
                upper_bound = possible_ranges[i][1]
                node_phi = np.random.uniform(lower_bound, upper_bound)
                G.node[node]['node_phi'] = node_phi

    elif fixed_phi == 0 or fixed_phi == 1:
        for node in list(G.nodes()):
            # Compute distance from the target: 
            dist_target=min([nx.shortest_path_length(G, source=node, target=target) for target in targets])
            dist_src_trg=0
            for s in sources: 
                dist_src_trg=max(dist_src_trg,min([nx.shortest_path_length(G, source=s, target=target) for target in targets]))
            
            if node in sources:
                range_of_phi=possible_ranges[0]
            elif node in targets:
                range_of_phi=possible_ranges[-1]
            else:
                range_of_phi=possible_ranges[1]
            
            #range_of_phi=(2**(5-dist_target),2**(6-dist_target))
            #range_of_phi=(((4.0*(diameter-dist_target))/diameter),((4.0*(1+diameter-dist_target))/diameter))
            #node_features=np.random.randn(feature_length)
            # TODO: Use a better feature computation for a given node
            #r=np.random.choice(len(possible_ranges))
            #range_of_phi=possible_ranges[r]
            lower_bounds=range_of_phi[0]
            upper_bounds=range_of_phi[1]
            node_phi=np.random.uniform(lower_bounds, upper_bounds)
            G.node[node]['node_phi']=node_phi
    
    phi=np.zeros(N)
    for node in list(G.nodes()):
            phi[node]=G.node[node]['node_phi']
    return phi        


def getSimplePath(G, path):
    
    """
    Collapses the input path which may contain cycles to obtain a simple path and returns the same.
    Input: path is a list of edges
    """
    path_nodes=[path[0][0]]+[e[1] for e in path]
    simple_path_nodes=[]
    
    # Generate node list of simple path
    idx=0
    while idx<len(path_nodes):
        node=path_nodes[idx]
        if not (node in path_nodes[idx+1:]):
            simple_path_nodes.append(node)
            idx+=1
        else:
            idx+=(path_nodes[idx+1:].index(node)+1)
    
    simple_path_edges=[(simple_path_nodes[i],simple_path_nodes[i+1]) for i in range(len(simple_path_nodes)-1)]
    
    return simple_path_edges
    
    
    
def returnGraph(fixed_graph=False, n_sources=1, n_targets=1, N_low=16, N_high=20, e_low=0.6, e_high=0.7, budget=2):
    
    if fixed_graph == 1:
        # define an arbitrary graph with a source and target node
        source=0
        target=6
        G = nx.Graph()
        nodes = [0,1,2,3,4,5,6]
        edges = [(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges) # undirected graph
        # G = nx.to_directed(G) # for directed graph
        G.graph['source']=source
        G.graph['target']=target
        G.graph['sources']=[source]
        G.graph['targets']=[target]
        G.graph['U']=np.array([10, 0])
        G.graph['budget']=budget
        
        G.node[target]['utility']=10
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution

    elif fixed_graph == 2:
        layers = [8,4,2,2]
        nodes = list(range(sum(layers)))
        sources, layer1, layer2, targets = nodes[:layers[0]], nodes[layers[0]:layers[0]+layers[1]], nodes[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]], nodes[-layers[3]:]
        transients = nodes[:-layers[3]]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(s, m1) for s in sources for m1 in layer1] + [(m1, m2) for m1 in layer1 for m2 in layer2] + [(m2, t) for m2 in layer2 for t in targets]) # undirected graph
        G.graph['sources']=sources
        G.graph['targets']=targets

        G.graph['U']=np.concatenate([np.random.rand(layers[-1]) * 10, np.array([0])])
        G.graph['budget']=budget
        
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution

    elif fixed_graph == 3:
        sizes = [5, 5, 5]
        probs = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.7]]
        is_connected = False
        while (not is_connected):
            G = nx.stochastic_block_model(sizes, probs)
            sources=[0]
            targets=[14]
            
            #Check if path exists:
            is_connected = nx.is_connected(G)

        G.graph['sources']=sources
        G.graph['targets']=targets
        G.graph['budget']=budget
        G.graph['U']=[]

        # Randomly assign utilities to targets in the RANGE HARD-CODED below
        for target in G.graph['targets']:
            G.node[target]['utility']=np.random.randint(low=5, high=10)
            G.graph['U'].append(G.node[target]['utility'])
        G.graph['U'].append(0)
        G.graph['U']=np.array(G.graph['U'])
        
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution

    else:
        is_connected = False
        src_target_is_ok=False
        while ((not is_connected) or (not src_target_is_ok)):
            N=np.random.randint(low=N_low, high=N_high)                     # Randomly pick number of Nodes
            p=np.random.uniform(low=e_low, high=e_high)             # Randomly pick Edges probability
            # Generate random graph
            # G = nx.gnp_random_graph(N,p)
            G = nx.random_geometric_graph(N,p)
            # G = nx.connected_watts_strogatz_graph(N,4,p) 

            sources_targets= np.random.choice(list(G.nodes()), size=n_sources+n_targets, replace=False)
            sources=sources_targets[:n_sources]
            targets=sources_targets[n_sources:]
            
            #Check if path exists:
            is_connected = nx.is_connected(G)
            
            if is_connected:
                #Check if src and TARGET ARE FAR ENOUGH
                diameter=nx.diameter(G)
                min_dist_src_target=diameter # Temporary large assignment
                for s in sources:
                    for t in targets:
                        dist_src_target=min(min_dist_src_target, nx.shortest_path_length(G, source=s, target=t))
                if min_dist_src_target>max((diameter/2.0),1):
                    src_target_is_ok=True
                
                
        G.graph['sources']=sources
        G.graph['targets']=targets
        G.graph['budget']=budget
        G.graph['U']=[]
        
        # Randomly assign utilities to targets in the RANGE HARD-CODED below
        for target in G.graph['targets']:
            G.node[target]['utility']=np.random.randint(low=5, high=10)
            G.graph['U'].append(G.node[target]['utility'])
        G.graph['U'].append(0)
        G.graph['U']=np.array(G.graph['U'])
        
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution

    # adding unit capacity to each edge
    for edge in G.edges():
        G[edge[0]][edge[1]]['capacity'] = 1
        G[edge[1]][edge[0]]['capacity'] = 1

    return G
        
# def generate_PathProbs_from_Attractiveness(G, coverage_prob,  phi, all_paths, n_paths,omega=4):
#     
#     #coverage_prob=torch.from_numpy(coverage_prob)
#     #all_paths=torch.from_numpy(all_paths)
#     #n_paths=torch.from_numpy(n_paths)
# 
#     N=nx.number_of_nodes(G) 
# 
#     # GENERATE EDGE PROBABILITIES 
#     edge_probs=torch.zeros((N,N))
#     for i, node in enumerate(list(G.nodes())):
#         nieghbors = list(G[node])
#         
#         smuggler_probs=torch.zeros(len(neighbors))
#         for j,neighbor in enumerate(neighbors):
#             e=(node, neighbor)
#             #pe= G.edge[node][neighbor]['coverage_prob']
#             pe=coverage_prob[node][neighbor]
#             smuggler_probs[j]=torch.exp(-omega*pe+phi[neighbor])
#         
#         smuggler_probs=smuggler_probs*1.0/torch.sum(smuggler_probs)
#         
#         for j,neighbor in enumerate(neighbors):
#             edge_probs[node,neighbor]=smuggler_probs[j]
#           
#             
#     
#     # GENERATE PATH PROBABILITIES
#     path_probs=torch.zeros(n_paths)
#     for path_number, path in enumerate(all_paths):
#         path_prob=torch.ones(1)
#         for i in range(len(path)-1):
#             path_prob=path_prob*(edge_probs[path[i], path[i+1]])
#         path_probs[path_number]=path_prob
#     path_probs=path_probs/torch.sum(path_probs)
#     path_probs[-1]=torch.max(torch.zeros(1), path_probs[-1]-(sum(path_probs)-1.000))
#     #print ("SUM1: ",sum(path_probs))
#     #path_probs=torch.from_numpy(path_probs)
#     #print ("Path probs:", path_probs, sum(path_probs))
#     
#     return path_probs

def generate_EdgeProbs_from_Attractiveness(G, coverage_probs, phi, omega=4):
    N=nx.number_of_nodes(G) 
    coverage_prob_matrix=torch.zeros((N,N))
    for i, e in enumerate(list(G.edges())):
        #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

    # GENERATE EDGE PROBABILITIES 
    adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
    exponential_term = torch.exp(- omega * coverage_prob_matrix) * torch.exp(phi) * adj
    transition_probs = exponential_term / torch.sum(exponential_term, keepdim=True, dim=1)
            
    return transition_probs

def attackerOracle(G, coverage_probs, phi, omega=4, num_paths=100):
    N=nx.number_of_nodes(G) 
    coverage_prob_matrix=torch.zeros((N,N))
    for i, e in enumerate(list(G.edges())):
        coverage_prob_matrix[e[0]][e[1]]=coverage_probs[i]
        coverage_prob_matrix[e[1]][e[0]]=coverage_probs[i] # for undirected graph only

    # EXACT EDGE PROBS
    biased_probs = generate_EdgeProbs_from_Attractiveness(G, coverage_probs, phi, omega=omega)

    # EMPIRICAL EDGE PROBS
    path_list = []
    simulated_defender_utility_list = []
    for _ in range(num_paths):
        path = getMarkovianWalk(G, biased_probs)
        # path = getSimplePath(G, path) # TODO
        path_list.append(path)

        defender_utility = -G.node[path[-1][1]]['utility']
        for e in path:
            defender_utility *= (1 - coverage_prob_matrix[e[0]][e[1]])
        simulated_defender_utility_list.append(defender_utility)

    simulated_defender_utility = np.mean(simulated_defender_utility_list)

    return path_list, simulated_defender_utility

def generateSyntheticData(node_feature_size, omega=4, 
                          n_graphs=20, samples_per_graph=100, empirical_samples_per_instance=10,
                          fixed_graph=False, path_type='random_walk',
                          N_low=16, N_high=20, e_low=0.6, e_high=0.7, budget=2, train_test_split_ratio=(0.7, 0.1, 0.2),
                          n_sources=1, n_targets=1, random_seed=0):
    
    # Random seed setting
    print("Random seed: {}".format(random_seed))
    if random_seed != 0:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # initialization
    data=[] # aggregate all the data first then split into training and testing
    net3= featureGenerationNet2(node_feature_size)
    n_samples = n_graphs * samples_per_graph
    
    print("N_samples: ", n_samples)
    for graph_number in range(n_graphs):
        
        '''
        # Pick the graph in cyclic fashion from the correct list of graphs
        graph_index=0
        if sample_number<n_training_samples:
            G=training_graphs[sample_number%n_training_graphs]
            graph_index=sample_number%n_training_graphs
        else:
            G=testing_graphs[sample_number%n_testing_graphs]
            graph_index=sample_number%n_testing_graphs
        '''
        while True:
            G = returnGraph(fixed_graph=fixed_graph, n_sources=n_sources, n_targets=n_targets, N_low=N_low, N_high=N_high, e_low=e_low, e_high=e_high, budget=budget)

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
            print(value)
            partition0, partition1 = set(partition[0]), set(partition[1])
            cut = []
            for idx, edge in enumerate(G.edges()):
                if edge[0] in partition0 and edge[1] in partition1:
                    cut.append(idx)
                elif edge[0] in partition1 and edge[1] in partition0:
                    cut.append(idx)

            print(cut)
            if value >= budget * 2:
                break

        # COMPUTE ADJACENCY MATRIX
        edge_index = torch.Tensor(list(nx.DiGraph(G).edges())).long().t()
        N=nx.number_of_nodes(G) 
        
        '''
        # Define node features for each of the n nodes
        for node in list(G.nodes()):
            node_features=np.random.randn(node_feature_size)
            # TODO: Use a better feature computation for a given node
            G.node[node]['node_features']=node_features
        
        # Generate features
        Fv=np.zeros((N,node_feature_size))
        
        for node in list(G.nodes()):
            Fv[node]=G.node[node]['node_features']
        '''
        for _ in range(samples_per_graph):
            # Randomly assign coverage probability
            private_coverage_prob = np.random.rand(nx.number_of_edges(G))
            private_coverage_prob = (private_coverage_prob / sum(private_coverage_prob)) * budget
            coverage_prob_matrix=torch.zeros(N,N)
            for i, e in enumerate(list(G.edges())):
                coverage_prob_matrix[e[0]][e[1]]=private_coverage_prob[i]
                coverage_prob_matrix[e[1]][e[0]]=private_coverage_prob[i]

            # Randomly generate attractiveness
            # phi is the attractiveness function, phi(v,f) for each of the N nodes, v
            phi=generatePhi(G, fixed_phi=fixed_graph)
            phi=torch.as_tensor(phi, dtype=torch.float)

            # Generate features from phi values
            Fv_torch=net3.forward(phi.view(-1,1), edge_index)
            Fv=Fv_torch.detach().numpy()
            
            # EXACT EDGE PROBS
            biased_probs = generate_EdgeProbs_from_Attractiveness(G, private_coverage_prob, phi, omega=omega)
            unbiased_probs = phi2prob(G, phi)

            # Call Attacker Oracle
            path_list, _ = attackerOracle(G, private_coverage_prob, phi, omega=omega, num_paths=empirical_samples_per_instance)

            # EMPIRICAL EDGE PROBS
            empirical_transition_probs=torch.zeros((N,N))
            for path in path_list:
                for e in path:
                    empirical_transition_probs[e[0]][e[1]]+=1

            row_sum = torch.sum(empirical_transition_probs, dim=1)
            adj = torch.Tensor(nx.adjacency_matrix(G).toarray())
            empirical_transition_probs[row_sum == 0] = adj[row_sum == 0]
            empirical_transition_probs = empirical_transition_probs / torch.sum(empirical_transition_probs, dim=1, keepdim=True)
            empirical_unbiased_probs = prob2unbiased(G, private_coverage_prob, empirical_transition_probs, omega)

            # DATA POINT
            if path_type=='random_walk_distribution':
                log_prob=torch.zeros(1)
                for path in path_list:
                    for e in path:
                        log_prob-=torch.log(biased_probs[e[0]][e[1]])
                log_prob /= len(path_list)
                data_point = (G, Fv, private_coverage_prob, phi, path_list, cut, log_prob, unbiased_probs)
                        
            elif path_type=='empirical_distribution':
                log_prob=torch.zeros(1)
                for path in path_list:
                    for e in path:
                        log_prob-=torch.log(empirical_transition_probs[e[0]][e[1]])
                log_prob /= len(path_list)
                data_point = (G, Fv, private_coverage_prob, phi, path_list, cut, log_prob, empirical_unbiased_probs)

            else:
                raise(TypeError)

            data.append(data_point)

    data = np.array(data)
    np.random.shuffle(data)

    print("average node size:", np.mean([x[0].number_of_nodes() for x in data]))
    print("average edge size:", np.mean([x[0].number_of_edges() for x in data]))

    train_size = int(train_test_split_ratio[0] * len(data))
    validate_size = int(train_test_split_ratio[1] * len(data))

    training_data, validate_data, testing_data = data[:train_size], data[train_size:train_size+validate_size], data[train_size+validate_size:]

    return np.array(training_data), np.array(validate_data), np.array(testing_data)


if __name__=="__main__":
    
    
    # define an arbitrary graph with a source and target node
    source=0
    target=6
    G= nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)], source=0, target=6)
    
