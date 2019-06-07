# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 02:33:57 2019

@author: Aditya
"""
import networkx as nx 
import numpy as np
import torch

from gcn import * 

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

def generatePhi(G, possible_ranges=[(0,1), (1,8), (8,10)]):
    
    N= nx.number_of_nodes(G)
    sources=G.graph['sources']
    targets=G.graph['targets']
    
    for node in list(G.nodes()):
        
        if node in sources:
            range_of_phi=possible_ranges[0]
        elif node in targets:
            range_of_phi=possible_ranges[-1]
        else:
            range_of_phi=possible_ranges[1]
        #node_features=np.random.randn(feature_length)
        # TODO: Use a better feature computation for a given node
        #r=np.random.choice(len(possible_ranges))
        #range_of_phi=possible_ranges[r]
        lower_bounds=range_of_phi[0]
        upper_bounds=range_of_phi[1]
        node_phi=np.random.uniform(lower_bounds, upper_bounds)
        G.node[node]['node_phi']=node_phi
    
    # Generate features
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
    
    
    
def returnGraph(fixed_graph=False, n_sources=1, n_targets=1, N_low=16, N_high=20, e_low=0.6, e_high=0.7, budget=0.05):
    
    if fixed_graph:
        # define an arbitrary graph with a source and target node
        source=0
        target=6
        G= nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)]) # undirected graph
        # G = nx.to_directed(G) # for directed graph
        G.graph['source']=source
        G.graph['target']=target
        G.graph['sources']=[source]
        G.graph['targets']=[target]
        G.graph['U']=np.array([10, -20])
        G.graph['budget']=budget*nx.number_of_edges(G)
        
        G.node[target]['utility']=10
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution
        return G
    
    else:
        # HARD CODE THE BELOW TWO VALUES
        N=np.random.randint(low=N_low, high=N_high)                     # Randomly pick number of Nodes
        edge_prob=np.random.uniform(low=e_low, high=e_high)          # Randomly pick Edges probability                                          
        
        # Generate random graph
        M=int(edge_prob*(N*(N-1)/2.0))                          # Calculate expected number of edges
        G=nx.gnm_random_graph(N, M)
        # G=G.to_directed()                                       # Change the undirected graph to directed graph
        
        # Pick source and target randomly and ensure that path exists
        # TODO:
        # Make size=src+trgt
        source, target= np.random.choice(list(G.nodes()), size=2, replace=False)
        path_exists_between_source_target=nx.has_path(G, source, target)
        while(not path_exists_between_source_target):
            source, target= np.random.choice(list(G.nodes()), size=2, replace=False)
            path_exists_between_source_target=nx.has_path(G, source, target)
        G.graph['source']=source
        G.graph['target']=target
        G.graph['sources']=[source]
        G.graph['targets']=[target]
        G.graph['budget']=budget*nx.number_of_edges(G)
        G.graph['U']=[]
        
        # Randomly assign utilities to targets in the RANGE HARD-CODED below
        for target in G.graph['targets']:
            G.node[target]['utility']=np.random.randint(20, high=50)
            G.graph['U'].append(G.node[target]['utility'])
        G.graph['U'].append(np.random.randint(-40, high=-20))
        #G.graph['U'].append(0) # indifferent of getting caught
        # G.graph['U'].append(np.random.randint(-80, high=-60)) # negative payoff
        G.graph['U']=np.array(G.graph['U'])
        
        sources=G.graph['sources']
        nodes=list(G.nodes())
        transients=[node for node in nodes if not (node in G.graph['targets'])]
        initial_distribution=np.array([1.0/len(sources) if n in sources else 0.0 for n in transients])
        G.graph['initial_distribution']=initial_distribution
        
        return G
        
def generate_PathProbs_from_Attractiveness(G, coverage_prob,  phi, all_paths, n_paths,omega=4):
    
    #coverage_prob=torch.from_numpy(coverage_prob)
    #all_paths=torch.from_numpy(all_paths)
    #n_paths=torch.from_numpy(n_paths)

    N=nx.number_of_nodes(G) 

    # GENERATE EDGE PROBABILITIES 
    edge_probs=torch.zeros((N,N))
    for i, node in enumerate(list(G.nodes())):
        nieghbors = list(G[node])
        
        smuggler_probs=torch.zeros(len(neighbors))
        for j,neighbor in enumerate(neighbors):
            e=(node, neighbor)
            #pe= G.edge[node][neighbor]['coverage_prob']
            pe=coverage_prob[node][neighbor]
            smuggler_probs[j]=torch.exp(-omega*pe+phi[neighbor])
        
        smuggler_probs=smuggler_probs*1.0/torch.sum(smuggler_probs)
        
        for j,neighbor in enumerate(neighbors):
            edge_probs[node,neighbor]=smuggler_probs[j]
          
            
    
    # GENERATE PATH PROBABILITIES
    path_probs=torch.zeros(n_paths)
    for path_number, path in enumerate(all_paths):
        path_prob=torch.ones(1)
        for i in range(len(path)-1):
            path_prob=path_prob*(edge_probs[path[i], path[i+1]])
        path_probs[path_number]=path_prob
    path_probs=path_probs/torch.sum(path_probs)
    path_probs[-1]=torch.max(torch.zeros(1), path_probs[-1]-(sum(path_probs)-1.000))
    #print ("SUM1: ",sum(path_probs))
    #path_probs=torch.from_numpy(path_probs)
    #print ("Path probs:", path_probs, sum(path_probs))
    
    return path_probs

def generate_EdgeProbs_from_Attractiveness(G, coverage_prob, phi,omega=4):
        
    N=nx.number_of_nodes(G) 

    # GENERATE EDGE PROBABILITIES 
    edge_probs=torch.zeros((N,N))
    for i, node in enumerate(list(G.nodes())):
        neighbors = list(G[node])
        
        smuggler_probs=torch.zeros(len(neighbors))
        for j,neighbor in enumerate(neighbors):
            e=(node, neighbor)
            #pe= G.edge[node][neighbor]['coverage_prob']
            pe=coverage_prob[node][neighbor]
            smuggler_probs[j]=torch.exp(-omega*pe+phi[neighbor])
        
        smuggler_probs=smuggler_probs*1.0/torch.sum(smuggler_probs)
        
        for j,neighbor in enumerate(neighbors):
            edge_probs[node,neighbor]=smuggler_probs[j]
            
    return edge_probs

def generateSyntheticData(node_feature_size, omega=4, 
                          n_training_graphs=80, n_testing_graphs=200, 
                          training_samples_per_graph=1, testing_samples_per_graph=1,
                          fixed_graph=False, path_type='random_walk',
                          N_low=16, N_high=20, e_low=0.6, e_high=0.7, budget=0.05):
    '''
    '''
    
    data=[]             # Not used anymore  
    training_data=[]
    testing_data=[]
    net1=GCNDataGenerationNet(node_feature_size)
    net3= featureGenerationNet(node_feature_size)        
    #training_graphs= [returnGraph(fixed_graph=fixed_graph) for _ in range(n_training_graphs)]
    #testing_graphs= [returnGraph(fixed_graph=fixed_graph) for _ in range(n_testing_graphs)]
    #print ("GRAPHS N: ",len(training_graphs), len(testing_graphs))
    #n_training_samples=int(n_data_samples*(1.0-testing_data_fraction))
    n_training_samples=n_training_graphs*training_samples_per_graph
    n_testing_samples=n_testing_graphs*testing_samples_per_graph
    n_data_samples=n_training_samples+n_testing_samples
    
    print("N_samples, Training/Testing: ", n_training_samples, n_testing_samples)
    for graph_number in range(n_training_graphs+n_testing_graphs):
        
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
        G=returnGraph(fixed_graph=fixed_graph,N_low=N_low, N_high=N_high, e_low=e_low, e_high=e_high, budget=budget)
        # COMPUTE ADJACENCY MATRIX
        A=nx.to_numpy_matrix(G)
        A_torch=torch.as_tensor(A, dtype=torch.float) 
        N=nx.number_of_nodes(G) 
        
            
        # Randomly assign coverage probability
        private_coverage_prob=np.random.rand(nx.number_of_edges(G))
        private_coverage_prob/=sum(private_coverage_prob)
        coverage_prob=torch.zeros(N,N)
        for i, e in enumerate(list(G.edges())):
            #G.edge[e[0]][e[1]]['coverage_prob']=coverage_prob[i]
            coverage_prob[e[0]][e[1]]=private_coverage_prob[i]
            coverage_prob[e[1]][e[0]]=private_coverage_prob[i]
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
        #
        #
        #
        
        Fv= generateFeatures(G, node_feature_size)
        Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
        # Generate attractiveness values for nodes
        phi=net1.forward(Fv_torch,A_torch).view(-1)
        
        #
        #
        '''
        REPLACE THE ABOVE CODE SNIPPET WITH FOLLOWING LINES TO GENERATE FEATURES FROM PHI INSTEAD OF OTHER WAY ROUND
        '''
        #
        #
        
        '''
        phi=generatePhi(G)
        phi=torch.as_tensor(phi, dtype=torch.float)
        #Generate features from phi values
        Fv_torch=net3.forward(phi.view(-1,1), A_torch)
        Fv=Fv_torch.detach().numpy()
        '''
        
        
        #phi=y.data.numpy()
        '''
        phi is the attractiveness function, phi(v,f) for each of the N nodes, v
        '''
        source=G.graph['source']
        target=G.graph['target']
                        
        if path_type=='simple_paths':
            # NOTE: THIS NEEDS TO BE MODIFIED WITH THE NEW DATA GENETATION SCHEME. CURRENTLY UNUSABLE

            if graph_number<n_training_graphs:
                n_samples_for_current_graph=training_samples_per_graph
                list_name=training_data
            else:
                n_samples_for_current_graph=testing_samples_per_graph
                list_name=testing_data
            for _ in range(n_samples_for_current_graph):    
                
                Fv= generateFeatures(G, node_feature_size)
            
                Fv_torch=torch.as_tensor(Fv, dtype=torch.float)
                # Generate attractiveness values for nodes
                phi=net1.forward(Fv_torch,A_torch).view(-1)
                #phi=y.data.numpy()
                '''
                phi is the attractiveness function, phi(v,f) for each of the N nodes, v
                '''           
                # COMPUTE ALL POSSIBLE PATHS
                all_paths=list(nx.all_simple_paths(G, source, target))
                n_paths=len(all_paths)
                       
                # GENERATE SYNTHETIC DATA:         
                path_probs=generate_PathProbs_from_Attractiveness(G,coverage_prob,phi, all_paths, n_paths)
                #print ("SUM2:", torch.sum(path_probs), path_probs)
                #data_point=np.random.choice(n_paths,size=1, p=path_probs)
                #data_point=(Fv, coverage_prob, path_probs)
                data_point=(G,Fv, coverage_prob, phi, path_probs)
                list_name.append(data_point)
            
        
        
        
        elif path_type=='random_walk':
            
            edge_probs=generate_EdgeProbs_from_Attractiveness(G, coverage_prob, phi)
                
            if graph_number<n_training_graphs:
                for _ in range(training_samples_per_graph):
                    path=getMarkovianWalk(G, edge_probs)
                    data_point=(G,Fv,coverage_prob, phi, path)
                    training_data.append(data_point)
            else:
                for _ in range(testing_samples_per_graph):
                    path=getMarkovianWalk(G, edge_probs)
                    data_point=(G,Fv,coverage_prob, phi, path)
                    testing_data.append(data_point)
            
        
        elif path_type=='random_walk_distribution':
            
            edge_probs=generate_EdgeProbs_from_Attractiveness(G, coverage_prob, phi)
            if graph_number<n_training_graphs:
                for _ in range(training_samples_per_graph):
                    path=getMarkovianWalk(G, edge_probs)
                    log_prob=torch.zeros(1)
                    for e in path: 
                        log_prob-=torch.log(edge_probs[e[0]][e[1]])
                    data_point=(G,Fv,coverage_prob, phi,path, log_prob)
                    training_data.append(data_point)
            else:
                for _ in range(testing_samples_per_graph):
                    path=getMarkovianWalk(G, edge_probs)
                    log_prob=torch.zeros(1)
                    for e in path: 
                        log_prob-=torch.log(edge_probs[e[0]][e[1]])
                    data_point=(G,Fv,coverage_prob, phi, path, log_prob)                    
                    testing_data.append(data_point)
                    
        elif path_type=='empirical_distribution':
            
            edge_probs=generate_EdgeProbs_from_Attractiveness(G, coverage_prob, phi)
                
            if graph_number<n_training_graphs:
                empirical_transition_prob=np.zeros((N,N))
                
                for _ in range(training_samples_per_graph):
                    path=getMarkovianWalk(G, edge_probs)
                    data_point=(G,Fv,coverage_prob, phi, path)
                    training_data.append(data_point)
                    for e in path:
                        empirical_transition_prob[e[0]][e[1]]+=1
                        
                for n in range(len(empirical_transition_prob)):
                    empirical_transition_prob[n]=empirical_transition_prob[n]/sum(empirical_transition_prob[n])
                empirical_transition_prob=torch.tensor(empirical_transition_prob)   
                for i in range(training_samples_per_graph):
                    path=training_data[-1-i][-1]
                    log_prob=torch.zeros(1)
                    for e in path: 
                        log_prob-=torch.log(empirical_transition_prob[e[0]][e[1]])
                    training_data[-1-i]+=tuple([log_prob])
                
            
            else:
                for _ in range(testing_samples_per_graph):
                    path=getMarkovianWalk(G, edge_probs)
                    data_point=(G,Fv,coverage_prob, phi, path)
                    testing_data.append(data_point)
        
    
    if path_type=='empirical_distribution':
        for graph_number in range(n_training_graphs):
            pass
            
            
        

                
    return np.array(training_data), np.array(testing_data)


if __name__=="__main__":
    
    
    # define an arbitrary graph with a source and target node
    source=0
    target=6
    G= nx.Graph([(source,1),(source,2),(1,2),(1,3),(1,4),(2,4),(2,5),(4,5),(3,target),(4,target),(5,target)], source=0, target=6)
    
