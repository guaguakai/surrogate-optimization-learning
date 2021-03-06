from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

aggregation_function_generation = 'add' # either mean or add
aggregation_function = 'add' # either mean or add

Conv = GraphConv

class featureGenerationNet2(nn.Module): # message passing version
    
    """
    For feature generation, assume a two layer NN to decompress phi to compressed features, followed by a 4 layer GCN
    to decompress this to features of size feature_size,
    
    """
    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[4, 7, 10, 16], nn_hidden_layer_sizes=[32, 16]):
        super(featureGenerationNet2, self).__init__()
        
        self.r1, self.r2, self.r3, self.r4 = gcn_hidden_layer_sizes       
        self.r5, self.r6                   = nn_hidden_layer_sizes

        #Define the layers of NN to predict the compressed feature vector for every node
        self.fc1 = nn.Linear(self.r4, self.r5)
        self.fc2 = nn.Linear(self.r5, self.r6)
        # self.fc3 = nn.Linear(self.r6, self.r7)
        self.fc3 = nn.Linear(self.r6, raw_feature_size)
        
        # Define the layers of gcn 
        self.gcn1 = Conv(1,       self.r1, aggr=aggregation_function_generation)
        self.gcn2 = Conv(self.r1, self.r2, aggr=aggregation_function_generation)
        self.gcn3 = Conv(self.r2, self.r3, aggr=aggregation_function_generation)
        self.gcn4 = Conv(self.r3, self.r4, aggr=aggregation_function_generation)

        # self.activation = nn.Softplus()
        self.activation = F.relu
        # self.activation = nn.Sigmoid()
        self.noise_std = 0.0

    def forward(self, x, edge_index):
        """
        Inputs:            
            phi  is the feature vector of size Nxr where r is the number of features of a single node and N is no. of nodes
            A is the adjacency matrix of the graph under consideration. 
        
        Output:
            Returns the feature matrix of size N X r
            
        """
        
        # Input, x is the nXk feature matrix with features for each of the n nodes. 
        #A=self.node_adj
        #x=torch.rand(10,25)
        x = self.activation(self.gcn1(x, edge_index)) # + self.noise_std * torch.randn(self.r1) 
        x = self.activation(self.gcn2(x, edge_index)) # + self.noise_std * torch.randn(self.r2)
        x = self.activation(self.gcn3(x, edge_index)) # + self.noise_std * torch.randn(self.r3)
        x = self.activation(self.gcn4(x, edge_index)) # + self.noise_std * torch.randn(self.r4)

        x = self.activation(self.fc1(x)) # + self.noise_std * torch.randn(self.r5) 
        x = self.activation(self.fc2(x)) # + self.noise_std * torch.randn(self.r6)
        # x = self.activation(self.fc3(x)) # + self.noise_std * torch.randn(self.r6)
        x = self.fc3(x)


        # Now, x is a nXr tensor consisting of features for each of the n nodes v.
        
        return x
    
class GCNPredictionNet2(nn.Module):

    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[8, 16], nn_hidden_layer_sizes=32):
        super(GCNPredictionNet2, self).__init__()
        
        r0 =raw_feature_size
        r1, r2 = gcn_hidden_layer_sizes
        n1 = nn_hidden_layer_sizes
        
        # Define the layers of gcn 
        self.gcn1 = Conv(r0, r1, aggr=aggregation_function)
        self.gcn2 = Conv(r1, r2, aggr=aggregation_function)
        # self.gcn3 = Conv(r2, r3, aggr=aggregation_function)
        # self.gcn4 = Conv(r3, r4, aggr=aggregation_function)
        
        #Define the layers of NN to predict the attractiveness function for every node
        # self.fc1 = nn.Linear(r2, 1)
        self.fc1 = nn.Linear(r2, n1)
        self.fc2 = nn.Linear(n1, 1)
        # self.fc3 = nn.Linear(n2, 1)

        # self.activation = nn.Softplus()
        self.activation = F.relu
        # self.activation = nn.Sigmoid()

        
        #self.node_adj=A

    def forward(self, x, edge_index):
        
        ''' 
        Input:
            x is the nXk feature matrix with features for each of the n nodes.
            A is the adjacency matrix for the graph under consideration
        '''
        
        x = self.activation(self.gcn1(x, edge_index))
        # x = self.dropout(x)
        x = self.activation(self.gcn2(x, edge_index))
        # x = self.activation(self.gcn3(x, edge_index))
        # x = self.activation(self.gcn4(x, edge_index))
        
        # x = self.dropout(x)
        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        x = self.fc2(x)
        x = x - torch.mean(x)

        # only allow attractiveness value between -10 ~ 10
        # x = -nn.Softplus()(-x + 10) + 10
        # x =  nn.Softplus()(x  + 10) - 10
        # x = x / (torch.std(x))
        # x = nn.ReLU6()(x)

        # Now, x is a nX1 tensor consisting of the predicted phi(v,f) for each of the n nodes v.
        
        return x


#net = Net()
#print(net)
