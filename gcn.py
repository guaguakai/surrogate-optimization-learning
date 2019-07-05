from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# class GCNDataGenerationNet(nn.Module):
#     
#     """
#     For data generation, assume a 4 layer gcn to compress features, followed by a two layer NN to predict phi(v,f)
#     
#     """
#     def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[20,15,12,10], nn_hidden_layer_sizes=[7,4]):
#         super(GCNDataGenerationNet, self).__init__()
#         
#         r1,r2,r3,r4=gcn_hidden_layer_sizes       
#         r5,r6=nn_hidden_layer_sizes
#         
#         # Define the layers of gcn 
#         self.gcn1 = nn.Linear(raw_feature_size, r1, bias=False)
#         self.gcn2 = nn.Linear(r1, r2, bias=False)
#         self.gcn3 = nn.Linear(r2, r3, bias=False)
#         self.gcn4 = nn.Linear(r3, r4, bias=False)
#         
#         #Define the layers of NN to predict the attractiveness function for every node
#         self.fc1 = nn.Linear(r4, r5)
#         self.fc2= nn.Linear (r5, r6)
#         self.fc3 = nn.Linear(r6, 1)
#         
#         #self.node_adj=A
# 
#     def forward(self, x, A):
#         """
#         Inputs:            
#             x  is the feature vector of size Nxr where r is the number of features of a single node and N is no. of nodes
#             A is the adjacency matrix of the graph under consideration. 
#         
#         Output:
#             Returns the compressed feature matrix of size N X r_compressed
#             
#         """
#         
#         # Input, x is the nXk feature matrix with features for each of the n nodes. 
#         #A=self.node_adj
#         #x=torch.rand(10,25)
# 
#         #x=torch.from_numpy(x)
#         x=F.relu(self.gcn1(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn2(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn3(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn4(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         
#         x=F.relu(self.fc1(x))
#         x=F.relu(self.fc2(x))
#         x=torch.sigmoid(self.fc3(x)) * 10 # scale up
#         #x=torch.mul(x, 1)
#         # Now, x is a nX1 tensor consisting of phi(v,f) for each of the n nodes v.
#         
#         return x
#     
#     
# class featureGenerationNet(nn.Module):
#     
#     """
#     For feature generation, assume a two layer NN to decompress phi to compressed features, followed by a 4 layer GCN
#     to decompress this to features of size feature_size,
#     
#     """
#     def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[20,15,12,10], nn_hidden_layer_sizes=[7,4]):
#         super(featureGenerationNet, self).__init__()
#         
#         r1,r2,r3,r4=gcn_hidden_layer_sizes       
#         r5,r6=nn_hidden_layer_sizes
# 
#         #Define the layers of NN to predict the compressed feature vector for every node
#         self.fc1 = nn.Linear(1, r6)
#         self.fc2= nn.Linear (r6,r5)
#         self.fc3 = nn.Linear(r5, r4)
#         
#         # Define the layers of gcn 
#         self.gcn1 = nn.Linear(r4, r3, bias=False)
#         self.gcn2 = nn.Linear(r3, r2, bias=False)
#         self.gcn3 = nn.Linear(r2, r1, bias=False)
#         self.gcn4 = nn.Linear(r1, raw_feature_size, bias=False)
#         
#         #self.node_adj=A
# 
#     def forward(self, x, A):
#         """
#         Inputs:            
#             phi  is the feature vector of size Nxr where r is the number of features of a single node and N is no. of nodes
#             A is the adjacency matrix of the graph under consideration. 
#         
#         Output:
#             Returns the feature matrix of size N X r
#             
#         """
#         
#         # Input, x is the nXk feature matrix with features for each of the n nodes. 
#         #A=self.node_adj
#         #x=torch.rand(10,25)
# 
#         x=F.relu(self.fc1(x))
#         x=F.relu(self.fc2(x))
#         x=torch.sigmoid(self.fc3(x))
# 
#         #x=torch.from_numpy(x)
#         x=F.relu(self.gcn1(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn2(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn3(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn4(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         
#         #x=torch.mul(x, 1)
#         # Now, x is a nXr tensor consisting of features for each of the n nodes v.
#         
#         return x
        
class featureGenerationNet2(nn.Module): # message passing version
    
    """
    For feature generation, assume a two layer NN to decompress phi to compressed features, followed by a 4 layer GCN
    to decompress this to features of size feature_size,
    
    """
    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[20,15,12,10], nn_hidden_layer_sizes=[32,16,8]):
        super(featureGenerationNet2, self).__init__()
        
        r1,r2,r3,r4 = gcn_hidden_layer_sizes       
        r5,r6,r7    = nn_hidden_layer_sizes

        #Define the layers of NN to predict the compressed feature vector for every node
        self.fc1 = nn.Linear(r4, r5)
        self.fc2 = nn.Linear(r5, r6)
        self.fc3 = nn.Linear(r6, r7)
        self.fc4 = nn.Linear(r7, raw_feature_size)
        
        # Define the layers of gcn 
        self.gcn1 = GraphConv(1,  r1, aggr='mean')
        self.gcn2 = GraphConv(r1, r2, aggr='mean')
        self.gcn3 = GraphConv(r2, r3, aggr='mean')
        self.gcn4 = GraphConv(r3, r4, aggr='mean')

        self.activation = nn.Softplus()
        # self.activation = F.relu
        #self.node_adj=A

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
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = self.activation(self.gcn3(x, edge_index))
        x = self.activation(self.gcn4(x, edge_index))

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)

        #x=torch.from_numpy(x)
        # x=F.relu(self.gcn3(x, edge_index))
        # x=self.gcn4(x, edge_index)
        
        # Now, x is a nXr tensor consisting of features for each of the n nodes v.
        
        return x
    
# class GCNPredictionNet(nn.Module):
# 
#     def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[15,10], nn_hidden_layer_sizes=5):
#         super(GCNPredictionNet, self).__init__()
#         
#         r1,r2=gcn_hidden_layer_sizes       
#         r3=nn_hidden_layer_sizes
#         
#         # Define the layers of gcn 
#         self.gcn1 = nn.Linear(raw_feature_size, r1, bias=False)
#         self.gcn2 = nn.Linear(r1, r2, bias=False)
#         
#         #Define the layers of NN to predict the attractiveness function for every node
#         self.fc1 = nn.Linear(r2, r3)
#         self.fc2 = nn.Linear(r3, 1)
#         
#         #self.node_adj=A
# 
#     def forward(self, x,A):
#         
#         ''' 
#         Input:
#             x is the nXk feature matrix with features for each of the n nodes.
#             A is the adjacency matrix for the graph under consideration
#         '''
#         
#         x=F.relu(self.gcn1(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         x=F.relu(self.gcn2(torch.matmul(A+torch.eye(A.size()[0]),x)))
#         
#         x=F.relu(self.fc1(x))
#         x=torch.sigmoid(self.fc2(x)) * 10 # scale up
#         # Now, x is a nX1 tensor consisting of the predicted phi(v,f) for each of the n nodes v.
#         x=torch.mul(x, 1)
#         
#         return x

class GCNPredictionNet2(nn.Module):

    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[15,10], nn_hidden_layer_sizes=5):
        super(GCNPredictionNet2, self).__init__()
        
        r1,r2=gcn_hidden_layer_sizes
        r3=nn_hidden_layer_sizes
        
        # Define the layers of gcn 
        self.gcn1 = GraphConv(raw_feature_size, r1, aggr='mean')
        self.gcn2 = GraphConv(r1, r2, aggr='mean')
        
        #Define the layers of NN to predict the attractiveness function for every node
        self.fc1 = nn.Linear(r2, 1)
        self.dropout = nn.Dropout()
        # self.fc1 = nn.Linear(r2, r3)
        # self.fc2 = nn.Linear(r3, 1)

        self.activation = nn.Softplus()
        # self.activation = F.relu
        
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
        
        # x=self.dropout(x)
        x = self.fc1(x)
        x = x - torch.min(x)
        # x = x * 10
        # x=F.relu(x)
        # x=self.fc2(x)
        # x=torch.sigmoid(x) * 20 # scale up
        # x = x / torch.max(x) * 10
        # Now, x is a nX1 tensor consisting of the predicted phi(v,f) for each of the n nodes v.
        
        return x


#net = Net()
#print(net)
