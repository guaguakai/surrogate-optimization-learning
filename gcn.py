from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNDataGenerationNet(nn.Module):
    
    """
    For data generation, assume a 4 layer gcn to compress features, followed by a two layer NN to predict phi(v,f)
    
    """
    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[20,15,12,10], nn_hidden_layer_sizes=[7,4]):
        super(GCNDataGenerationNet, self).__init__()
        
        r1,r2,r3,r4=gcn_hidden_layer_sizes       
        r5,r6=nn_hidden_layer_sizes
        
        # Define the layers of gcn 
        self.gcn1 = nn.Linear(raw_feature_size, r1, bias=False)
        self.gcn2 = nn.Linear(r1, r2, bias=False)
        self.gcn3 = nn.Linear(r2, r3, bias=False)
        self.gcn4 = nn.Linear(r3, r4, bias=False)
        
        #Define the layers of NN to predict the attractiveness function for every node
        self.fc1 = nn.Linear(r4, r5)
        self.fc2= nn.Linear (r5,r6)
        self.fc3 = nn.Linear(r6, 1)
        
        #self.node_adj=A

    def forward(self, x, A):
        """
        Inputs:            
            x  is the feature vector of size Nxr where r is the number of features of a single node and N is no. of nodes
            A is the adjacency matrix of the graph under consideration. 
        
        Output:
            Returns the compressed feature matrix of size N X r_compressed
            
        """
        
        # Input, x is the nXk feature matrix with features for each of the n nodes. 
        #A=self.node_adj
        #x=torch.rand(10,25)

        #x=torch.from_numpy(x)
        x=F.relu(self.gcn1(torch.matmul(A+torch.eye(A.size()[0]),x)))
        x=F.relu(self.gcn2(torch.matmul(A+torch.eye(A.size()[0]),x)))
        x=F.relu(self.gcn3(torch.matmul(A+torch.eye(A.size()[0]),x)))
        x=F.relu(self.gcn4(torch.matmul(A+torch.eye(A.size()[0]),x)))
        
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.sigmoid(self.fc3(x))
        #x=torch.mul(x, 1)
        # Now, x is a nX1 tensor consisting of phi(v,f) for each of the n nodes v.
        
        return x

class GCNPredictionNet(nn.Module):

    def __init__(self, raw_feature_size, gcn_hidden_layer_sizes=[15,10], nn_hidden_layer_sizes=5):
        super(GCNPredictionNet, self).__init__()
        
        r1,r2=gcn_hidden_layer_sizes       
        r3=nn_hidden_layer_sizes
        
        # Define the layers of gcn 
        self.gcn1 = nn.Linear(raw_feature_size, r1, bias=False)
        self.gcn2 = nn.Linear(r1, r2, bias=False)
        
        #Define the layers of NN to predict the attractiveness function for every node
        self.fc1 = nn.Linear(r2, r3)
        self.fc2 = nn.Linear(r3, 1)
        
        #self.node_adj=A

    def forward(self, x,A):
        
        ''' 
        Input:
            x is the nXk feature matrix with features for each of the n nodes.
            A is the adjacency matrix for the graph under consideration
        '''
        
        x=F.relu(self.gcn1(torch.matmul(A+torch.eye(A.size()[0]),x)))
        x=F.relu(self.gcn2(torch.matmul(A+torch.eye(A.size()[0]),x)))
        
        x=F.relu(self.fc1(x))
        x=F.sigmoid(self.fc2(x))
        # Now, x is a nX1 tensor consisting of the predicted phi(v,f) for each of the n nodes v.
        #x=torch.mul(x, 1)
        
        return x



#net = Net()
#print(net)