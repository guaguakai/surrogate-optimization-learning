import torch
import torch.nn as nn
from utils import computeCovariance

def linear_block(in_channels, out_channels, activation='ReLU'):
    if activation == 'ReLU':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               torch.nn.Dropout(p=0.5),
               nn.LeakyReLU()
               )
    elif activation == 'Sigmoid':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               torch.nn.Dropout(p=0.5),
               nn.Sigmoid()
               )

class PortfolioModel(nn.Module):
    def __init__(self, input_size=20, output_size=1):
        # input:  features
        # output: embedding
        super(PortfolioModel, self).__init__()
        self.input_size, self.output_size = input_size, output_size
        self.model = nn.Sequential(
                linear_block(input_size, 100),
                linear_block(100, 100),
                linear_block(100, output_size, activation='Sigmoid')
                )

    def forward(self, x):
        y = self.model(x)
        return (y - 0.5)

class CovarianceModel(nn.Module):
    def __init__(self, n):
        super(CovarianceModel, self).__init__()
        self.n = n
        self.latent_dim = 32
        self.embedding = nn.Embedding(num_embeddings=self.n, embedding_dim=self.latent_dim)

    def forward(self):
        security_embeddings = self.embedding(torch.LongTensor(range(self.n)))
        cov = computeCovariance(security_embeddings)
        return cov


