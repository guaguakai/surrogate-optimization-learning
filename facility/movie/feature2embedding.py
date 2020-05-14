import torch
import torch.nn as nn

def linear_block(in_channels, out_channels, activation='ReLU'):
    if activation == 'ReLU':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               nn.ReLU()
               )
    elif activation == 'Sigmoid':
        return nn.Sequential(
               nn.Linear(in_channels, out_channels),
               nn.BatchNorm1d(out_channels),
               nn.Sigmoid()
               )

class Feature2Embedding(nn.Module):
    def __init__(self, input_size=20, output_size=8):
        # input:  features
        # output: embedding
        super(Feature2Embedding, self).__init__()
        self.input_size, self.output_size = input_size, output_size
        self.model = nn.Sequential(
                linear_block(input_size, 512),
                linear_block(512, 512),
                linear_block(512, 256),
                # linear_block(256, output_size, activation='ReLU')
                linear_block(256, output_size, activation='Sigmoid')
                )

    def forward(self, x):
        return self.model(x)

