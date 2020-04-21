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

class FacilityNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(FacilityNN, self).__init__()
        # feature_shape = (n, k), each node has k features
        # output_shape  = (n, m), each node has m prediction (cost)
        input_size, output_size = input_shape[0] * input_shape[1], output_shape[0] * output_shape[1]
        self.input_shape, self.output_shape = input_shape, output_shape
        self.input_size, self.output_size = input_size, output_size
        self.model = nn.Sequential(
                linear_block(input_size, 512),
                linear_block(512, 512),
                linear_block(512, 256),
                linear_block(256, output_size, activation='ReLU')
                # linear_block(256, output_size, activation='Sigmoid')
                )

    def forward(self, x):
        batch_size = len(x)
        x_flatten = x.view(batch_size, -1)
        output = torch.clamp(self.model(x_flatten).view(batch_size, *self.output_shape), min=0)
        return output / 2 * torch.mean(output)

class FeatureNN(nn.Module):
    def __init__(self, input_shape, output_shape): # generated feature size
        super(FeatureNN, self).__init__()
        input_size, output_size = input_shape[0] * input_shape[1], output_shape[0] * output_shape[1]
        self.input_shape, self.output_shape = input_shape, output_shape
        self.input_size, self.output_size = input_size, output_size
        self.model = nn.Sequential(
                linear_block(input_size, 512),
                linear_block(512, 512),
                linear_block(512, 256),
                linear_block(256, output_size, activation='ReLU')
                # linear_block(256, output_size, activation='Sigmoid')
                )

    def forward(self, x):
        batch_size = len(x)
        x_flatten = x.view(batch_size, -1)
        output = self.model(x_flatten).view(batch_size, *self.output_shape)
        return output

