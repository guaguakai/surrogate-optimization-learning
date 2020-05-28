import sys
import pandas as pd
import torch
import numpy as np
import qpth
import scipy
import cvxpy as cp
import random
import argparse
import tqdm
import time
import datetime as dt
from cvxpylayers.torch import CvxpyLayer

import torch.nn
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

def computeCovariance(covariance_mat):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n = len(covariance_mat)
    cosine_matrix = torch.zeros((n,n))
    for i in range(n):
        cosine_matrix[i] = cos(covariance_mat, covariance_mat[i].repeat(n,1))
    return cosine_matrix

def generateDataset(data_loader, n=200, num_samples=100):
    feature_mat, target_mat, feature_cols, covariance_mat, target_name, dates, symbols = data_loader.load_pytorch_data()
    feature_mat    = feature_mat[:num_samples,:n]
    target_mat     = target_mat[:num_samples,:n]
    covariance_mat = covariance_mat[:num_samples,:n]
    symbols = symbols[:n]
    dates = dates[:num_samples]

    num_samples = len(dates)

    sample_shape, feature_size = feature_mat.shape, feature_mat.shape[-1]

    feature_mat = feature_mat.reshape(-1,feature_size)
    feature_mat = (feature_mat - torch.mean(feature_mat, dim=0)) / (torch.std(feature_mat, dim=0) + 1e-5)
    feature_mat = feature_mat.reshape(sample_shape, feature_size)

    dataset = data_utils.TensorDataset(feature_mat, covariance_mat, target_mat)

    indices = list(range(num_samples))
    np.random.shuffle(indices)

    train_size, validate_size = int(num_samples * 0.7), int(num_samples * 0.1)
    train_indices    = indices[:train_size]
    validate_indices = indices[train_size:train_size+validate_size]
    test_indices     = indices[train_size+validate_size:]

    batch_size = 1
    train_dataset    = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validate_dataset = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validate_indices))
    test_dataset     = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    # train_dataset    = dataset[train_indices]
    # validate_dataset = dataset[validate_indices]
    # test_dataset     = dataset[test_indices]

    return train_dataset, validate_dataset, test_dataset
