import torch
import tqdm
import time
from utils import computeCovariance

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

MAX_NORM = 0.1

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

from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection

alpha = 1e-5
REG = 1e-5

def computeCovariance(covariance_mat):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n = len(covariance_mat)
    cosine_matrix = torch.zeros((n,n))
    for i in range(n):
        cosine_matrix[i] = cos(covariance_mat, covariance_mat[i].repeat(n,1))
    return cosine_matrix

def generateDataset(data_loader, n=200, num_samples=100):
    feature_mat, target_mat, feature_cols, covariance_mat, target_name, dates, symbols = data_loader.load_pytorch_data()
    np.random.seed(0)
    symbol_indices = np.random.choice(n, len(symbols), replace=False)
    feature_mat    = feature_mat[:num_samples,symbol_indices]
    target_mat     = target_mat[:num_samples,symbol_indices]
    covariance_mat = covariance_mat[:num_samples,symbol_indices]
    symbols = symbols[symbol_indices]
    dates = dates[:num_samples]

    num_samples = len(dates)

    sample_shape, feature_size = feature_mat.shape, feature_mat.shape[-1]

    feature_mat = feature_mat.reshape(-1,feature_size)
    feature_mat = (feature_mat - torch.mean(feature_mat, dim=0)) / (torch.std(feature_mat, dim=0) + 1e-5)
    feature_mat = feature_mat.reshape(sample_shape, feature_size)

    dataset = data_utils.TensorDataset(feature_mat, covariance_mat, target_mat)

    indices = list(range(num_samples))
    # np.random.shuffle(indices)

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

def train_portfolio(model, covariance_model, optimizer, epoch, dataset, training_method='two-stage', device='cpu', evaluate=False):
    model.train()
    covariance_model.train()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)

            if evaluate:
                if training_method == 'two-stage':
                    Q = torch.eye(n)
                else:
                    # Q = torch.eye(n)
                    Q = covariance_model() + torch.eye(n) * REG  # TODO

                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = torch.cholesky(Q)
                # =============== solving QP using CVXPY ===============
                x_var = cp.Variable(n)
                L_para = cp.Parameter((n,n))
                p_para = cp.Parameter(n)
                constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) <= 1]
                objective = cp.Minimize(0.5 * cp.sum_squares(L_para @ x_var) - p_para.T @ x_var)
                problem = cp.Problem(objective, constraints)

                cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                x, = cvxpylayer(L, p)

                obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            # ====================== back-prop =====================
            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'two-stage':
                    loss.backward()
                elif training_method == 'decision-focused':
                    (-obj).backward()
                    # (-obj + loss).backward() # TODO
                    for parameter in model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("no grad is backpropagated...")
                pass
            optimizer.step()
            backward_time += time.time() - backward_start_time

            train_losses.append(loss.item())
            train_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%') 

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def surrogate_train_portfolio(model, covariance_model, T, optimizer, epoch, dataset, training_method='surrogate', device='cpu', evaluate=False):
    model.train()
    covariance_model.train()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    T_size = T.shape[1]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)

            Q = covariance_model() + torch.eye(n) * REG 

            forward_time += time.time() - forward_start_time
            inference_start_time = time.time()

            p = predictions @ T
            L = torch.cholesky(T.t() @ Q @ T)
            # =============== solving QP using CVXPY ===============
            y_var = cp.Variable(T_size)
            L_para = cp.Parameter((T_size,T_size))
            p_para = cp.Parameter(T_size)
            T_para = cp.Parameter((n,T_size))
            constraints = [y_var >= 0, T_para @ y_var >= 0, cp.sum(T_para @ y_var) <= 1]
            objective = cp.Minimize(0.5 * cp.sum_squares(L_para @ y_var) - p_para.T @ y_var)
            problem = cp.Problem(objective, constraints)

            cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para, T_para], variables=[y_var])
            y, = cvxpylayer(L, p, T)
            x = T @ y

            obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

            inference_time += time.time() - inference_start_time

            # ====================== back-prop =====================
            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'surrogate':
                    covariance = computeCovariance(T.t())
                    T_weight = 0.00001
                    T_loss     = torch.sum(covariance) - torch.sum(torch.diag(covariance))

                    (-obj + T_loss * T_weight).backward()
                    for parameter in model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("no grad is backpropagated...")
                pass
            optimizer.step()
            T.data = normalize_matrix_positive(T.data)
            backward_time += time.time() - backward_start_time

            train_losses.append(loss.item())
            train_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%', T_loss=f'{T_loss:.3f}')

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def validate_portfolio(model, covariance_model, scheduler, epoch, dataset, training_method='two-stage', device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]

            loss = loss_fn(predictions, labels)

            if evaluate:
                Q = covariance_model() + torch.eye(n) * REG 

                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = torch.cholesky(Q)
                # =============== solving QP using CVXPY ===============
                x_var = cp.Variable(n)
                L_para = cp.Parameter((n,n))
                p_para = cp.Parameter(n)
                constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) <= 1]
                objective = cp.Minimize(0.5 * cp.sum_squares(L_para @ x_var) - p_para.T @ x_var)
                problem = cp.Problem(objective, constraints)

                cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                x, = cvxpylayer(L, p)

                obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)

    if (epoch > 0):
        if training_method == "two-stage":
            scheduler.step(average_loss)
        elif training_method == "decision-focused" or training_method == "surrogate":
            scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def surrogate_validate_portfolio(model, covariance_model, T, scheduler, epoch, dataset, training_method='surrogate', device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    T_size = T.shape[1]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)

            Q = covariance_model() + torch.eye(n) * REG 

            forward_time += time.time() - forward_start_time
            inference_start_time = time.time()

            p = predictions @ T
            L = torch.cholesky(T.t() @ Q @ T)
            # =============== solving QP using CVXPY ===============
            y_var = cp.Variable(T_size)
            L_para = cp.Parameter((T_size,T_size))
            p_para = cp.Parameter(T_size)
            T_para = cp.Parameter((n,T_size))
            constraints = [y_var >= 0, T_para @ y_var >= 0, cp.sum(T_para @ y_var) <= 1]
            objective = cp.Minimize(0.5 * cp.sum_squares(L_para @ y_var) - p_para.T @ y_var)
            problem = cp.Problem(objective, constraints)

            cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para, T_para], variables=[y_var])
            y, = cvxpylayer(L, p, T)
            x = T @ y

            obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

            validate_losses.append(loss.item())
            validate_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%')

    average_loss    = np.mean(validate_losses)
    average_obj     = np.mean(validate_objs)

    if (epoch > 0):
        if training_method == "two-stage":
            scheduler.step(average_loss)
        elif training_method == "decision-focused" or training_method == "surrogate":
            scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj

def test_portfolio(model, covariance_model, epoch, dataset, device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            Q = covariance_model() + torch.eye(n) * REG 

            if epoch == -1:
                predictions = labels
                Q = Q_real
            loss = loss_fn(predictions, labels)

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = torch.cholesky(Q)
                # =============== solving QP using CVXPY ===============
                x_var = cp.Variable(n)
                L_para = cp.Parameter((n,n))
                p_para = cp.Parameter(n)
                constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) <= 1]
                objective = cp.Minimize(0.5 * cp.sum_squares(L_para @ x_var) - p_para.T @ x_var)
                problem = cp.Problem(objective, constraints)

                cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                x, = cvxpylayer(L, p)

                obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%') 

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def surrogate_test_portfolio(model, covariance_model, T, epoch, dataset, device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    T_size = T.shape[1]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)

            Q = covariance_model() + torch.eye(n) * REG 

            forward_time += time.time() - forward_start_time
            inference_start_time = time.time()

            p = predictions @ T
            L = torch.cholesky(T.t() @ Q @ T)
            # =============== solving QP using CVXPY ===============
            y_var = cp.Variable(T_size)
            L_para = cp.Parameter((T_size,T_size))
            p_para = cp.Parameter(T_size)
            T_para = cp.Parameter((n,T_size))
            constraints = [y_var >= 0, T_para @ y_var >= 0, cp.sum(T_para @ y_var) <= 1]
            objective = cp.Minimize(0.5 * cp.sum_squares(L_para @ y_var) - p_para.T @ y_var)
            problem = cp.Problem(objective, constraints)

            cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para, T_para], variables=[y_var])
            y, = cvxpylayer(L, p, T)
            x = T @ y

            obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj
