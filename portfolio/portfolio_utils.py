import torch
import tqdm
import time
from utils import computeCovariance

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

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
from sqrtm import sqrtm

alpha = 2
REG = 0.1
solver = 'cvxpy'

MAX_NORM = 0.1
T_MAX_NORM = 0.1

def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

def computeCovariance(covariance_mat):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n = len(covariance_mat)
    cosine_matrix = torch.zeros((n,n))
    for i in range(n):
        cosine_matrix[i] = cos(covariance_mat, covariance_mat[i].repeat(n,1))
    return cosine_matrix

def generateDataset(data_loader, n=200, num_samples=100):
    feature_mat, target_mat, feature_cols, covariance_mat, target_name, dates, symbols = data_loader.load_pytorch_data()
    symbol_indices = np.random.choice(len(symbols), n, replace=False)
    feature_mat    = feature_mat[:num_samples,symbol_indices]
    target_mat     = target_mat[:num_samples,symbol_indices]
    covariance_mat = covariance_mat[:num_samples,symbol_indices]
    symbols = [symbols[i] for i in symbol_indices]
    dates = dates[:num_samples]

    num_samples = len(dates)

    sample_shape, feature_size = feature_mat.shape, feature_mat.shape[-1]

    # ------ normalization ------
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
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)
            Q = covariance_model() * (1 - REG) + torch.eye(n) * REG  # TODO

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = sqrtm(Q) # torch.cholesky(Q)
                # =============== solving QP using qpth ================
                if solver == 'qpth':
                    G = -torch.eye(n)
                    h = torch.zeros(n)
                    A = torch.ones(1,n)
                    b = torch.ones(1)
                    qp_solver = qpth.qp.QPFunction()
                    x = qp_solver(alpha * Q, -p, G, h, A, b)[0]
                # =============== solving QP using CVXPY ===============
                elif solver == 'cvxpy':
                    x_var = cp.Variable(n)
                    L_para = cp.Parameter((n,n))
                    p_para = cp.Parameter(n)
                    constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
                    objective = cp.Minimize(0.5 * alpha * cp.sum_squares(L_para @ x_var) + p_para.T @ x_var)
                    problem = cp.Problem(objective, constraints)

                    cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                    x, = cvxpylayer(L, -p)

                obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

                inference_time += time.time() - inference_start_time
                # ======= opt ===
                # p_opt = labels
                # L_opt = torch.cholesky(Q_real)
                # x_opt, = cvxpylayer(L_opt, p_opt)
                # opt = labels @ x_opt - 0.5 * alpha * x.t() @ Q_real @ x
                # print('obj:', obj, 'opt:', opt)
            else:
                obj = torch.Tensor([0])

            # ====================== back-prop =====================
            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'two-stage':
                    Q_loss = torch.norm(Q - Q_real)
                    (loss + Q_loss).backward()
                elif training_method == 'decision-focused':
                    (-obj).backward()
                    # (-obj + loss).backward() # TODO
                    for parameter in model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    for parameter in covariance_model.parameters():
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

def surrogate_train_portfolio(model, covariance_model, T_init, optimizer, T_optimizer, epoch, dataset, training_method='surrogate', device='cpu', evaluate=False):
    model.train()
    covariance_model.train()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    T_size = T_init.shape[1]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)


            # randomly select column to update
            # T = init_T
            T = T_init.detach().clone()
            random_column = torch.randint(T_init.shape[1], [1])
            T[:,random_column] = T_init[:,random_column]

            Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 

            forward_time += time.time() - forward_start_time
            inference_start_time = time.time()

            p = predictions @ T
            L = sqrtm(T.t() @ Q @ T) # torch.cholesky(T.t() @ Q @ T)
            # =============== solving QP using qpth ================
            if solver == 'qpth':
                G = -torch.eye(n) @ T
                h = torch.zeros(n)
                A = torch.ones(1,n) @ T
                b = torch.ones(1)
                qp_solver = qpth.qp.QPFunction()
                y = qp_solver(alpha * T.t() @ Q @ T, -p, G, h, A, b)[0]
                x = T @ y
            # =============== solving QP using CVXPY ===============
            elif solver == 'cvxpy':
                y_var = cp.Variable(T_size)
                L_para = cp.Parameter((T_size,T_size))
                p_para = cp.Parameter(T_size)
                T_para = cp.Parameter((n,T_size))
                constraints = [T_para @ y_var >= 0, cp.sum(T_para @ y_var) == 1]
                objective = cp.Minimize(0.5 * alpha * cp.sum_squares(L_para @ y_var) + p_para.T @ y_var)
                problem = cp.Problem(objective, constraints)

                cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para, T_para], variables=[y_var])
                y, = cvxpylayer(L, -p, T)
                x = T @ y
            # print("predicted objective value:", predictions.t() @ x - 0.5 * alpha * x.t() @ Q @ x)

            obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x
            # print("real objective value:", obj)

            inference_time += time.time() - inference_start_time

            # ====================== back-prop =====================
            optimizer.zero_grad()
            T_optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'surrogate':
                    covariance = computeCovariance(T.t())
                    T_weight = 0.0
                    TS_weight = 0.0
                    T_loss     = torch.sum(covariance) - torch.sum(torch.diag(covariance))

                    (-obj + T_weight * T_loss).backward()
                    for parameter in model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    for parameter in covariance_model.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    T_init.grad = torch.clamp(T_init.grad, min=-T_MAX_NORM, max=T_MAX_NORM)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("no grad is backpropagated...")
                pass
            optimizer.step()
            T_optimizer.step()
            T_init.data = normalize_matrix_positive(T_init.data)
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
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]

            loss = loss_fn(predictions, labels)

            if evaluate:
                Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 

                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = sqrtm(Q) # torch.cholesky(Q)
                # =============== solving QP using qpth ================
                if solver == 'qpth':
                    G = -torch.eye(n)
                    h = torch.zeros(n)
                    A = torch.ones(1,n)
                    b = torch.ones(1)
                    qp_solver = qpth.qp.QPFunction()
                    x = qp_solver(alpha * Q, -p, G, h, A, b)[0]
                # =============== solving QP using CVXPY ===============
                elif solver == 'cvxpy':
                    x_var = cp.Variable(n)
                    L_para = cp.Parameter((n,n))
                    p_para = cp.Parameter(n)
                    constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
                    objective = cp.Minimize(0.5 * alpha * cp.sum_squares(L_para @ x_var) - p_para.T @ x_var)
                    problem = cp.Problem(objective, constraints)
    
                    cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                    x, = cvxpylayer(L, p)

                obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

                inference_time += time.time() - inference_start_time
            else:
                obj = torch.Tensor([0])

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

    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def surrogate_validate_portfolio(model, covariance_model, T, scheduler, T_scheduler, epoch, dataset, training_method='surrogate', device='cpu', evaluate=False):
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
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            loss = loss_fn(predictions, labels)

            Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 

            forward_time += time.time() - forward_start_time
            inference_start_time = time.time()

            p = predictions @ T
            L = sqrtm(T.t() @ Q @ T) # torch.cholesky(T.t() @ Q @ T)
            # =============== solving QP using qpth ================
            if solver == 'qpth':
                G = -torch.eye(n) @ T
                h = torch.zeros(n)
                A = torch.ones(1,n) @ T
                b = torch.ones(1)
                qp_solver = qpth.qp.QPFunction()
                y = qp_solver(alpha * T.t() @ Q @ T, -p, G, h, A, b)[0]
                x = T @ y
            # =============== solving QP using CVXPY ===============
            elif solver == 'cvxpy':
                y_var = cp.Variable(T_size)
                L_para = cp.Parameter((T_size,T_size))
                p_para = cp.Parameter(T_size)
                T_para = cp.Parameter((n,T_size))
                constraints = [T_para @ y_var >= 0, cp.sum(T_para @ y_var) == 1]
                objective = cp.Minimize(0.5 * alpha * cp.sum_squares(L_para @ y_var) + p_para.T @ y_var)
                problem = cp.Problem(objective, constraints)
    
                cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para, T_para], variables=[y_var])
                y, = cvxpylayer(L, -p, T)
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
        elif training_method == "decision-focused":
            scheduler.step(-average_obj)
        elif training_method == "surrogate":
            # covariance = computeCovariance(T.t())
            # T_loss     = torch.sum(covariance) - torch.sum(torch.diag(covariance))
            scheduler.step(-average_obj)
            T_scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj

def test_portfolio(model, covariance_model, epoch, dataset, device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG

            if epoch == -1:
                predictions = labels
                Q = Q_real
            else:
                predictions = model(features.float())[:,0]
                Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 

            loss = loss_fn(predictions, labels)

            if evaluate:
                forward_time += time.time() - forward_start_time
                inference_start_time = time.time()

                p = predictions
                L = sqrtm(Q) # torch.cholesky(Q)
                # =============== solving QP using qpth ================
                if solver == 'qpth':
                    G = -torch.eye(n)
                    h = torch.zeros(n)
                    A = torch.ones(1,n)
                    b = torch.ones(1)
                    qp_solver = qpth.qp.QPFunction()
                    x = qp_solver(alpha * Q, -p, G, h, A, b)[0]
                    # x_opt = qp_solver(alpha * Q_real, -labels, G, h, A, b)[0]
                # =============== solving QP using CVXPY ===============
                elif solver == 'cvxpy':
                    x_var = cp.Variable(n)
                    L_para = cp.Parameter((n,n))
                    p_para = cp.Parameter(n)
                    constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
                    objective = cp.Minimize(0.5 * alpha * cp.sum_squares(L_para @ x_var) + p_para.T @ x_var)
                    problem = cp.Problem(objective, constraints)

                    cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para], variables=[x_var])
                    x, = cvxpylayer(L, -p)

                obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x
                # opt = labels @ x_opt - 0.5 * alpha * x_opt.t() @ Q_real @ x_opt
                # print('obj:', obj, 'opt:', opt)

                inference_time += time.time() - inference_start_time
                # ======= opt ===
                # p_opt = labels
                # L_opt = torch.cholesky(Q_real)
                # x_opt, = cvxpylayer(L_opt, p_opt)
                # opt = labels @ x_opt - 0.5 * alpha * x_opt.t() @ Q_real @ x_opt
                # test_opts.append(opt.item())
            else:
                obj = torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%') 

    # print('opts:', test_opts)
    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def surrogate_test_portfolio(model, covariance_model, T, epoch, dataset, device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    T_size = T.shape[1]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features[0].to(device), covariance_mat[0].to(device), labels[0,:,0].to(device).float() # only one single data
            n = len(covariance_mat)
            Q_real = computeCovariance(covariance_mat) * (1 - REG) + torch.eye(n) * REG
            predictions = model(features.float())[:,0]
            Q = covariance_model() * (1 - REG) + torch.eye(n) * REG 
            loss = loss_fn(predictions, labels)

            forward_time += time.time() - forward_start_time
            inference_start_time = time.time()

            p = predictions @ T
            L = sqrtm(T.t() @ Q @ T) # torch.cholesky(T.t() @ Q @ T)
            # =============== solving QP using qpth ================
            if solver == 'qpth':
                G = -torch.eye(n) @ T
                h = torch.zeros(n)
                A = torch.ones(1,n) @ T
                b = torch.ones(1)
                qp_solver = qpth.qp.QPFunction()
                y = qp_solver(alpha * T.t() @ Q @ T, -p, G, h, A, b)[0]
                x = T @ y
            # =============== solving QP using CVXPY ===============
            elif solver == 'cvxpy':
                y_var = cp.Variable(T_size)
                L_para = cp.Parameter((T_size,T_size))
                p_para = cp.Parameter(T_size)
                T_para = cp.Parameter((n,T_size))
                constraints = [T_para @ y_var >= 0, cp.sum(T_para @ y_var) == 1]
                objective = cp.Minimize(0.5 * alpha * cp.sum_squares(L_para @ y_var) + p_para.T @ y_var)
                problem = cp.Problem(objective, constraints)

                cvxpylayer = CvxpyLayer(problem, parameters=[L_para, p_para, T_para], variables=[y_var])
                y, = cvxpylayer(L, -p, T)
                x = T @ y

            obj = labels @ x - 0.5 * alpha * x.t() @ Q_real @ x

            # ======= opt ===
            # p_opt = labels @ T
            # L_opt = torch.cholesky(T.t() @ Q_real @ T)
            # y_opt, = cvxpylayer(L_opt, p_opt, T)
            # x_opt = T @ y_opt
            # opt = labels @ x_opt - 0.5 * alpha * x_opt.t() @ Q_real @ x_opt
            # test_opts.append(opt.item())

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj
