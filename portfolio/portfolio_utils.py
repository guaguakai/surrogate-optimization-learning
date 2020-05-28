import torch
import tqdm
import time
from utils import computeCovariance

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

MAX_NORM = 0.1

def train_portfolio(model, covariance_model, optimizer, epoch, dataset, training_method='two-stage', device='cpu', evaluate=False):
    model.train()
    covariance_model.train()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    alpha = 0.0001
    REG = 0.0001

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
                # ================== check the optimal solution =======
                L_real = torch.cholesky(Q_real)
                x_opt, = cvxpylayer(L_real, labels)
                opt = labels @ x_opt - 0.5 * alpha * x_opt.t() @ Q_real @ x_opt
            else:
                obj, opt = torch.Tensor([0]), torch.Tensor([0])

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
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%', opt=f'{opt.item()*100:.6f}%')

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def surrogate_train_portfolio():
    pass

def validate_portfolio(model, covariance_model, scheduler, epoch, dataset, training_method='two-stage', device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    alpha = 0.01
    REG = 0.01

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
                # ================== check the optimal solution =======
                L_real = torch.cholesky(Q_real)
                x_opt, = cvxpylayer(L_real, labels)
                opt = labels @ x_opt - 0.5 * alpha * x_opt.t() @ Q_real @ x_opt
            else:
                obj, opt = torch.Tensor([0]), torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%', opt=f'{opt.item()*100:.6f}%')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)

    if (epoch > 0):
        if training_method == "two-stage":
            scheduler.step(average_loss)
        elif training_method == "decision-focused" or training_method == "surrogate-decision-focused":
            scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def surrogate_validate_portfolio():
    pass

def test_portfolio(model, covariance_model, epoch, dataset, device='cpu', evaluate=False):
    model.eval()
    covariance_model.eval()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    alpha = 0.01
    REG = 0.01

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
                # ================== check the optimal solution =======
                L_real = torch.cholesky(Q_real)
                x_opt, = cvxpylayer(L_real, labels)
                opt = labels @ x_opt - 0.5 * alpha * x_opt.t() @ Q_real @ x_opt
            else:
                obj, opt = torch.Tensor([0]), torch.Tensor([0])

            test_losses.append(loss.item())
            test_objs.append(obj.item())
            tqdm_loader.set_postfix(loss=f'{loss.item():.6f}', obj=f'{obj.item()*100:.6f}%', opt=f'{opt.item()*100:.6f}%')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj # , (forward_time, inference_time, qp_time, backward_time)

def surrogate_test_portfolio():
    pass
