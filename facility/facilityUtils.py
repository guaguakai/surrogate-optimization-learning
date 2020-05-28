import sys
import tqdm
import time
import numpy as np
import qpth
import qpthlocal
import random
import scipy
import autograd
import torch
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import pairwise_distances

from gurobipy import *
from types import SimpleNamespace

from facilityNN import FacilityNN, FeatureNN
from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection, point_projection
from facilityDerivative import getObjective, getDerivative, getOptimalDecision, getHessian
from facilitySurrogateDerivative import getSurrogateObjective, getSurrogateDerivative, getSurrogateHessian, getSurrogateOptimalDecision

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from plot_utils import plot_graph

# Random Seed Initialization
# SEED = 1289 #  random.randint(0,10000)
# print("Random seed: {}".format(SEED))
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

MAX_NORM = 0.1

# uncapacitated facility location problem
def generateInstance(n, m):
    # while True:
    #     G = nx.random_geometric_graph(n+m, p)
    #     if nx.is_connected(G):
    #         break
    # pos = nx.get_node_attributes(G, 'pos')
    # for e in G.edges():
    #     G[e]['weight'] = np.linalg.norm(np.array(pos[e[0]]) - np.array(pos[e[1]]))
    # c = distance[:n,n:] # shipping cost per product

    c_prob = np.random.random((n,m)) * 0.2 # average 0.1 like
    c = np.random.binomial(np.ones((n,m)).tolist(), c_prob)
    d = np.ones(m) # customer demand
    f = np.ones(n) # facility open cost
    # d = np.random.random(m) # customer demand
    # f = np.random.random(n) # facility open cost
    return SimpleNamespace(n=n, m=m, c=c, d=d, f=f)

def generateFeatures(n, m, instances, feature_size=32):
    labels = torch.Tensor([instance.c for instance in instances])
    feature_net = FeatureNN(input_shape=(n,m), output_shape=(n,feature_size))
    feature_net.eval()
    features = feature_net(labels).detach()
    return features, labels

def generateDataset(n, m, num_instances, feature_size=32):
    instances = [generateInstance(n,m) for i in range(num_instances)]
    features, labels = generateFeatures(n, m, instances, feature_size=feature_size)

    train_size    = int(np.floor(num_instances * 0.7))
    validate_size = int(np.floor(num_instances * 0.1)) 
    test_size     = num_instances - train_size - validate_size

    entire_dataset = data_utils.TensorDataset(features, labels)

    indices = list(range(num_instances))
    np.random.shuffle(indices)

    train_indices    = indices[:train_size]
    validate_indices = indices[train_size:train_size+validate_size]
    test_indices     = indices[train_size+validate_size:]

    batch_size = 1
    train_loader    = data_utils.DataLoader(entire_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    validate_loader = data_utils.DataLoader(entire_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validate_indices))
    test_loader     = data_utils.DataLoader(entire_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    return SimpleNamespace(train=train_loader, test=test_loader, validate=validate_loader)

def MILPSolver(instance):
    model = Model()
    model.params.OutputFlag=0
    model.params.TuneOutput=0
    n, m, c, d, f = instance.n, instance.m, instance.c, instance.d, instance.f

    # adding variables
    x = model.addVars(n, vtype=GRB.BINARY)
    z = model.addVars(n, m, vtype=GRB.BINARY)

    # adding constraints
    for j in range(m):
        model.addConstr(z.sum('*', j) == 1, name='customer{}'.format(j))
        for i in range(n):
            model.addConstr(z[i,j] <= x[i], name='supply{},{}'.format(i,j))

    obj1 = quicksum(c[i,j] * d[j] * z[i,j] for i in range(n) for j in range(m))
    obj2 = quicksum(f[i] * x[i] for i in range(n))
    obj = obj1 + obj2
    model.setObjective(obj, GRB.MINIMIZE)

    model.update()
    model.write("milp.lp")
    model.optimize()
    x_values = np.array([x[i].x for i in range(n)])
    z_values = np.array([[z[i,j].x for j in range(m)] for i in range(n)])
    obj_value = model.objVal

    print(x_values, z_values)
    print('optimal:', obj_value)
    return SimpleNamespace(x=x_values, z=z_values, obj=obj_value)

def LPSolver(instance):
    model = Model()
    model.params.OutputFlag=0
    model.params.TuneOutput=0
    n, m, c, d, f = instance.n, instance.m, instance.c, instance.d, instance.f

    # Adding variables
    x = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    z = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)

    # Adding constraints
    for j in range(m):
        model.addConstr(z.sum('*', j) == 1, name='customer{}'.format(j))

    # Better formulation
    for i in range(n):
        for j in range(m):
            model.addConstr(z[i,j] <= x[i], name='supply{},{}'.format(i,j))

    # Loosen formulation
    # M = 1000
    # for i in range(n):
    #     model.addConstr(z.sum(i, '*') <= M * x[i], name='supply{},{}'.format(i,j))

    obj1 = quicksum(c[i,j] * d[j] * z[i,j] for i in range(n) for j in range(m))
    obj2 = quicksum(f[i] * x[i] for i in range(n))
    obj = obj1 + obj2
    model.setObjective(obj, GRB.MINIMIZE)

    model.update()
    model.write("lp.lp")
    model.optimize()
    x_values = np.array([x[i].x for i in range(n)])
    z_values = np.array([[z[i,j].x for j in range(m)] for i in range(n)])
    obj_value = model.objVal

    return SimpleNamespace(x=x_values, z=z_values, obj=obj_value)

def createConstraintMatrix(m, n, budget):
    variable_size = n
    A = torch.ones(1,n)
    b = torch.Tensor([budget])
    G = -torch.eye(n) 
    h = torch.zeros(n) 
    # G = torch.cat((-torch.eye(n),  torch.eye(n)))
    # h = torch.cat((torch.zeros(n), torch.ones(n)))
    return A, b, G, h

def createSurrogateConstraintMatrix(m, n, budget):
    variable_size = n
    A = torch.ones(1,n)
    b = torch.Tensor([budget])
    G = torch.eye(n) 
    h = torch.ones(n)
    return A, b, G, h

def train_submodular(net, optimizer, epoch, sample_instance, dataset, lr=0.1, training_method='two-stage', device='cpu', evaluate=True):
    net.train()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)
    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    REG = 0.0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            net_start_time = time.time()
            features, labels = features.to(device), labels.to(device)
            if epoch >= 0:
                outputs = net(features)
            else:
                outputs = labels
            # two-stage loss
            loss = loss_fn(outputs, labels)
            forward_time += time.time() - net_start_time

            # decision-focused loss
            objective_value_list = []
            batch_size = len(labels)
            for (label, output) in zip(labels, outputs):
                forward_start_time = time.time()
                if training_method == 'decision-focused':
                    inference_start_time = time.time()
                    min_fun = -np.inf
                    for _ in range(1):
                        tmp_result = getOptimalDecision(n, m, output, d, f, budget=budget, REG=REG)
                        if tmp_result.fun > min_fun:
                            optimize_result = tmp_result
                            min_fun = tmp_result.fun
                    inference_time += time.time() - inference_start_time
                    optimal_x = torch.Tensor(optimize_result.x)
                    if optimize_result.success:
                        qp_start_time = time.time()
                        newA, newb = torch.Tensor(), torch.Tensor()
                        newG = torch.cat((A, G))
                        newh = torch.cat((b, h))

                        Q = getHessian(optimal_x, n, m, output, d, f, REG=REG) + torch.eye(n) * 10
                        L = torch.cholesky(Q)
                        jac = -getDerivative(optimal_x, n, m, output, d, f, create_graph=True, REG=REG)
                        p = jac - Q @ optimal_x
                        # qp_solver = qpth.qp.QPFunction()
                        # x = qp_solver(Q, p, G, h, A, b)[0]

                        if True:
                            # =============== solving QP using CVXPY ===============
                            x_default = cp.Variable(n)
                            G_default, h_default = cp.Parameter(newG.shape), cp.Parameter(newh.shape)
                            L_default = cp.Parameter((n,n))
                            p_default = cp.Parameter(n)
                            constraints = [G_default @ x_default <= h_default]
                            objective = cp.Minimize(0.5 * cp.sum_squares(L_default @ x_default) + p_default.T @ x_default)
                            problem = cp.Problem(objective, constraints)

                            cvxpylayer = CvxpyLayer(problem, parameters=[G_default, h_default, L_default, p_default], variables=[x_default])
                            coverage_qp_solution, = cvxpylayer(newG, newh, L, p)
                            x = coverage_qp_solution

                        # except:
                        #     print("CVXPY solver fails... Usually because Q is not PSD")
                        #     x = optimal_x


                    else: 
                        print('Optimization failed...')
                        x = optimal_x
                    obj = getObjective(x, n, m, label, d, f, REG=0)
                    qp_time += time.time() - qp_start_time

                elif training_method == 'two-stage':
                    if evaluate:
                        inference_start_time = time.time()
                        optimize_result = getOptimalDecision(n, m, output, d, f, budget=budget, REG=REG)
                        x = torch.Tensor(optimize_result.x)
                        obj = getObjective(x, n, m, label, d, f, REG=0)
                        inference_time += time.time() - inference_start_time
                        qp_time = 0
                    else:
                        obj = torch.Tensor([0])
                        qp_time = 0
                else:
                    raise ValueError('Not implemented method!')

                objective_value_list.append(obj)
            objective = sum(objective_value_list) / batch_size

            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'two-stage':
                    loss.backward()
                elif training_method == 'decision-focused':
                    # (-objective).backward()
                    (-objective * 0.5 + loss * 0.5).backward() # TODO
                    for parameter in net.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("no grad is backpropagated...")
                pass
            optimizer.step()
            backward_time += time.time() - backward_start_time

            train_losses.append(loss.item())
            train_objs.append(objective.item())

            average_loss = np.mean(train_losses)
            average_obj = np.mean(train_objs)
            # Print status
            tqdm_loader.set_postfix(loss=f'{average_loss:.6f}', obj=f'{average_obj:.6f}')

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def surrogate_train_submodular(net, init_T, optimizer, T_optimizer, epoch, sample_instance, dataset, lr=0.1, training_method='two-stage', device='cpu'):
    net.train()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    train_losses, train_objs, train_T_losses = [], [], []
    x_size, variable_size = init_T.shape
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createSurrogateConstraintMatrix(m, n, budget)
    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, labels = features.to(device), labels.to(device)
            if epoch >= 0: 
                outputs = net(features)
            else:
                outputs = labels
            # two-stage loss
            loss = loss_fn(outputs, labels)
            forward_time += time.time() - forward_start_time

            # decision-focused loss
            objective_value_list, T_loss_list = [], []
            batch_size = len(labels)

            # randomly select column to update
            T = init_T
            # T = init_T.detach().clone()
            # random_column = torch.randint(init_T.shape[1], [1])
            # T[:,random_column] = init_T[:,random_column]

            # if batch_idx == 0:
            #     plot_graph(labels.detach().numpy(), T.detach().numpy(), epoch)

            for (label, output) in zip(labels, outputs):
                if training_method == 'surrogate':
                    # output = label # for debug only # TODO
                    inference_start_time = time.time()
                    optimize_result = getSurrogateOptimalDecision(T, n, m, output, d, f, budget=budget) # end-to-end for both T and net
                    inference_time += time.time() - inference_start_time
                    optimal_y = torch.Tensor(optimize_result.x)

                    if optimize_result.success:
                        optimal_y = torch.Tensor(optimize_result.x)
                        newA, newb = torch.Tensor(), torch.Tensor()
                        newG = torch.cat((A @ T, G @ T, -torch.eye(variable_size)))
                        newh = torch.cat((b, h, torch.zeros(variable_size)))
                        # newG = torch.cat((A @ T, G @ T, -torch.eye(variable_size), torch.eye(variable_size)))
                        # newh = torch.cat((b, h, torch.zeros(variable_size), torch.ones(variable_size)))

                        qp_start_time = time.time()
                        Q = getSurrogateHessian(T, optimal_y, n, m, output, d, f).detach() + torch.eye(len(optimal_y)) * 10
                        L = torch.cholesky(Q)
                        jac = -getSurrogateDerivative(T, optimal_y, n, m, output, d, f, create_graph=True)
                        p = jac - Q @ optimal_y
                        # qp_solver = qpth.qp.QPFunction() # TODO unknown bug

                        # try:
                        #     y = qp_solver(Q, p, newG, newh, newA, newb)[0]
                        #     x = T @ y
                        # except:
                        #     y = optimal_y
                        #     x = T.detach() @ optimal_y
                        #     print('qp error! no gradient!')

                        if True:
                            # =============== solving QP using CVXPY ===============
                            y_default = cp.Variable(variable_size)
                            G_default, h_default = cp.Parameter(newG.shape), cp.Parameter(newh.shape)
                            L_default = cp.Parameter((variable_size, variable_size))
                            p_default = cp.Parameter(variable_size)
                            constraints = [G_default @ y_default <= h_default]
                            objective = cp.Minimize(0.5 * cp.sum_squares(L_default @ y_default) + p_default.T @ y_default)
                            problem = cp.Problem(objective, constraints)

                            cvxpylayer = CvxpyLayer(problem, parameters=[G_default, h_default, L_default, p_default], variables=[y_default])
                            coverage_qp_solution, = cvxpylayer(newG, newh, L, p)
                            y = coverage_qp_solution
                            x = T @ y

                        # time test...
                        # time_test_start = time.time()
                        # for i in range(20):
                        #    _ = getDerivative(x, n, m, output, d, f)
                        # print('original gradient time:', time.time() - time_test_start)

                        # time_test_start = time.time()
                        # for i in range(20):
                        #    _ = getSurrogateDerivative(T, y, n, m, output, d, f, create_graph=False)
                        # print('surrogate gradient time:', time.time() - time_test_start)

                        # except:
                        #     print("CVXPY solver fails... Usually because Q is not PSD")
                        #     y = optimal_y
                        #     x = T.detach() @ optimal_y

                    else: # torch.norm(y.detach() - optimal_y) > 0.05: # TODO
                        print('Optimization failed...')
                        y = optimal_y
                        x = T.detach() @ optimal_y

                    qp_time += time.time() - qp_start_time
                else:
                    raise ValueError('Not implemented method!')

                obj = getObjective(x, n, m, label, d, f)
                tmp_T_loss = 0 # torch.sum((projected_real_optimal_x - real_optimal_x) ** 2).item()
                
                objective_value_list.append(obj)
                T_loss_list.append(tmp_T_loss)

            # print(pairwise_distances(T.t().detach().numpy()))
            objective  = sum(objective_value_list) / batch_size
            T_loss     = torch.sum(T.t() @ T - torch.diag(T.t() @ T)) # sum(T_loss_list) / batch_size
            # print('objective', objective)

            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'two-stage':
                    loss.backward()
                    optimizer.step()
                elif training_method == 'decision-focused':
                    (-objective).backward()
                    for parameter in net.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    optimizer.step()
                elif training_method == 'surrogate':
                    T_optimizer.zero_grad()
                    (-objective - T_loss).backward()
                    # T_loss.backward() # TODO: minimizing reparameterization loss

                    for parameter in net.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-MAX_NORM, max=MAX_NORM)
                    init_T.grad = torch.clamp(init_T.grad, min=-MAX_NORM, max=MAX_NORM)
                    optimizer.step()
                    T_optimizer.step()
                    init_T.data = normalize_matrix_positive(init_T.data)
                else:
                    raise ValueError('Not implemented method')
            except:
                print("Error! No grad is backpropagated...")
                pass
            backward_time += time.time() - backward_start_time

            train_losses.append(loss.item())
            train_objs.append(objective.item())
            train_T_losses.append(T_loss)

            average_loss   = np.mean(train_losses)
            average_obj    = np.mean(train_objs)
            average_T_loss = np.mean(train_T_losses)
            # Print status
            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}')
            # tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}', T_loss=f'{average_T_loss:.3f}')

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

def validate_submodular(net, scheduler, epoch, sample_instance, dataset, training_method='two-stage', device='cpu', evaluate=True):
    net.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            features, labels = features.to(device), labels.to(device)
            if epoch >= 0:
                outputs = net(features)
            else:
                outputs = labels
            # two-stage loss
            loss = loss_fn(outputs, labels)

            # decision-focused loss
            objective_value_list = []
            batch_size = len(labels)
            for (label, output) in zip(labels, outputs):
                if evaluate:
                    optimize_result = getOptimalDecision(n, m, output, d, f, budget=budget)
                    optimal_x = torch.Tensor(optimize_result.x)
                    obj = getObjective(optimal_x, n, m, label, d, f)
                else:
                    obj = torch.Tensor([0])
                objective_value_list.append(obj)
            objective = sum(objective_value_list) / batch_size

            test_losses.append(loss.item())
            test_objs.append(objective.item())

            average_loss = np.mean(test_losses)
            average_obj  = np.mean(test_objs)

            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)

    if (epoch > 0):
        if training_method == "two-stage":
            scheduler.step(average_loss)
        elif training_method == "decision-focused" or training_method == "surrogate-decision-focused":
            scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj

def surrogate_validate_submodular(net, scheduler, T_scheduler, T, epoch, sample_instance, dataset, training_method='two-stage', device='cpu'):
    net.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            features, labels = features.to(device), labels.to(device)
            if epoch >= 0:
                outputs = net(features)
            else:
                outputs = labels
            # two-stage loss
            loss = loss_fn(outputs, labels)

            # decision-focused loss
            objective_value_list = []
            batch_size = len(labels)
            for (label, output) in zip(labels, outputs):
                optimize_result = getSurrogateOptimalDecision(T.detach(), n, m, output, d, f, budget=budget)
                optimal_y = torch.Tensor(optimize_result.x)
                obj = getSurrogateObjective(T.detach(), optimal_y, n, m, label, d, f)
                objective_value_list.append(obj)
            objective = sum(objective_value_list) / batch_size

            test_losses.append(loss.item())
            test_objs.append(objective.item())

            average_loss = np.mean(test_losses)
            average_obj  = np.mean(test_objs)

            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)

    if (epoch > 0):
        if training_method == "surrogate":
            scheduler.step(-average_obj)
            T_scheduler.step(-average_obj)
        else:
            raise TypeError("Not Implemented Method")

    return average_loss, average_obj

def test_submodular(net, epoch, sample_instance, dataset, device='cpu', evaluate=True):
    net.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            features, labels = features.to(device), labels.to(device)
            if epoch >= 0:
                outputs = net(features)
            else:
                outputs = labels

            # two-stage loss
            loss = loss_fn(outputs, labels)

            # decision-focused loss
            objective_value_list = []
            batch_size = len(labels)
            for (label, output) in zip(labels, outputs):
                if evaluate:
                    optimize_result = getOptimalDecision(n, m, output, d, f, budget=budget)
                    optimal_x = torch.Tensor(optimize_result.x)
                    obj = getObjective(optimal_x, n, m, label, d, f)
                else:
                    obj = torch.Tensor([0])
                objective_value_list.append(obj)
            objective = sum(objective_value_list) / batch_size

            test_losses.append(loss.item())
            test_objs.append(objective.item())

            average_loss = np.mean(test_losses)
            average_obj  = np.mean(test_objs)

            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj

def surrogate_test_submodular(net, T, epoch, sample_instance, dataset, device='cpu'):
    net.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            features, labels = features.to(device), labels.to(device)
            if epoch >= 0:
                outputs = net(features)
            else:
                outputs = labels
            # two-stage loss
            loss = loss_fn(outputs, labels)

            # decision-focused loss
            objective_value_list = []
            batch_size = len(labels)
            for (label, output) in zip(labels, outputs):
                optimize_result = getSurrogateOptimalDecision(T.detach(), n, m, output, d, f, budget=budget)
                optimal_y = torch.Tensor(optimize_result.x)
                obj = getSurrogateObjective(T.detach(), optimal_y, n, m, label, d, f)
                objective_value_list.append(obj)
            objective = sum(objective_value_list) / batch_size

            test_losses.append(loss.item())
            test_objs.append(objective.item())

            average_loss = np.mean(test_losses)
            average_obj  = np.mean(test_objs)

            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}')

    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj

