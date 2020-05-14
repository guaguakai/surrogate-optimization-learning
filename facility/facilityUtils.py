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
from facilityDerivative import getObjective, getManualDerivative, getDerivative, getOptimalDecision, getHessian
from facilitySurrogateDerivative import getSurrogateObjective, getSurrogateDerivative, getSurrogateManualDerivative, getSurrogateHessian, getSurrogateOptimalDecision

# Random Seed Initialization
# SEED = 1289 #  random.randint(0,10000)
# print("Random seed: {}".format(SEED))
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

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
    G = torch.cat((-torch.eye(n),  torch.eye(n)))
    h = torch.cat((torch.zeros(n), torch.ones(n)))

    return A, b, G, h

def LPCreateConstraintMatrix(m, n):
    # min  1/2 x^T Q x + x^T p
    # s.t. A x =  b
    #      G x <= h
    variable_size = n + n * m
    A = torch.cat((torch.eye(m).repeat(1,n), torch.zeros(m,n)), axis=1)
    A = torch.Tensor(A)
    b = torch.ones(m)
    G = torch.cat((torch.cat((torch.eye(n*m), torch.repeat_interleave(-torch.eye(n), repeats=m, dim=0)), axis=1), -torch.eye(n*m + n), torch.eye(n*m + n) ), axis=0)
    h = torch.cat((torch.zeros(n*m), torch.zeros(n*m+n), torch.ones(n*m+n)))

    return A, b, G, h

def LPCreateSurrogateConstraintMatrix(m, n):
    # min  1/2 x^T Q x + x^T p
    # s.t. A x =  b
    #      G x <= h
    variable_size = n + n * m
    A = torch.cat((torch.eye(m).repeat(1,n), torch.zeros(m,n)), axis=1)
    A = torch.Tensor(A)
    b = torch.ones(m)
    G = torch.cat((torch.eye(n*m), torch.repeat_interleave(-torch.eye(n), repeats=m, dim=0)), axis=1)
    h = torch.zeros(n*m)

    return A, b, G, h

def train_submodular(net, optimizer, epoch, sample_instance, dataset, lr=0.1, training_method='two-stage', device='cpu'):
    net.train()
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()
    train_losses, train_objs = [], []
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)
    forward_time, qp_time, backward_time = 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, labels) in enumerate(tqdm_loader):
            net_start_time = time.time()
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            # two-stage loss
            loss = loss_fn(outputs, labels)
            forward_time += time.time() - net_start_time

            # decision-focused loss
            objective_value_list = []
            batch_size = len(labels)
            for (label, output) in zip(labels, outputs):
                forward_start_time = time.time()
                optimize_result = getOptimalDecision(n, m, output, d, f, budget=budget)
                if training_method == 'decision-focused':
                    forward_time += time.time() - forward_start_time
                optimal_x = torch.Tensor(optimize_result.x).requires_grad_(True)

                qp_start_time = time.time()
                if training_method == 'decision-focused':
                    newA, newb = torch.Tensor(), torch.Tensor()
                    newG = torch.cat((A, G))
                    newh = torch.cat((b, h))

                    Q = getHessian(optimal_x, n, m, output, d, f) + torch.eye(n) * 10
                    jac = -getManualDerivative(optimal_x, n, m, output, d, f)
                    p = jac - Q @ optimal_x
                    qp_solver = qpth.qp.QPFunction()
                    # qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI)
                    x = qp_solver(Q, p, newG, newh, newA, newb)[0]
                    if torch.norm(x.detach() - optimal_x) > 0.5:
                        # debugging message
                        print('incorrect solution due to high mismatch {}'.format(torch.norm(x.detach() - optimal_x)))
                        # print('optimal x:', optimal_x)
                        # print('x:        ', x)
                        # scipy_obj = 0.5 * optimal_x @ Q @ optimal_x + p @ optimal_x 
                        # scipy_obj = getObjective(optimal_x, n, m, output, d, f)
                        # qp_obj    = 0.5 * x @ Q @ x + p @ x 
                        # qp_obj = getObjective(x, n, m, output, d, f)
                        # print('objective values scipy: {}, QP: {}'.format(scipy_obj, qp_obj))
                        # print('constraint on optimal_x: Ax-b={}, Gx-h={}'.format(A @ optimal_x - b, G @ optimal_x - h))
                        # print('constraint on x:         Ax-b={}, Gx-h={}'.format(A @ x - b, G @ x - h))
                        x = optimal_x
                    qp_time += time.time() - qp_start_time
                elif training_method == 'two-stage':
                    x = optimal_x
                else:
                    raise ValueError('Not implemented method!')
                qp_time += time.time() - qp_start_time

                obj = getObjective(x, n, m, label, d, f)

                objective_value_list.append(obj)
            objective = sum(objective_value_list) / batch_size

            optimizer.zero_grad()
            backward_start_time = time.time()
            try:
                if training_method == 'two-stage':
                    loss.backward()
                elif training_method == 'decision-focused':
                    (-objective).backward()
                    for parameter in net.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-0.01, max=0.01)
                else:
                    raise ValueError('Not implemented method')
            except:
                # print("no grad is backpropagated...")
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
    return average_loss, average_obj, (forward_time, qp_time, backward_time)

def surrogate_train_submodular(net, init_T, optimizer, T_optimizer, epoch, sample_instance, dataset, lr=0.1, training_method='two-stage', device='cpu'):
    net.train()
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()
    train_losses, train_objs, train_T_losses = [], [], []
    x_size, variable_size = init_T.shape
    n, m, d, f, budget = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f), sample_instance.budget
    A, b, G, h = createConstraintMatrix(m, n, budget)
    forward_time, qp_time, backward_time = 0, 0, 0

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
            objective_value_list, T_loss_list = [], []
            batch_size = len(labels)

            # randomly select column to update
            T = init_T
            # T = init_T.detach().clone()
            # random_column = torch.randint(init_T.shape[1], [1])
            # T[:,random_column] = init_T[:,random_column]

            for (label, output) in zip(labels, outputs):
                if training_method == 'surrogate':
                    # output = label # for debug only # TODO
                    forward_start_time = time.time()
                    optimize_result = getSurrogateOptimalDecision(T, n, m, output, d, f, budget=budget) # end-to-end for both T and net
                    forward_time += time.time() - forward_start_time
                    optimal_y = torch.Tensor(optimize_result.x)

                    newA, newb = torch.Tensor(), torch.Tensor()
                    newG = torch.cat((A @ T, G @ T, -torch.eye(variable_size)))
                    newh = torch.cat((b, h, torch.zeros(variable_size)))
                    # newG = torch.cat((G @ T, - T, T)) # torch.eye(variable_size)))
                    # newh = torch.cat((h, torch.zeros(x_size), torch.ones(x_size)))

                    qp_start_time = time.time()
                    Q = getSurrogateHessian(T, optimal_y, n, m, output, d, f).detach() + torch.eye(len(optimal_y)) * 10
                    jac = -getSurrogateManualDerivative(T, optimal_y, n, m, output, d, f)
                    p = jac - Q @ optimal_y
                    qp_solver = qpthlocal.qp.QPFunction() # TODO unknown bug
                    # qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI)
                    try:
                        y = qp_solver(Q, p, newG, newh, newA, newb)[0]
                        x = T @ y
                        if torch.norm(x.detach() - T.detach() @ optimal_y) > 0.05: # TODO
                            print('incorrect solution due to high mismatch {}'.format(torch.norm(x.detach() - T.detach() @ optimal_y)))
                            # print(x, T.detach() @ optimal_y)
                            # scipy_obj = getSurrogateObjective(T.detach(), optimal_y, n, m, output, d, f)
                            # qp_obj = getSurrogateObjective(T.detach(), y, n, m, output, d, f)
                            # print('objective values scipy: {}, QP: {}'.format(scipy_obj, qp_obj))
                            y = optimal_y
                            x = T.detach() @ optimal_y
                    except:
                        y = optimal_y
                        x = T.detach() @ optimal_y
                        print('qpth error! no gradient!')
                    qp_time += time.time() - qp_start_time
                else:
                    raise ValueError('Not implemented method!')

                obj = getObjective(x, n, m, label, d, f)
                tmp_T_loss = 0 # torch.sum((projected_real_optimal_x - real_optimal_x) ** 2).item()
                
                objective_value_list.append(obj)
                T_loss_list.append(tmp_T_loss)

            # print(pairwise_distances(T.t().detach().numpy()))
            objective  = sum(objective_value_list) / batch_size
            T_loss     = sum(T_loss_list) / batch_size
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
                        parameter.grad = torch.clamp(parameter.grad, min=-0.01, max=0.01)
                    optimizer.step()
                elif training_method == 'surrogate':
                    T_optimizer.zero_grad()
                    (-objective).backward()
                    # T_loss.backward() # TODO: minimizing reparameterization loss

                    for parameter in net.parameters():
                        parameter.grad = torch.clamp(parameter.grad, min=-0.01, max=0.01)
                    init_T.grad = torch.clamp(init_T.grad, min=-0.01, max=0.01)
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
            tqdm_loader.set_postfix(loss=f'{average_loss:.3f}', obj=f'{average_obj:.3f}', T_loss=f'{average_T_loss:.3f}')

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, qp_time, backward_time)

def validate_submodular(net, scheduler, epoch, sample_instance, dataset, training_method='two-stage', device='cpu'):
    net.eval()
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()
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
                optimize_result = getOptimalDecision(n, m, output, d, f, budget=budget)
                optimal_x = torch.Tensor(optimize_result.x)
                obj = getObjective(optimal_x, n, m, label, d, f)
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
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()
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

def test_submodular(net, epoch, sample_instance, dataset, device='cpu'):
    net.eval()
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()
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
                optimize_result = getOptimalDecision(n, m, output, d, f, budget=budget)
                optimal_x = torch.Tensor(optimize_result.x)
                obj = getObjective(optimal_x, n, m, label, d, f)
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
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()
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

def train_LP(net, optimizer, epoch, sample_instance, dataset, lr=0.1, training_method='two-stage', device='cpu'):
    # train a single epoch
    net.train()
    loss_fn = torch.nn.BCELoss()
    train_losses, train_objs, train_optimals = [], [], []
    n, m, d, f = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f)
    A, b, G, h = LPCreateConstraintMatrix(m, n)

    for batch_idx, (features, labels) in enumerate(tqdm.tqdm(dataset)):
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)
        # two-stage loss
        loss = loss_fn(outputs, labels)

        # decision-focused loss
        objective_value, optimal_value = 0, 0
        batch_size = len(features)
        variable_size = n * m + n
        Q = torch.eye(variable_size) * 0.01
        for (label, output) in zip(labels, outputs):
            p = torch.cat(((output * d).flatten(), f))
            qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI)
            zx = qp_solver(Q, p, G, h, A, b)

            real_p = torch.cat((torch.Tensor(label * d).flatten(), torch.Tensor(f)))
            obj = zx @ real_p
            result = LPSolver(SimpleNamespace(n=n, m=m, c=label.detach().numpy(), d=sample_instance.d, f=sample_instance.f))
            objective_value += obj
            optimal_value += result.obj
        objective = objective_value / batch_size
        optimal = optimal_value / batch_size

        optimizer.zero_grad()
        if training_method == 'two-stage':
            loss.backward()
        elif training_method == 'decision-focused':
            objective.backward()
            for parameter in net.parameters():
                parameter.grad = torch.clamp(parameter.grad, min=-0.01, max=0.01)
        else:
            raise ValueError('Not implemented method')
        optimizer.step()

        train_losses.append(loss.item())
        train_objs.append(obj.item())
        train_optimals.append(optimal)

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    average_optimal = np.mean(train_optimals) 
    return average_loss, average_obj, average_optimal

def surrogate_train_LP(net, optimizer, epoch, sample_instance, dataset, lr=0.1, training_method='two-stage', device='cpu'):
    # train a single epoch
    net.train()
    loss_fn = torch.nn.BCELoss()
    train_losses, train_objs, train_optimals = [], [], []
    n, m, d, f = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f)
    A, b, G, h = LPCreateConstraintMatrix(m, n)

    for batch_idx, (features, labels) in enumerate(tqdm.tqdm(dataset)):
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)
        # two-stage loss
        loss = loss_fn(outputs, labels)

        # decision-focused loss
        objective_value, optimal_value = 0, 0
        batch_size = len(features)
        variable_size = T_size
        Q = torch.eye(T_size) * 0.01 # + 0.01 * (T.detach().t() @ T.detach())
        for (label, output) in zip(labels, outputs):
            p = torch.cat(((output * d).flatten(), f)) @ T
            # qp_solver = qpth.qp.QPFunction()
            qp_solver = qpthlocal.qp.QPFunction(verbose=True, solver=qpthlocal.qp.QPSolvers.GUROBI)
            zx = T @ qp_solver(Q, p, G @ T, h, new_A, new_b)[0]

            real_p = torch.cat((torch.Tensor(label * d).flatten(), torch.Tensor(f)))
            obj = zx @ real_p
            result = LPSolver(SimpleNamespace(n=n, m=m, c=label.detach().numpy(), d=sample_instance.d, f=sample_instance.f))
            objective_value += obj
            optimal_value += result.obj
        objective = objective_value / batch_size
        optimal = optimal_value / batch_size

        optimizer.zero_grad()
        T_optimizer.zero_grad()
        if training_method == 'two-stage':
            loss.backward()
        elif training_method == 'decision-focused' or training_method == 'surrogate':
            objective.backward()
            for parameter in net.parameters():
                parameter.grad = torch.clamp(parameter.grad, min=-0.01, max=0.01)
        else:
            raise ValueError('Not implemented method')
        optimizer.step()
        T_optimizer.step()
        T.data = normalize_projection(T.data, A, b[:,None])

        train_losses.append(loss.item())
        train_objs.append(obj.item())
        train_optimals.append(optimal)

    average_loss    = np.mean(train_losses)
    average_obj     = np.mean(train_objs)
    average_optimal = np.mean(train_optimals) 
    return average_loss, average_obj, average_optimal

def test_LP(net, optimizer, epoch, sample_instance, dataset, device='cpu'):
    # test a single epoch
    net.eval()
    loss_fn = torch.nn.BCELoss()
    test_losses, test_objs, test_optimals = [], [], []
    n, m, d, f = sample_instance.n, sample_instance.m, torch.Tensor(sample_instance.d), torch.Tensor(sample_instance.f)
    for batch_idx, (features, labels) in enumerate(dataset):
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)
        # two-stage loss
        loss = loss_fn(outputs, labels)
        test_losses.append(loss.item())
        # objective value
        objective_value, optimal_value = 0, 0
        batch_size = len(features)
        for (label, output) in zip(labels, outputs):
            result = LPSolver(SimpleNamespace(n=n, m=m, c=output.detach().numpy(), d=sample_instance.d, f=sample_instance.f))
            zx = torch.cat((torch.Tensor(result.z).flatten(), torch.Tensor(result.x)))
            real_p = torch.cat((torch.Tensor(label * d).flatten(), torch.Tensor(f)))
            obj = zx @ real_p
            objective_value += obj

            opt_result = LPSolver(SimpleNamespace(n=n, m=m, c=label.detach().numpy(), d=sample_instance.d, f=sample_instance.f))
            optimal_value += opt_result.obj
        objective = objective_value / batch_size
        optimal = optimal_value / batch_size

        test_losses.append(loss.item())
        test_objs.append(obj.item())
        test_optimals.append(optimal)
    average_loss    = np.mean(test_losses)
    average_obj     = np.mean(test_objs)
    average_optimal = np.mean(test_optimals) 
    sys.stdout.write(f'Epoch {epoch} | Test Loss: {average_loss:.3f}  | Test Objective Value: {average_obj:.3f}  | Test Optimal Value: {average_optimal:.3f} \n')
    sys.stdout.flush()
    return average_loss, average_obj, average_optimal

