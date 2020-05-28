import torch
import numpy as np
import scipy
import autograd
# from lml import LML
import qpth
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import time

# def getObjective(x_value, n, m, c, d, f, REG=0):
#     x_value = torch.clamp(x_value, max=1)
#     obj = 0
#     for j in range(m):
#         p = cp.Variable(n)
#         x = cp.Parameter(n)
#         constraints = [p >= 0, p <= x, cp.sum(p) <= d[j]]
#         objective = cp.Minimize(- c[:,j] @ x - cp.sum(cp.entr(x)))
#         problem = cp.Problem(objective, constraints)
# 
#         cvxpylayer = CvxpyLayer(problem, parameters=[x], variables=[p])
#         selection, = cvxpylayer(x_value)
#         obj += selection @ c[:,j]
# 
#     return obj

# def getObjective(x, n, m, c, d, f, REG=0):
#     x = torch.clamp(x, max=1)
#     p = torch.zeros(m)
#     softmax = torch.nn.Softmax()
#     for j in range(m):
#         floor = torch.min(softmax(c[:,j]), x)
#         # each customer selects the top d_j items
#         preference_ordering = torch.argsort(c[:,j], descending=True)
#         remaining = float(d[j]) - torch.sum(floor)
#         selected_amount = floor
#         for i in preference_ordering:
#             prob = x[i] - floor[i]
#             if remaining - prob <= 0:
#                 selected_amount[i] += remaining
#                 remaining = 0
#             else:
#                 selected_amount[i] += prob
#                 remaining -= prob
#         p[j] = selected_amount @ c[:,j]
# 
#     p_value = torch.sum(p) - 0.5 * REG * x @ x
#     return p_value

def binary_search(weigths, sums, budget, low, high):
    if high > low:
        mid = (high + low) // 2
    else:
        return -1


def getObjective(x_var, n, m, c, d, f, REG=0):
    start_time = time.time()
    x = torch.clamp(x_var, max=1)
    p = torch.zeros(m)
    for j in range(m):
        # each customer selects the top d_j items
        preference_ordering = torch.argsort(c[:,j], descending=True)
        remaining = float(d[j])
        selected_amount = torch.zeros(n)
        for idx, i in enumerate(preference_ordering):
            if remaining - x[i] <= 0:
                selected_amount[i] = remaining
                remaining = 0
            else:
                selected_amount[i] = x[i]
                remaining -= x[i]
        p[j] = selected_amount @ c[:,j]

    p_value = torch.sum(p) #- 0.5 * REG * torch.var(x)
    return p_value

def getOldManualDerivative(x_var, n, m, c, d, f, REG=0):
    x = torch.clamp(x_var, max=1)
    grad = torch.zeros(n)
    for j in range(m):
        # each customer selects the top d_j items
        preference_ordering = torch.argsort(c[:,j], descending=True)
        remaining = float(d[j])
        selection = []
        for i in preference_ordering:
            if remaining - x[i] < 0:
                for previous_i in selection:
                    grad[previous_i] -= c[i,j]
                break
            else:
                selection.append(i)
                remaining -= x[i]
                grad[i] += c[i,j]

    return grad - REG * x

def getDerivative(x, n, m, c, d, f, create_graph=False, REG=0):
    start_time = time.time()
    x_var = x.detach().requires_grad_(True)
    obj = getObjective(x_var, n, m, c, d, f, REG=REG)
    x_grad = torch.autograd.grad(obj, x_var, retain_graph=True, create_graph=create_graph)[0] # TODO!! allow_unused is not known
    return x_grad

def getOptimalDecision(n, m, c, d, f, budget, initial_x=None, REG=0):
    start_time = time.time()
    if initial_x is None:
        # initial_x = np.zeros(n)
        initial_x = np.random.rand(n)
        initial_x = initial_x * budget / np.sum(initial_x)

    getObj = lambda x: -getObjective(torch.Tensor(x), n, m, c.detach(), d, f, REG=REG).detach().item() # maximize objective
    getJac = lambda x: -getDerivative(torch.Tensor(x), n, m, c.detach(), d, f, REG=REG).detach().numpy()

    bounds = [(0,np.inf)]*n
    eq_fn = lambda x: budget - sum(x)
    constraints = [{'type': 'ineq', 'fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}]
    options = {'maxiter': 20, 'ftol': 1e-2, 'disp': False}
    # options = {'maxiter': 100, 'ftol': 1e-3}

    optimize_result = scipy.optimize.minimize(getObj, initial_x, method='SLSQP', jac=getJac, bounds=bounds, constraints=constraints, options=options)
    # optimize_result = scipy.optimize.minimize(getObj, initial_x, method='trust-constr', jac=getJac, bounds=bounds, constraints=constraints, options=options, hess=np.zeros((n,n)))

    return optimize_result

def getHessian(x, n, m, c, d, f, REG=0):
    return torch.eye(n) * REG

