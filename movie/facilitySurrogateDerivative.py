import torch
import numpy as np
import scipy
import autograd
from facilityDerivative import getObjective, getDerivative
import time

def getSurrogateObjective(T, y, n, m, c, d, f, REG=0):
    start_time = time.time()
    x = T @ y
    p_value = getObjective(x, n, m, c, d, f, REG=REG)
    return p_value

# def getSurrogateManualDerivative(T, y, n, m, c, d, f, REG=0):
#     y_var = y.detach().requires_grad_(True)
#     x_var = T @ y_var
#     x_grad = getManualDerivative(x_var, n, m, c, d, f, REG=REG)
#     y_grad = T.t() @ x_grad
#     return y_grad

def getSurrogateDerivative(T, y, n, m, c, d, f, REG=0, create_graph=False):
    start_time = time.time()
    y_var = y.detach().requires_grad_(True)
    obj = getSurrogateObjective(T, y_var, n, m, c, d, f, REG=REG)
    y_grad = torch.autograd.grad(obj, y_var, retain_graph=True, create_graph=create_graph)[0]

    # x = T @ y
    # x_grad = getDerivative(x, n, m, c, d, f, REG=REG, create_graph=create_graph)
    # y_grad = x_grad @ T

    return y_grad

def getSurrogateOptimalDecision(T, n, m, c, d, f, budget, initial_y=None, REG=0):
    start_time = time.time()
    variable_size = T.shape[1]
    if initial_y is None:
        # initial_x = torch.rand(n)
        # initial_x = initial_x * budget / torch.sum(initial_x)
        # initial_y = (torch.pinverse(T) @ initial_x).detach().numpy()

        initial_y = np.zeros(variable_size)
        # initial_y = np.random.rand(variable_size)
        # initial_y = initial_y * budget / np.sum(initial_y)

    getObj = lambda y: -getSurrogateObjective(T.detach(), torch.Tensor(y), n, m, c.detach(), d, f, REG=REG).detach().item() # maximize objective
    getJac = lambda y: -getSurrogateDerivative(T.detach(), torch.Tensor(y), n, m, c.detach(), d, f, REG=REG, create_graph=False).detach().numpy()

    bounds = [(0,np.inf)]*variable_size
    # eq_fn    = lambda y: budget - sum( y)
    eq_fn    = lambda y: budget - sum( T.detach().numpy() @ y)
    # ineq_fn  = lambda y: T.detach().numpy() @ y
    # ineq_fn2 = lambda y: 1 - T.detach().numpy() @ y
    # constraints = [{'type': 'eq', 'fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}]
    constraints = [
            {'type': 'ineq', 'fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}, 
            # {'type': 'ineq', 'fun': ineq_fn, 'jac': autograd.jacobian(ineq_fn)},
            # {'type': 'ineq', 'fun': ineq_fn2, 'jac': autograd.jacobian(ineq_fn2)}
            ]
    options = {'maxiter': 20, 'ftol': 1e-2, 'disp': False}
    # options = {'maxiter': 100, 'disp': True}
    # tol = 1e-6

    optimize_result = scipy.optimize.minimize(getObj, initial_y, method='SLSQP', jac=getJac, constraints=constraints, options=options, bounds=bounds)
    # optimize_result = scipy.optimize.minimize(getObj, initial_y, method='trust-constr', jac=getJac, constraints=constraints, options=options, bounds=bounds)

    return optimize_result

def getSurrogateHessian(T, y, n, m, c, d, f, REG=0):
    return REG * T.t() @ T

