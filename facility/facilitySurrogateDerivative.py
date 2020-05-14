import torch
import numpy as np
import scipy
import autograd
from facilityDerivative import getObjective

REG = 0.1

def getSurrogateObjective(T, y, n, m, c, d, f):
    x = T @ y
    p_value = getObjective(x, n, m, c, d, f)
    return p_value

def getSurrogateManualDerivative(T, y, n, m, c, d, f):
    y_var = y.detach().requires_grad_(True)
    obj = getSurrogateObjective(T, y_var, n, m, c, d, f)
    y_grad = torch.autograd.grad(obj, y_var, retain_graph=True, create_graph=True)[0]
    return y_grad

def getSurrogateDerivative(T, y, n, m, c, d, f):
    y_var = y.detach().requires_grad_(True)
    obj = getSurrogateObjective(T.detach(), y_var, n, m, c, d, f)
    y_grad = torch.autograd.grad(obj, y_var)[0]
    # y_grad = torch.autograd.grad(obj, y_var, retain_graph=True)[0]
    return y_grad

def getSurrogateOptimalDecision(T, n, m, c, d, f, budget, initial_y=None):
    variable_size = T.shape[1]
    if initial_y is None:
        initial_y = np.zeros(variable_size)
        # initial_y = np.random.rand(variable_size)
        # initial_y = initial_y * budget / np.sum(initial_y)

    getObj = lambda y: -getSurrogateObjective(T.detach(), torch.Tensor(y), n, m, c.detach(), d, f).detach().item() / 100 # maximize objective
    getJac = lambda y: -getSurrogateDerivative(T.detach(), torch.Tensor(y), n, m, c.detach(), d, f).detach().numpy() / 100

    bounds = [(0,np.inf)]*variable_size
    eq_fn    = lambda y: budget - sum( T.detach().numpy() @ y)
    ineq_fn  = lambda y: T.detach().numpy() @ y
    ineq_fn2 = lambda y: 1 - T.detach().numpy() @ y
    # constraints = [{'type': 'eq', 'fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}]
    constraints = [
            {'type': 'eq', 'fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}, 
            {'type': 'ineq', 'fun': ineq_fn, 'jac': autograd.jacobian(ineq_fn)},
            {'type': 'ineq', 'fun': ineq_fn2, 'jac': autograd.jacobian(ineq_fn2)}
            ]
    options = {'maxiter': 100, 'eps':1e-12, 'disp': True}
    # options = {'maxiter': 100, 'disp': True}
    # tol = 1e-6

    optimize_result = scipy.optimize.minimize(getObj, initial_y, method='SLSQP', jac=getJac, constraints=constraints, options=options, bounds=bounds)
    # optimize_result = scipy.optimize.minimize(getObj, initial_y, method='trust-constr', jac=getJac, constraints=constraints, options=options, bounds=bounds, tol=tol)

    return optimize_result

def getSurrogateHessian(T, y, n, m, c, d, f):
    return REG * T.t() @ T

