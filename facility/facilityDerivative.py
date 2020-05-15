import torch
import numpy as np
import scipy
import autograd

REG = 0.0

def getObjective(x, n, m, c, d, f):
    p = torch.zeros(m)
    for j in range(m):
        # each customer selects the top d_j items
        preference_ordering = torch.argsort(c[:,j], descending=True)
        remaining = float(d[j])
        selected_amount = torch.zeros(n)
        for i in preference_ordering:
            if remaining - x[i] < 0:
                selected_amount[i] = remaining
                break
            else:
                selected_amount[i] = x[i]
                remaining -= x[i]
        p[j] = selected_amount @ c[:,j]

    p_value = torch.sum(p) - 0.5 * REG * x @ x
    return p_value


def getManualDerivative(x, n, m, c, d, f):
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

def getDerivative(x, n, m, c, d, f, create_graph=False):
    x_var = x.detach().requires_grad_(True)
    obj = getObjective(x_var, n, m, c, d, f)
    x_grad = torch.autograd.grad(obj, x_var, retain_graph=True, create_graph=create_graph)[0] # TODO!! allow_unused is not known
    return x_grad

def getOptimalDecision(n, m, c, d, f, budget, initial_x=None):
    if initial_x is None:
        initial_x = np.zeros(n)
        # initial_x = np.random.rand(n)
        # initial_x = initial_x * budget / np.sum(initial_x)

    getObj = lambda x: -getObjective(torch.Tensor(x), n, m, c.detach(), d, f).detach().item() # maximize objective
    getJac = lambda x: -getManualDerivative(torch.Tensor(x), n, m, c.detach(), d, f).detach().numpy()

    bounds = [(0,1)]*n
    eq_fn = lambda x: budget - sum(x)
    constraints = [{'type': 'ineq', 'fun': eq_fn, 'jac': autograd.jacobian(eq_fn)}]
    options = {'maxiter': 100, 'ftol': 1e-4}

    optimize_result = scipy.optimize.minimize(getObj, initial_x, method='SLSQP', jac=getJac, bounds=bounds, constraints=constraints, options=options)
    # optimize_result = scipy.optimize.minimize(getObj, initial_x, method='trust-constr', jac=getJac, bounds=bounds, constraints=constraints)

    return optimize_result

def getHessian(x, n, m, c, d, f):
    return torch.eye(n) * REG

