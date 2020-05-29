import os
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

from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import SP500DataLoader
from portfolio_utils import computeCovariance, generateDataset
from model import PortfolioModel, CovarianceModel
from portfolio_utils import train_portfolio, surrogate_train_portfolio, validate_portfolio, surrogate_validate_portfolio, test_portfolio, surrogate_test_portfolio
from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection

if __name__ == '__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument('--filepath', type=str, default='', help='filename under folder results')
    parser.add_argument('--method', type=int, default=0, help='0: two-stage, 1: decision-focused, 2: surrogate-decision-focused')
    parser.add_argument('--T-size', type=int, default=10, help='the size of reparameterization metrix')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--n', type=int, default=50, help='number of items')
    parser.add_argument('--num-samples', type=int, default=0, help='number of samples, 0 -> all')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, help='random seed')

    args = parser.parse_args()

    SEED = args.seed #  random.randint(0,10000)
    print("Random seed: {}".format(SEED))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    portfolio_opt_dir = os.path.abspath(os.path.dirname(__file__))
    print("portfolio_opt_dir:", portfolio_opt_dir)

    sp500_data_dir = os.path.join(portfolio_opt_dir, "data", "sp500")
    sp500_data = SP500DataLoader(sp500_data_dir, "sp500",
                                 start_date=dt.datetime(2004, 1, 1),
                                 end_date=dt.datetime(2017, 1, 1),
                                 collapse="daily",
                                 overwrite=False,
                                 verbose=True)

    filepath = args.filepath
    seed = 0
    n = args.n
    num_samples = args.num_samples if args.num_samples != 0 else 4010
    num_epochs = args.epochs
    lr = args.lr
    method_id = args.method
    if method_id == 0:
        training_method = 'two-stage'
    elif method_id == 1:
        training_method = 'decision-focused'
    elif method_id == 2:
        training_method = 'surrogate'
    else:
        raise ValueError('Not implemented methods')
    print("Training method: {}".format(training_method))

    train_dataset, validate_dataset, test_dataset = generateDataset(sp500_data, n=n, num_samples=num_samples)
    feature_size = train_dataset.dataset[0][0].shape[1]

    model = PortfolioModel(input_size=feature_size, output_size=1)
    covariance_model = CovarianceModel(n=n)

    if training_method == 'surrogate':
        T_size = args.T_size
        init_T = normalize_matrix_positive(torch.rand(n, T_size))
        T = torch.tensor(init_T, requires_grad=True)
        # T_optimizer = torch.optim.Adam([T], lr=T_lr)
        # T_scheduler = ReduceLROnPlateau(T_optimizer, 'min')
        optimizer = torch.optim.Adam(list(model.parameters()) + list(covariance_model.parameters()) + [T], lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(covariance_model.parameters()), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

    train_loss_list, train_obj_list = [], []
    test_loss_list,  test_obj_list  = [], []
    validate_loss_list,  validate_obj_list = [], []

    print('n: {}, lr: {}'.format(n,lr))
    print('Start training...')
    evaluate = False if training_method == 'two-stage' else True
    total_forward_time, total_inference_time, total_qp_time, total_backward_time = 0, 0, 0, 0
    forward_time_list, inference_time_list, qp_time_list, backward_time_list = [], [], [], []
    for epoch in range(-1, num_epochs):
        if epoch == num_epochs - 1:
            evaluate = True
        start_time = time.time()
        forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
        if training_method == 'surrogate':
            if epoch == -1:
                print('Testing the optimal solution...')
                train_loss, train_obj = test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=True)
            elif epoch == 0:
                print('Testing the initial solution quality...')
                train_loss, train_obj = surrogate_test_portfolio(model, covariance_model, T.detach(), epoch, train_dataset, evaluate=evaluate)
            else:
                train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = surrogate_train_portfolio(model, covariance_model, T, optimizer, epoch, train_dataset, training_method=training_method)
        elif training_method == 'decision-focused' or training_method == 'two-stage':
            if epoch == -1:
                print('Testing the optimal solution...')
                train_loss, train_obj = test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=True)
            elif epoch == 0:
                print('Testing the initial solution quality...')
                train_loss, train_obj = test_portfolio(model, covariance_model, epoch, train_dataset, evaluate=evaluate)
            else:
                train_loss, train_obj, (forward_time, inference_time, qp_time, backward_time) = train_portfolio(model, covariance_model, optimizer, epoch, train_dataset, training_method=training_method, evaluate=evaluate)
        else:
            raise ValueError('Not implemented')
        total_forward_time   += forward_time
        total_inference_time += inference_time
        total_qp_time        += qp_time
        total_backward_time  += backward_time

        forward_time_list.append(forward_time)
        inference_time_list.append(inference_time)
        qp_time_list.append(qp_time)
        backward_time_list.append(backward_time)

        # ================ validating ==================
        if training_method == 'surrogate':
            if epoch == -1:
                validate_loss, validate_obj = test_portfolio(model, covariance_model, epoch, validate_dataset, evaluate=True)
            else:
                validate_loss, validate_obj = surrogate_validate_portfolio(model, covariance_model, T.detach(), scheduler, epoch, validate_dataset, training_method=training_method)
        else:
            if epoch == -1:
                validate_loss, validate_obj = test_portfolio(model, covariance_model, epoch, validate_dataset, evaluate=True)
            else:
                validate_loss, validate_obj = validate_portfolio(model, covariance_model, scheduler, epoch, validate_dataset, training_method=training_method, evaluate=evaluate)

        # ================== testing ===================
        if training_method == 'surrogate':
            if epoch == -1:
                test_loss, test_obj = test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=True)
            else:
                test_loss, test_obj = surrogate_test_portfolio(model, covariance_model, T.detach(), epoch, test_dataset)
        else:
            if epoch == -1:
                test_loss, test_obj = test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=True)
            else:
                test_loss, test_obj = test_portfolio(model, covariance_model, epoch, test_dataset, evaluate=evaluate)

        # =============== printing data ================
        sys.stdout.write(f'Epoch {epoch} | Train Loss:    \t {train_loss:.8f} \t | Train Objective Value:    \t {train_obj*100:.6f}% \n')
        sys.stdout.write(f'Epoch {epoch} | Validate Loss: \t {validate_loss:.8f} \t | Validate Objective Value: \t {validate_obj*100:.6f}% \n')
        sys.stdout.write(f'Epoch {epoch} | Test Loss:     \t {test_loss:.8f} \t | Test Objective Value:     \t {test_obj*100:.6f}% \n')
        sys.stdout.flush()

        # ============== recording data ================
        end_time = time.time()
        print("Epoch {}, elapsed time: {}, forward time: {}, inference time: {}, qp time: {}, backward time: {}".format(epoch, end_time - start_time, forward_time, inference_time, qp_time, backward_time))

        train_loss_list.append(train_loss)
        train_obj_list.append(train_obj)
        test_loss_list.append(test_loss)
        test_obj_list.append(test_obj)
        validate_loss_list.append(validate_loss)
        validate_obj_list.append(validate_obj)

        # record the data every epoch
        f_output = open('results/performance/' + filepath + "{}-SEED{}.csv".format(training_method,SEED), 'w')
        f_output.write('Epoch, {}\n'.format(epoch))
        f_output.write('training loss,' + ','.join([str(x) for x in train_loss_list]) + '\n')
        f_output.write('training obj,'  + ','.join([str(x) for x in train_obj_list])  + '\n')
        f_output.write('validating loss,' + ','.join([str(x) for x in validate_loss_list]) + '\n')
        f_output.write('validating obj,'  + ','.join([str(x) for x in validate_obj_list])  + '\n')
        f_output.write('testing loss,'  + ','.join([str(x) for x in test_loss_list])  + '\n')
        f_output.write('testing obj,'   + ','.join([str(x) for x in test_obj_list])   + '\n')
        f_output.close()

        f_time = open('results/time/' + filepath + "{}-SEED{}.csv".format(training_method, SEED), 'w')
        f_time.write('Epoch, {}\n'.format(epoch))
        f_time.write('Random seed, {}, forward time, {}, inference time, {}, qp time, {}, backward_time, {}\n'.format(str(seed), total_forward_time, total_inference_time, total_qp_time, total_backward_time))
        f_time.write('forward time,'   + ','.join([str(x) for x in forward_time_list]) + '\n')
        f_time.write('inference time,' + ','.join([str(x) for x in inference_time_list]) + '\n')
        f_time.write('qp time,'        + ','.join([str(x) for x in qp_time_list]) + '\n')
        f_time.write('backward time,'  + ','.join([str(x) for x in backward_time_list]) + '\n')
        f_time.close()

        # ============= early stopping criteria =============
        kk = 6
        if epoch >= kk*2-1:
            if training_method == 'two-stage':
                if evaluate:
                    break
                GE_counts = np.sum(np.array(validate_loss_list[-kk:]) >= np.array(validate_loss_list[-2*kk:-kk]) - 1e-6)
                print('Generalization error increases counts: {}'.format(GE_counts))
                if GE_counts == kk or np.sum(np.isnan(validate_loss_list[-kk:])) == kk:
                    evaluate = True
            else: # surrogate or decision-focused
                GE_counts = np.sum(np.array(validate_obj_list[-kk:]) <= np.array(validate_obj_list[-2*kk:-kk]) + 1e-6)
                print('Generalization error increases counts: {}'.format(GE_counts))
                if GE_counts == kk or np.sum(np.isnan(validate_obj_list[-kk:])) == kk:
                    break
