import sys
import tqdm
import numpy as np
import pandas as pd
import qpth
import random
import argparse
import torch
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

from gurobipy import *
from types import SimpleNamespace

from facilityUtils import createConstraintMatrix
from facilityDerivative import getObjective, getDerivative, getManualDerivative, getHessian, getOptimalDecision
from facilitySurrogateDerivative import getSurrogateObjective, getSurrogateDerivative, getSurrogateManualDerivative, getSurrogateHessian, getSurrogateOptimalDecision
from facilityUtils import train_submodular, test_submodular, surrogate_train_submodular, surrogate_test_submodular # train, surrogate_train, test
from utils import normalize_matrix, normalize_matrix_positive, normalize_vector, normalize_matrix_qr, normalize_projection

from movie.gmf import GMFWrapper
from movie.mlp import MLPWrapper
from movie.neumf import NeuMFWrapper
from movie.data import SampleGenerator

# Random Seed Initialization
SEED = 1289 #  random.randint(0,10000)
print("Random seed: {}".format(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if __name__ == '__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='GCN Interdiction')
    parser.add_argument('--filepath', type=str, default='movie_results/', help='filename under folder results')
    parser.add_argument('--method', type=int, default=0, help='0: two-stage, 1: decision-focused, 2: surrogate-decision-focused')
    parser.add_argument('--T-size', type=int, default=10, help='the size of reparameterization metrix')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--budget', type=int, default=20, help='budget')
    parser.add_argument('--n', type=int, default=50, help='number of items')
    parser.add_argument('--m', type=int, default=50, help='number of users')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    args = parser.parse_args()
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

    filepath = args.filepath

    # ============= Loading Movie Data =============
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

    # ================= Model setup =================
    from config import gmf_config, mlp_config, neumf_config
    # config = gmf_config
    # net = GMFWrapper(config=config)
    config = mlp_config
    net = MLPWrapper(config=mlp_config)
    # config = neumf_config
    # net = NeuMFWrapper(config=neumf_config)

    # ============ DataLoader for training ==========
    n, m = args.n, args.m # n: # of facilities or movies, m: # of customers or users
    num_epochs = args.epochs
    sample_generator = SampleGenerator(ratings=ml1m_rating, item_size=n, user_chunk_size=m)
    train_dataset, test_dataset = sample_generator.instance_a_train_loader_chunk(num_negatives=config['num_negative'])

    # =============== Learning setting ==============
    budget = args.budget
    lr = args.lr
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    sample_instance = SimpleNamespace(n=n, m=m, d=2*np.ones(m), f=np.ones(n), budget=budget) # dummy sample instance that is used to store the given n, m, d, f 
    A, b, G, h = createConstraintMatrix(m, n, budget)

    # surrogate setup
    if training_method == 'surrogate':
        # A, b, G, h = LPCreateSurrogateConstraintMatrix(m, n)
        variable_size = n
        T_size = args.T_size
        # init_T = normalize_matrix(torch.rand(variable_size, T_size))
        init_T = normalize_matrix_positive(torch.rand(variable_size, T_size))
        T = torch.tensor(init_T, requires_grad=True)
        T_lr = lr
        T_optimizer = torch.optim.Adam([T], lr=T_lr)

    train_loss_list, train_obj_list, train_opt_list = [], [], []
    test_loss_list,  test_obj_list,  test_opt_list  = [], [], []
    for epoch in range(-1, num_epochs):
        if training_method == 'surrogate':
            if epoch == -1:
                print('Not training in the first epoch...')
                train_loss, train_obj, train_opt = surrogate_train_submodular(net, T, optimizer, T_optimizer, epoch, sample_instance, train_dataset, training_method=training_method, disable=True)
            else:
                train_loss, train_obj, train_opt = surrogate_train_submodular(net, T, optimizer, T_optimizer, epoch, sample_instance, train_dataset, training_method=training_method)
        elif training_method == 'decision-focused' or training_method == 'two-stage':
            if epoch == -1:
                print('Not training in the first epoch...')
                train_loss, train_obj, train_opt = train_submodular(net, optimizer, epoch, sample_instance, train_dataset, training_method=training_method, disable=True)
            else:
                train_loss, train_obj, train_opt = train_submodular(net, optimizer, epoch, sample_instance, train_dataset, training_method=training_method)
        else:
            raise ValueError('Not implemented')
        # validate(dataset.validate)
        if training_method == 'surrogate':
            test_loss, test_obj, test_opt = surrogate_test_submodular(net, T, epoch, sample_instance, test_dataset)
        else:
            test_loss, test_obj, test_opt = test_submodular(net, epoch, sample_instance, test_dataset)

        random.shuffle(train_dataset)

        train_loss_list.append(train_loss)
        train_obj_list.append(train_obj)
        train_opt_list.append(train_opt)
        test_loss_list.append(test_loss)
        test_obj_list.append(test_obj)
        test_opt_list.append(test_opt)

        # record the data every epoch
        f_output = open(filepath + "{}.csv".format(training_method), 'w')
        f_output.write('training loss,' + ','.join([str(x) for x in train_loss_list]) + '\n')
        f_output.write('training obj,'  + ','.join([str(x) for x in train_obj_list])  + '\n')
        f_output.write('training opt,'  + ','.join([str(x) for x in train_opt_list])  + '\n')
        f_output.write('testing loss,'  + ','.join([str(x) for x in test_loss_list])  + '\n')
        f_output.write('testing obj,'   + ','.join([str(x) for x in test_obj_list])   + '\n')
        f_output.write('testing opt,'   + ','.join([str(x) for x in test_opt_list])   + '\n')

        f_output.close()
