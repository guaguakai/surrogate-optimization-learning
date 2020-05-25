import numpy as np
import argparse
import pandas as pd

def customized_write(f, methods, N_list, array):
    f.write('n,' + ','.join(methods) + '\n')
    for N, line in zip(N_list, array):
        f.write(str(N) + ',' + ','.join(['{:.3f}'.format(x) for x in line]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Recommendation')
    parser.add_argument('--filename', type=str, help='filename under folder results')

    args = parser.parse_args()
    filename = args.filename

    N_list = [20, 30, 40]
    methods = ['two-stage', 'decision-focused', 'hybrid', 'surrogate-decision-focused'] # ['two-stage', 'decision-focused', 'surrogate']

    performance_prefix = 'results/random/'
    time_prefix        = 'results/time/random/'
    postfix            = 'p0.2_b3.0_cut10_noise0.2.csv'

    column_names = ['n'] + methods
    testing_losses    = pd.DataFrame(columns=column_names) 
    testing_objs      = pd.DataFrame(columns=column_names)
    training_losses   = pd.DataFrame(columns=column_names)
    training_objs     = pd.DataFrame(columns=column_names)
    validating_losses = pd.DataFrame(columns=column_names)
    validating_objs   = pd.DataFrame(columns=column_names)

    optimal_objs      = pd.DataFrame(columns=column_names)

    time_column_names = ['n'] + methods
    forward_time  = pd.DataFrame(columns=time_column_names) 
    qp_time       = pd.DataFrame(columns=time_column_names)
    backward_time = pd.DataFrame(columns=time_column_names)

    performance_header = ['random seed T', 'random seed',
            'train loss T', 'train loss', 'train defu T', 'train defu', 'train opt T', 'train opt',
            'validate loss T', 'validate loss', 'validate defu T', 'validate defu', 'validate opt T', 'validate opt',
            'test loss T', 'test loss', 'test defu T', 'test defu', 'test opt T', 'test opt']
    time_header        = ['random seed T', 'random seed', 
            'forward time T', 'forward time',
            'qp time T', 'qp time',
            'backward time T', 'backward time']

    sample_set = list(set(range(1,31)) - set([]))
    for N_idx, N in enumerate(N_list):
        tmp_test_loss_dict     = {'n': N}
        tmp_test_obj_dict      = {'n': N}
        tmp_train_loss_dict    = {'n': N}
        tmp_train_obj_dict     = {'n': N}
        tmp_validate_loss_dict = {'n': N}
        tmp_validate_obj_dict  = {'n': N}
        tmp_optimal_obj_dict   = {'n': N}

        tmp_forward_dict       = {'n': N}
        tmp_qp_dict            = {'n': N}
        tmp_backward_dict      = {'n': N}

        for method_idx, method in enumerate(methods):
            performance_path = performance_prefix + '{}_{}_coverage_n{}_'.format(filename, method, N) + postfix
            performance_pd   = pd.read_csv(performance_path, names=performance_header)

            time_path        = time_prefix + '{}_{}_coverage_n{}_'.format(filename, method, N) + postfix
            time_pd          = pd.read_csv(time_path, names=time_header)

            # filtering
            performance_pd   = performance_pd[performance_pd['random seed'].isin(sample_set)]
            time_pd          = time_pd[time_pd['random seed'].isin(sample_set)]

            # assert the right number of samples
            random_seeds = sorted(list(performance_pd['random seed']))
            assert random_seeds == sample_set, 'Random seed does not match: N {}, method {}'.format(N, method)

            # computing losses and objs
            tmp_test_loss_dict[method]     = np.mean(performance_pd['test loss'].astype(float))
            tmp_train_loss_dict[method]    = np.mean(performance_pd['train loss'].astype(float))
            tmp_validate_loss_dict[method] = np.mean(performance_pd['validate loss'].astype(float))

            tmp_test_obj_dict[method]      = np.mean(performance_pd['test defu'].astype(float))
            tmp_train_obj_dict[method]     = np.mean(performance_pd['train defu'].astype(float))
            tmp_validate_obj_dict[method]  = np.mean(performance_pd['validate defu'].astype(float))

            tmp_optimal_obj_dict[method]   = np.mean(performance_pd['test opt'].astype(float))

            tmp_forward_dict[method]       = np.mean(time_pd['forward time'].astype(float))
            tmp_qp_dict[method]            = np.mean(time_pd['qp time'].astype(float))
            tmp_backward_dict[method]      = np.mean(time_pd['backward time'].astype(float))

        testing_losses    = testing_losses.append(pd.DataFrame(tmp_test_loss_dict, index=[N_idx]))
        training_losses   = training_losses.append(pd.DataFrame(tmp_train_loss_dict, index=[N_idx]))
        validating_losses = validating_losses.append(pd.DataFrame(tmp_validate_loss_dict, index=[N_idx]))

        testing_objs      = testing_objs.append(pd.DataFrame(tmp_test_obj_dict, index=[N_idx]))
        training_objs     = training_objs.append(pd.DataFrame(tmp_train_obj_dict, index=[N_idx]))
        validating_objs   = validating_objs.append(pd.DataFrame(tmp_validate_obj_dict, index=[N_idx]))
        optimal_objs      = optimal_objs.append(pd.DataFrame(tmp_optimal_obj_dict, index=[N_idx]))

        forward_time      = forward_time.append(pd.DataFrame(tmp_forward_dict, index=[N_idx]))
        qp_time           = qp_time.append(pd.DataFrame(tmp_qp_dict, index=[N_idx]))
        backward_time     = backward_time.append(pd.DataFrame(tmp_backward_dict, index=[N_idx]))

    stats_path = 'stats/'

    testing_objs.to_csv(stats_path + 'testing_objs.csv', index=False)
    optimal_objs.to_csv(stats_path + 'optimal_objs.csv', index=False)

    forward_time.to_csv(stats_path + 'forward_time.csv', index=False)
    qp_time.to_csv(stats_path + 'qp_time.csv', index=False)
    backward_time.to_csv(stats_path + 'backward_time.csv', index=False)


