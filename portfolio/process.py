import numpy as np
import argparse

def customized_write(f, methods, N_list, array):
    f.write('n,' + ','.join(methods) + '\n')
    for N, line in zip(N_list, array):
        f.write(str(N) + ',' + ','.join(['{:.8f}'.format(x) for x in line]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Recommendation')
    parser.add_argument('--filename', type=str, help='filename under folder results')
    parser.add_argument('--T', type=int, default=10, help='T size')

    args = parser.parse_args()
    filename = args.filename
    T = args.T

    N_list = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250]
    methods = ['two-stage', 'decision-focused', 'surrogate']# ['two-stage', 'decision-focused', 'surrogate']

    performance_prefix = 'results/performance/'
    time_prefix        = 'results/time/'

    seed_list = set(range(1,31)) # - set([])
    testing_losses    = np.zeros((len(N_list), len(methods), len(seed_list)))
    testing_objs      = np.zeros((len(N_list), len(methods), len(seed_list)))
    testing_opts      = np.zeros((len(N_list), len(methods), len(seed_list)))
    testing_initials  = np.zeros((len(N_list), len(methods), len(seed_list)))

    training_losses   = np.zeros((len(N_list), len(methods), len(seed_list)))
    training_objs     = np.zeros((len(N_list), len(methods), len(seed_list)))
    training_opts     = np.zeros((len(N_list), len(methods), len(seed_list)))

    validating_losses = np.zeros((len(N_list), len(methods), len(seed_list)))
    validating_objs   = np.zeros((len(N_list), len(methods), len(seed_list)))
    validating_opts   = np.zeros((len(N_list), len(methods), len(seed_list)))


    forward_time   = np.zeros((len(N_list), len(methods), len(seed_list)))
    inference_time = np.zeros((len(N_list), len(methods), len(seed_list)))
    qp_time        = np.zeros((len(N_list), len(methods), len(seed_list)))
    backward_time  = np.zeros((len(N_list), len(methods), len(seed_list)))
    training_time  = np.zeros((len(N_list), len(methods), len(seed_list)))

    for N_idx, N in enumerate(N_list):
        for method_idx, method in enumerate(methods):
            for seed_idx, seed in enumerate(seed_list):
                if method == 'surrogate':
                    method = 'T{}-'.format(str(N//10)) + method
                f_performance = open(performance_prefix + filename + 'N{}-'.format(N) + method + '-SEED{}'.format(seed) + '.csv', 'r')
    
                finished_epoch = int(f_performance.readline().split(',')[1])
                print("N: {}, finished epoch: {}, seed: {}".format(N, finished_epoch, seed))
                # assert finished_epoch == 49, "N: {}, method: {} incorrectly finished".format(N, method)
    
                line = [float(x) for x in f_performance.readline().split(',')[1:]]
                tmp_training_losses = line[1:]
    
                line = [float(x) for x in f_performance.readline().split(',')[1:]]
                tmp_training_objs, training_opts[N_idx, method_idx, seed_idx] = line[1:], line[0]
    
                line = [float(x) for x in f_performance.readline().split(',')[1:]]
                tmp_validating_losses = line[1:]
    
                line = [float(x) for x in f_performance.readline().split(',')[1:]]
                tmp_validating_objs, validating_opts[N_idx, method_idx, seed_idx] = line[1:], line[0]
    
                line = [float(x) for x in f_performance.readline().split(',')[1:]]
                tmp_testing_losses = line[1:]
    
                line = [float(x) for x in f_performance.readline().split(',')[1:]]
                tmp_testing_objs, testing_opts[N_idx, method_idx, seed_idx], testing_initials[N_idx, method_idx, seed_idx] = line[1:], line[0], line[1]
    
                if method == 'two-stage':
                    selected_idx = -1 # np.argmin(tmp_validating_losses)
                else:
                    selected_idx = np.argmax(tmp_validating_objs)
    
                training_losses[N_idx, method_idx, seed_idx] = tmp_training_losses[selected_idx]
                training_objs[N_idx, method_idx, seed_idx]   = tmp_training_objs[selected_idx]
    
                testing_losses[N_idx, method_idx, seed_idx]  = tmp_testing_losses[selected_idx]
                testing_objs[N_idx, method_idx, seed_idx]    = tmp_testing_objs[selected_idx]
                f_performance.close()
    
                f_time        = open(time_prefix        + filename + 'N{}-'.format(N) + method + '-SEED{}'.format(seed) + '.csv', 'r')
    
                finished_epoch = int(f_time.readline().split(',')[1])
                print("N: {}, finished epoch: {}".format(N, finished_epoch))
                # assert finished_epoch == 49, "N: {}, method: {} incorrectly finished".format(N, method)
    
                line = f_time.readline().split(',')
                training_time[N_idx, method_idx, seed_idx] = (float(line[3]) + float(line[5]) + float(line[7]) + float(line[9])) / finished_epoch
                if method == 'two-stage':
                    line = f_time.readline().split(',')
                    forward_time[N_idx, method_idx, seed_idx] = float(line[selected_idx])
                    line = f_time.readline().split(',')
                    inference_time[N_idx, method_idx, seed_idx] = float(line[selected_idx])
                    line = f_time.readline().split(',')
                    qp_time[N_idx, method_idx, seed_idx] = float(line[selected_idx])
                    line = f_time.readline().split(',')
                    backward_time[N_idx, method_idx, seed_idx]  = float(line[selected_idx])
                else:
                    line = f_time.readline().split(',')
                    forward_time[N_idx, method_idx, seed_idx] = float(line[selected_idx + 2])
                    line = f_time.readline().split(',')
                    inference_time[N_idx, method_idx, seed_idx] = float(line[selected_idx + 2])
                    line = f_time.readline().split(',')
                    qp_time[N_idx, method_idx, seed_idx] = float(line[selected_idx + 2])
                    line = f_time.readline().split(',')
                    backward_time[N_idx, method_idx, seed_idx]  = float(line[selected_idx + 2])
                f_time.close()

    testing_stds = np.std(testing_opts - testing_objs, axis=2)
    testing_objs = np.mean(testing_opts - testing_objs, axis=2)
    testing_opts = np.mean(testing_opts, axis=2)
    testing_initials = np.mean(testing_initials, axis=2)

    training_time = np.mean(training_time, axis=2)

    forward_time   = np.mean(forward_time, axis=2)
    inference_time = np.mean(inference_time, axis=2)
    qp_time        = np.mean(qp_time, axis=2)
    backward_time  = np.mean(backward_time, axis=2)


    stats_path = 'stats/'
    f_stats_objs  = open(stats_path + 'training_objs.csv', 'w')
    f_stats_stds  = open(stats_path + 'training_stds.csv', 'w')
    f_stats_total = open(stats_path + 'total_time.csv', 'w')
    f_stats_time  = open(stats_path + 'time.csv', 'w')
    f_stats_opts  = open(stats_path + 'optimal.csv', 'w')

    customized_write(f_stats_objs, methods, N_list, testing_objs)
    customized_write(f_stats_stds, methods, N_list, testing_stds)
    customized_write(f_stats_opts, methods, N_list, testing_opts)
    customized_write(f_stats_opts, methods, N_list, testing_initials)

    customized_write(f_stats_total, methods, N_list, training_time)

    customized_write(f_stats_time, methods, N_list, forward_time)
    customized_write(f_stats_time, methods, N_list, inference_time)
    customized_write(f_stats_time, methods, N_list, qp_time)
    customized_write(f_stats_time, methods, N_list, backward_time)

    f_stats_objs.close()
    f_stats_time.close()


