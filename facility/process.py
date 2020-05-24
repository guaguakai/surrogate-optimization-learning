import numpy as np
import argparse

def customized_write(f, methods, N_list, array):
    f.write('n,' + ','.join(methods) + '\n')
    for N, line in zip(N_list, array):
        f.write(str(N) + ',' + ','.join(['{:.3f}'.format(x) for x in line]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Recommendation')
    parser.add_argument('--filename', type=str, help='filename under folder results')
    parser.add_argument('--T', type=int, help='T size')

    args = parser.parse_args()
    filename = args.filename
    T = args.T

    N_list = [20, 30, 40, 50, 60, 80, 100, 120, 150]
    methods = ['two-stage', 'decision-focused', 'surrogate']# ['two-stage', 'decision-focused', 'surrogate']

    performance_prefix = 'movie_results/performance/'
    time_prefix        = 'movie_results/time/'

    testing_losses    = np.zeros((len(N_list), len(methods) + 1))
    testing_objs      = np.zeros((len(N_list), len(methods) + 1))
    training_losses   = np.zeros((len(N_list), len(methods) + 1))
    training_objs     = np.zeros((len(N_list), len(methods) + 1))
    validating_losses = np.zeros((len(N_list), len(methods) + 1))
    validating_objs   = np.zeros((len(N_list), len(methods) + 1))


    forward_time  = np.zeros((len(N_list), len(methods)))
    qp_time       = np.zeros((len(N_list), len(methods)))
    backward_time = np.zeros((len(N_list), len(methods)))

    for N_idx, N in enumerate(N_list):
        for method_idx, method in enumerate(methods):
            if method == 'surrogate':
                method = 'T{}-'.format(str(10)) + method
            f_performance = open(performance_prefix + filename + 'N{}-'.format(N) + method + '.csv', 'r')

            finished_epoch = int(f_performance.readline().split(',')[1])
            print("N: {}, finished epoch: {}".format(N, finished_epoch))
            # assert finished_epoch == 49, "N: {}, method: {} incorrectly finished".format(N, method)

            line = [float(x) for x in f_performance.readline().split(',')[1:]]
            tmp_training_losses, training_losses[N_idx,-1] = line[1:], line[0]

            line = [float(x) for x in f_performance.readline().split(',')[1:]]
            tmp_training_objs, training_objs[N_idx,-1] = line[1:], line[0]

            line = [float(x) for x in f_performance.readline().split(',')[1:]]
            tmp_validating_losses, validating_losses[N_idx,-1] = line[1:], line[0]

            line = [float(x) for x in f_performance.readline().split(',')[1:]]
            tmp_validating_objs, validating_objs[N_idx,-1] = line[1:], line[0]

            line = [float(x) for x in f_performance.readline().split(',')[1:]]
            tmp_testing_losses, testing_losses[N_idx,-1] = line[1:], line[0]

            line = [float(x) for x in f_performance.readline().split(',')[1:]]
            tmp_testing_objs, testing_objs[N_idx,-1] = line[1:], line[0]

            if method == 'two-stage':
                selected_idx = np.argmin(tmp_validating_losses)
            else:
                selected_idx = np.argmax(tmp_validating_objs)

            training_losses[N_idx, method_idx] = tmp_training_losses[selected_idx]
            training_objs[N_idx, method_idx]   = tmp_training_objs[selected_idx]

            testing_losses[N_idx, method_idx]  = tmp_testing_losses[selected_idx]
            testing_objs[N_idx, method_idx]    = tmp_testing_objs[selected_idx]
            f_performance.close()

            f_time        = open(time_prefix        + filename + 'N{}-'.format(N) + method + '.csv', 'r')

            finished_epoch = int(f_time.readline().split(',')[1])
            print("N: {}, finished epoch: {}".format(N, finished_epoch))
            # assert finished_epoch == 49, "N: {}, method: {} incorrectly finished".format(N, method)

            line = f_time.readline().split(',')
            forward_time[N_idx, method_idx], qp_time[N_idx, method_idx], backward_time[N_idx, method_idx]  = float(line[3]), float(line[5]), float(line[7])
            f_time.close()


    stats_path = 'stats/'
    f_stats_objs = open(stats_path + 'training_objs.csv', 'w')
    f_stats_time = open(stats_path + 'time.csv', 'w')

    customized_write(f_stats_objs, methods, N_list, testing_objs)
    customized_write(f_stats_time, methods, N_list, forward_time)
    customized_write(f_stats_time, methods, N_list, qp_time)
    customized_write(f_stats_time, methods, N_list, backward_time)

    f_stats_objs.close()
    f_stats_time.close()


