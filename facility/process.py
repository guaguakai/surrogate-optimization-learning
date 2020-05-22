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

    N_list = [30, 40, 50, 60, 70, 80, 90, 100, 120]
    methods = ['two-stage', 'decision-focused', 'surrogate']

    performance_prefix = 'movie_results/performance/'
    time_prefix        = 'movie_results/time/'

    testing_losses  = np.zeros((len(N_list), len(methods)))
    testing_objs    = np.zeros((len(N_list), len(methods)))
    training_losses = np.zeros((len(N_list), len(methods)))
    training_objs   = np.zeros((len(N_list), len(methods)))


    forward_time  = np.zeros((len(N_list), len(methods)))
    qp_time       = np.zeros((len(N_list), len(methods)))
    backward_time = np.zeros((len(N_list), len(methods)))

    for N_idx, N in enumerate(N_list):
        for method_idx, method in enumerate(methods):
            if method == 'surrogate':
                method = 'T{}-'.format(str(T)) + method
            f_performance = open(performance_prefix + filename + 'N{}-'.format(N) + method + '.csv', 'r')

            tmp_training_losses = [float(x) for x in f_performance.readline().split(',')[2:]]
            tmp_training_objs   = [float(x) for x in f_performance.readline().split(',')[2:]]
            tmp_testing_losses  = [float(x) for x in f_performance.readline().split(',')[2:]]
            tmp_testing_objs    = [float(x) for x in f_performance.readline().split(',')[2:]]

            if method == 'two-stage':
                selected_idx = np.argmin(tmp_training_losses)
            else:
                selected_idx = np.argmax(tmp_training_objs)

            training_losses[N_idx, method_idx] = tmp_training_losses[selected_idx]
            training_objs[N_idx, method_idx]   = tmp_training_objs[selected_idx]

            testing_losses[N_idx, method_idx]  = tmp_testing_losses[selected_idx]
            testing_objs[N_idx, method_idx]    = tmp_testing_objs[selected_idx]
            f_performance.close()

            f_time        = open(time_prefix        + filename + 'N{}-'.format(N) + method + '.csv', 'r')
            assert int(f_time.readline().split(',')[1]) == 49, "N: {}, method: {} incorrectly finished".format(N, method)
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

