import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_time(filename, key_list=[]):
    f = open(filename, 'r')
    print(filename)
    forward_time_list, qp_time_list, backward_time_list = [], [], []
    while True:
        item = f.readline()
        if not item: break
        item = item[:-1].split(',')
        seed = int(item[1])
        if seed not in key_list:
            continue
        forward_time_list.append(float(item[3]))
        qp_time_list.append(float(item[5]))
        backward_time_list.append(float(item[7]))

    print('running time filename', filename)
    print(key_list)

    forward_time_average  = np.mean(forward_time_list)
    qp_time_average       = np.mean(qp_time_list)
    backward_time_average = np.mean(backward_time_list)
    return forward_time_average, qp_time_average, backward_time_average
