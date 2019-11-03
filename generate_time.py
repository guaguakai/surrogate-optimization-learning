import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_time(filename):
    f = open(filename, 'r')
    running_time_list = []
    optimizing_time_list = []
    key_list = []
    while True:
        item = f.readline()
        if not item: break
        item = item[:-1].split(',')
        running_time_list.append(float(item[3]))
        if len(item) > 5:
            optimizing_time_list.append(float(item[5]))
        key_list.append(int(item[1]))

    print('running time filename', filename)
    print(key_list)

    running_time_average = np.mean(running_time_list)
    optimizing_time_list = np.mean(optimizing_time_list) if len(optimizing_time_list) > 0 else 0
    return running_time_average, optimizing_time_list
