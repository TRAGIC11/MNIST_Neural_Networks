import random
import numpy as np


def DisplayData(X):
    li = random.sample(range(0, 5000), 100)
    lol = np.zeros((220, 220))
    for i in range(10):
        for k in range(22):
            a = np.array([])
            for j in range(10):
                rt = make_np_array(X[li[i*10+j]])
                a = np.concatenate((a, rt[k]))
            lol[22*i + k] = a[:]

    return np.transpose(lol)


def Display_visualization(Theta):
    lol = np.zeros((110, 110))
    for i in range(5):
        for k in range(22):
            a = np.array([])
            for j in range(5):
                rt = make_np_array(Theta[i*5+j])
                a = np.concatenate((a, rt[k]))
            lol[22*i + k] = a[:]
    return np.transpose(lol)


def make_np_array(data):

    numpy_X = []

    for j in range(0, 20):
        numpy_X.append(data[j*20:(j+1)*20])
    numpy_X = np.array(numpy_X)
    numpy_X = np.pad(numpy_X, pad_width=1, mode='constant', constant_values=-1)
    return numpy_X
