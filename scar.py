import random
import numpy as np

c = 0.1  # Label frequency


def scar(y_train, label, e=0.1):
    y_train[y_train != label] = 0
    y_train[y_train == label] = 1
    s_train = change_label(y_train, e)
    return y_train, s_train


def change_label(y, e):
    a = np.random.rand(len(y))
    b = a < e
    b = b.astype(int)
    c = np.logical_and(y, b)
    return c


def get_label(y, e):
    if y == 0:
        return 0
    else:
        rand = random.random()
        return int(rand < e)
