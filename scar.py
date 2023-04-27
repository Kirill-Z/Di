import random
import numpy as np

c = 0.1  # Label frequency


def scar(x_train, y_train, label, e=0.1):
    y_train[y_train != label] = -1
    y_train[y_train == label] = 1
    s_train = change_label(y_train, e)
    x_train, y_train, s_train, len_true_data = replace_data(x_train, y_train, s_train)
    return x_train, y_train, s_train, len_true_data


def replace_data(x_train, y_train, s_train):
    sorted_x = list()
    sorted_y = list()
    sorted_s = list()

    x_train = x_train.tolist()
    y_train = y_train.tolist()
    s_train = s_train.tolist()

    len_data = len(s_train)
    i = 0
    while i < len_data:
        if s_train[i] == 1:
            sorted_s.append(s_train[i])
            sorted_y.append(y_train[i])
            sorted_x.append(x_train[i])
            del s_train[i]
            del y_train[i]
            del x_train[i]
            len_data -= 1
            continue
        i += 1
    len_true_data = len(sorted_s)

    sorted_x.extend(x_train)
    sorted_y.extend(y_train)
    sorted_s.extend(s_train)

    sorted_x_arr = np.array(sorted_x)
    sorted_y_arr = np.array(sorted_y)
    sorted_s_arr = np.array(sorted_s)

    return sorted_x_arr, sorted_y_arr, sorted_s_arr, len_true_data


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
