import random

import numpy as np
from keras.datasets import mnist

from scar import scar


def load_data():
    pass


def label_to_unlabel(label=8):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    unique, count = np.unique(y_train, return_counts=True)
    num_of_keys = dict(zip(unique, count))
    unlabel = scar(y_train, 8, 0.5)
    count = 0
    for i in unlabel:
        if i == 1:
            count += 1
    print(count)


def get_a_percent_of_the_total(len_whole, len_part):
    return 100 * float(len_part)/float(len_whole)


def get_a_percent_of_the_positive_data(whole, percent=100):
    return int(whole*percent/100)


if __name__ == '__main__':
    label_to_unlabel()


