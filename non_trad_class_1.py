import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
import plotly.express as px
import sklearn.model_selection
from matplotlib import pyplot
import math
import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
import warnings
import tensorflow as tf


def load_MNIST_data(x_train_all, y_train_all):
    # @markdown Select positive and negative classes:
    Positive_class = '8'  # @param ["0","1","2","3","4","5","6","7","8","9"]
    #Negative_class = '3'  # @param ["0","1","2","3","4","5","6","7","8","9"]
    Negative_class = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    #assert Positive_class is not Negative_class, \
    #    "Positive and negative sets must be different"
    digit_P = int(Positive_class)
    #digit_N = int(Negative_class)

    # Load entire data set

    # combine data for PU problem
    X_orig = x_train_all
    y_orig = y_train_all

    # isolate desired digits
    X_P = X_orig[y_orig == digit_P, :, :]
    X_N = X_orig[y_orig == Negative_class[0], :, :]
    for n in Negative_class[1:]:
        X_N = np.concatenate((X_N, X_orig[y_orig == n, :, :]))

    y_P = np.ones((X_P.shape[0], 1))
    y_N = np.zeros((X_N.shape[0], 1))

    X_P = X_P.reshape(X_P.shape[0], X_P.shape[1] * X_P.shape[2])
    X_N = X_N.reshape(X_N.shape[0], X_N.shape[1] * X_N.shape[2])

    # Combine
    X = np.concatenate((X_P, X_N))
    y = np.concatenate((y_P, y_N))
    X, y = shuffle(X, y)

    return X, y


def convert_to_PU(X, y, c):
    n, m = X.shape
    pos_size = int(c * np.sum(y))

    # Separate positive and negative data
    y_reshaped = y.reshape(y.shape[0])

    pos_mask = y_reshaped == 1
    neg_mask = y_reshaped == 0

    pos = X[pos_mask, :]
    neg = X[neg_mask, :]

    # Shuffle pos and neg before dividing them up
    pos = shuffle(pos)
    neg = shuffle(neg)

    P = pos[0:pos_size, :]
    Q = pos[pos_size:, :]
    N = neg

    U = np.concatenate((Q, N))

    X = np.concatenate((P, U))
    y = np.concatenate((np.ones((pos.shape[0], 1)), np.zeros((neg.shape[0], 1))))
    s = np.concatenate((np.ones((P.shape[0], 1)), np.zeros((U.shape[0], 1))))

    # Shuffle again
    X, y, s = shuffle(X, y, s)

    return X, y, s


empty = np.zeros((0,0))
def shuffle(x, y=empty, z=empty):
  order = np.random.permutation(x.shape[0])
  x_shuffled = x[order,:]
  y_flag = False
  z_flag = False

  if y.shape[0] > 0:
    assert y.shape[0] == x.shape[0], 'Arrays must have the same length.'
    y_shuffled = y[order,:]
    y_flag = True

  if z.shape[0] > 0:
    assert z.shape[0] == x.shape[0], 'Arrays must have the same length.'
    z_shuffled = z[order,:]
    z_flag = True

  #Accomodate different number of outputs
  if y_flag and z_flag:
    return x_shuffled, y_shuffled, z_shuffled
  elif y_flag and not z_flag:
    return x_shuffled, y_shuffled
  elif not y_flag and not z_flag:
    return x_shuffled


def evaluate(y, y_hat, s, y_test):
    y_hat_bin = (y_hat >= 0.5) * 1  # This threshold could be adjusted
    # CM = confusion_matrix(y, y_hat_bin)]

    y_reshaped = y.reshape(y.shape[0])
    y_test_reshaped = y_test.reshape(y_test.shape[0])

    TP = (y_hat_bin[y_test_reshaped == 1] == 1).sum()
    TN = (y_hat_bin[y_test_reshaped == 0] == 0).sum()
    FP = (y_hat_bin[y_test_reshaped == 0] == 1).sum()
    FN = (y_hat_bin[y_test_reshaped == 1] == 0).sum()

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f = 2 * TP / (2 * TP + FP + FN)

    print("\nPrecision", precision)
    print("Recall", recall)
    print('F score', f)

    print(len(y))
    print(len(y_hat_bin))
    accuracy = accuracy_score(y, y_hat_bin)
    recall = recall_score(y, y_hat_bin)
    precision = precision_score(y, y_hat_bin)
    fscore = f1_score(y, y_hat_bin)
    mcc = matthews_corrcoef(y, y_hat_bin)

    return accuracy, fscore, recall, precision, mcc


def DecisionTree(X_train, X_test, y, y_test, s):
    clf = DecisionTreeClassifier()
    estimator = clf.fit(X_train, s)
    predicted_class = clf.predict(X_test)
    stats = evaluate(y, predicted_class, s, y_test)
    return stats


warnings.filterwarnings("ignore")

# @markdown ##Experiment Parameters
#C_vals = '0.3, 0.5, 0.7'  # @param["0.5", "0.1, 0.5, 0.9", "0.1, 0.2, ..., 0.9", "custom"]
C_vals = '0.3'
num_runs = 3  # @param {type:"integer"}

if C_vals == '0.5':
    c_vals = [0.5]
elif C_vals == '0.3':
    c_vals = [0.3]
elif C_vals == '0.3, 0.5, 0.7':
    c_vals = [0.3, 0.5, 0.7]

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

X, y = load_MNIST_data(x_train, y_train)
X_test, y_test = load_MNIST_data(x_test, y_test)



results = pd.DataFrame(columns=['c', 'Accuracy', 'F-score',
                                'Precision', 'Recall', 'MCC',])

for c in c_vals:
    print('c = ', c)
    # Randomly select subsets to be known and unknown
    X_loop, y_loop, s_loop = convert_to_PU(X, y, c)
    X_loop_test, y_loop_test, s_loop_test = convert_to_PU(X_test, y_test, 1)
    stats = DecisionTree(X_loop, X_loop_test, y_loop, y_loop_test, s_loop)

    test_stats = {
        'c': c,
        'Accuracy': stats[0],
        'F-score': stats[1],
        'Recall': stats[2],
        'Precision': stats[3],
        'MCC': stats[4],
    }

    results = results._append(test_stats, ignore_index=True)

print(results)




