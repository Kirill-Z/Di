from keras.datasets import mnist
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scar import scar
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from keras.layers import Dense
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras import backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def del_data_by_indices(x_data, y_data, indices):
    x_positive = list()
    y_positive = list()

    x_data = x_data.tolist()
    y_data = y_data.tolist()

    count_del_indeces = 0
    for i in np.sort(indices):
        i -= count_del_indeces
        x_positive.append(x_data[i])
        y_positive.append(y_data[i])
        del x_data[i]
        del y_data[i]
        count_del_indeces += 1

    y_data = np.array(y_data)
    y_data[y_data == 1] = -1
    y_data = y_data.tolist()

    x_positive.extend(x_data)
    y_positive.extend(y_data)

    return np.array(x_positive), np.array(y_positive)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

y_train[y_train != 8] = -1
y_train[y_train == 8] = 1

y_test[y_test != 8] = -1
y_test[y_test == 8] = 1

len_positive = np.count_nonzero(y_train == 1)

c = 0.7

c_len_positive = round(c * len_positive)

positive_indices = np.random.choice(np.where(y_train == 1)[0], size=c_len_positive, replace=False)

image_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

P = x_train[positive_indices]
P_y = y_train[positive_indices]
print("y_train", y_train)
x_train, y_train = del_data_by_indices(x_train, y_train, positive_indices)
print("y_train", y_train)
num_of_data = 0
if num_of_data != 0:
    end_num_of_data = num_of_data + c_len_positive
    x_train = x_train[:end_num_of_data]
    y_train = y_train[:end_num_of_data]


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print(predictions)
print(y_test)
TP = (predictions[y_test == 1] == 1).sum()
TN = (predictions[y_test == -1] == -1).sum()
FP = (predictions[y_test == -1] == 1).sum()
FN = (predictions[y_test == 1] == -1).sum()
acc = (TP+TN)/(TP+TN + FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f = 2*TP/(2*TP+FP+FN)

print(precision_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(f1_score(y_test, predictions))

print("\nPrecision", precision)
print("Recall", recall)
print('F score', f)
