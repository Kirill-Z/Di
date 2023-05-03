import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from pulearn import ElkanotoPuClassifier
import warnings
import tensorflow as tf
from CNN import cnn, cnn_2
import csv


def read_csv(file_name):
    file_path = "/home/kirill/PycharmProjects/Di/pu_learning/" + file_name
    file = open(file_path, "r")
    data = list(csv.reader(file, delimiter=","))
    data = list(map(list, zip(*data)))
    file.close()
    return data


def convert_txt_to_csv(file_name):
    file = "/home/kirill/PycharmProjects/Di/" + file_name


def convert_data_to_binary(x_train_all, y_train_all):
    # @markdown Select positive and negative classes:
    Positive_class = 8
    Negative_class = [0, 1, 2, 3, 4, 5, 6, 7, 9]

    # Load entire data set
    # combine data for PU problem
    X_orig = x_train_all
    y_orig = y_train_all

    # isolate desired digits
    X_P = X_orig[y_orig == Positive_class, :, :]
    X_N = X_orig[y_orig == Negative_class[0], :, :]
    for n in Negative_class[1:]:
        X_N = np.concatenate((X_N, X_orig[y_orig == n, :, :]))


    y_P = np.ones((X_P.shape[0], 1))
    y_N = np.full((X_N.shape[0], 1), 0)

    X_P = X_P.reshape(X_P.shape[0], X_P.shape[1] * X_P.shape[2])
    X_N = X_N.reshape(X_N.shape[0], X_N.shape[1] * X_N.shape[2])
    # Combine
    X = np.concatenate((X_P, X_N))
    y = np.concatenate((y_P, y_N))

    X, y = shuffle(X, y)
    return X, y


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

    warnings.filterwarnings("ignore")


def convert_to_PU(X, y, c, num_of_data=60000):
    n, m = X.shape
    len_y = len(np.where(y == 1.)[0])
    pos_size = int(c * len_y)
    # Separate positive and negative data
    y_reshaped = y.reshape(y.shape[0])

    pos_mask = y_reshaped == 1.
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
    y = np.concatenate((np.ones((pos.shape[0], 1)), np.full((neg.shape[0], 1), 0)))
    s = np.concatenate((np.ones((P.shape[0], 1)), np.full((U.shape[0], 1), 0)))
    end_num_of_data = 60000
    if num_of_data != 60000:
        end_num_of_data = num_of_data + pos_size
        X = X[:end_num_of_data]
        y = y[:end_num_of_data]
        s = s[:end_num_of_data]
    # Shuffle again
    X, y, s = shuffle(X, y, s)
    return X, y, s, end_num_of_data


def get_predicted_class(estimator, x_train, s_train, x_test):
    estimator.fit(x_train, s_train)
    return estimator.predict(x_test)


def get_estimates(y_test, y_pred, c, num_of_data):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test.ravel(), y_pred)
    stat = {
        "c": c, "Num of data": int(num_of_data), "Precision": round(precision[1], 3),
        "Recall": round(recall[1], 3), "F1-score": round(f1_score[1], 3)
    }
    return stat


def get_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X, y = convert_data_to_binary(x_train, y_train)
    X_test, y_test = convert_data_to_binary(x_test, y_test)

    return X, y, X_test, y_test


def main():
    print("Loading dataset")
    X, y, X_test, y_test = get_data()

    c_list = [0.3, 0.5, 0.7, 1]
    num_of_data_list = [60000, 45000, 30000, 15000, 1000]

    result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])
    trad_result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])

    X_test, y_test, s_test, _ = convert_to_PU(X_test, y_test, 1)


    estimator = "cnn_2"
    if estimator == "cnn_2":
        for c in c_list:
            for num_of_data in num_of_data_list:
                print("c", c, "\nnum_of_data:", num_of_data)
                X_train, y_train, s_train, shape_size = convert_to_PU(X, y, c, num_of_data)
                precision, recall, f1_score = cnn_2(X_train, s_train.ravel(), X_test, s_test.ravel(), shape_size)
                stat = {
                    "c": c, "Num of data": int(num_of_data), "Precision": round(precision, 3),
                    "Recall": round(recall, 3), "F1-score": round(f1_score, 3)
                }
                result = result._append(stat, ignore_index=True)
        print(result)
    else:
        for c in c_list:
            for num_of_data in num_of_data_list:
                print("c", c, "\nnum_of_data:", num_of_data)

                X_train, y_train, s_train, _ = convert_to_PU(X, y, c, num_of_data)

                print("PU learning in progress...")
                estimator = RandomForestClassifier()
                y_pred = get_predicted_class(estimator, X_train, s_train.ravel(), X_test)
                stat = get_estimates(y_test.ravel(), y_pred, c, num_of_data)
                result = result._append(stat, ignore_index=True)

                print("Regular learning in progress...")
                y_pred = get_predicted_class(estimator, X_train, s_train.ravel(), X_test)
                stat = get_estimates(y_test.ravel(), y_pred, c, num_of_data)
                trad_result = trad_result._append(stat, ignore_index=True)

        print(result)
        print(trad_result)


if __name__ == '__main__':
    main()

