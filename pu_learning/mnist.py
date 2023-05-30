import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from pulearn import ElkanotoPuClassifier
import warnings
import tensorflow as tf
from CNN import cnn
from puLearning.adapter import PUAdapter
from matplotlib import pyplot


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
    y_N = np.full((X_N.shape[0], 1), -1)

    X_P = X_P.reshape(X_P.shape[0], X_P.shape[1] * X_P.shape[2])
    X_N = X_N.reshape(X_N.shape[0], X_N.shape[1] * X_N.shape[2])

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


def convert_to_PU(X, y, c, num_of_data, default_num_of_data):
    n, m = X.shape
    len_y = len(np.where(y == 1.)[0])
    pos_size = int(c * len_y)
    # Separate positive and negative data
    y_reshaped = y.reshape(y.shape[0])

    pos_mask = y_reshaped == 1.
    neg_mask = y_reshaped == -1.

    pos = X[pos_mask, :]
    neg = X[neg_mask, :]

    # Shuffle pos and neg before dividing them up
    pos = shuffle(pos)
    neg = shuffle(neg)

    P = pos[0:pos_size, :]
    Q = pos[pos_size:, :]
    N = neg

    U = np.concatenate((Q, N))
    U = shuffle(U)

    X = np.concatenate((P, U))
    y = np.concatenate((np.ones((pos.shape[0], 1)), np.full((neg.shape[0], 1), -1.)))
    s = np.concatenate((np.ones((P.shape[0], 1)), np.full((U.shape[0], 1), -1.)))

    if num_of_data == default_num_of_data:
        end_num_of_data = default_num_of_data
    elif num_of_data != default_num_of_data:
        end_num_of_data = num_of_data + pos_size
        X = X[:end_num_of_data]
        y = y[:end_num_of_data]
        s = s[:end_num_of_data]

    X, y, s = shuffle(X, y, s)
    return X, y, s, end_num_of_data


def get_predicted_class(estimator, x_train, s_train, x_test):
    estimator.fit(x_train, s_train)
    return estimator.predict(x_test)


def get_estimates(y_test, y_pred, c, num_of_negative_data, num_of_positive_data):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    stat = {
        "c": c, "Num of negative data": int(num_of_negative_data), "Num of positive data": int(num_of_positive_data),
        "Total num of data": int(num_of_negative_data)+int(num_of_positive_data), "Precision": round(precision[1], 3), "Recall": round(recall[1], 3), "F1-score": round(f1_score[1], 3)
    }
    return stat




class MnistEstimator:
    def __init__(self, estimator=RandomForestClassifier(n_jobs=4), neural_network=False):
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_train_test_data()
        self.estimator = estimator
        self.neural_network = neural_network
        self.percent_of_positive_data = [0.3, 0.5, 0.7, 1]
        self.num_of_data_list = [60000, 45000, 30000, 15000, 5000, 1000]
        self.result = pd.DataFrame(
            columns=[
                "c", "Num of negative data", "Num of positive data", "Total num of data", "Precision", "Recall",
                "F1-score"
            ])

    def get_train_test_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, y_train = convert_data_to_binary(x_train, y_train)
        x_test, y_test = convert_data_to_binary(x_test, y_test)

        return x_train, x_test, y_train, y_test

    def get_estimates(self, inscription, percent_of_positive_data=1):
        for num_of_data in self.num_of_data_list:
            print(inscription)
            print("percent of positive data:", percent_of_positive_data, "\nnum_of_data:", num_of_data)
            x_test, y_test = shuffle(self.x_test, self.y_test)
            x_train, y_train, s_train, _ = convert_to_PU(self.x_train, self.y_train, percent_of_positive_data, num_of_data, len(self.x_train))
            y_pred = get_predicted_class(PUAdapter(self.estimator), x_train, s_train.ravel(), x_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()
            print(len(np.where(s_train == 1.)[0]))
            print(conf_matrix)
            print("tn:", tn, "fp:", fp)
            print("fn:", fn, "tp:", tp)
            num_positive_data = len(np.where(s_train == 1.)[0])
            num_negative_data = len(np.where(s_train == -1.)[0])
            stat = get_estimates(y_test.ravel(), y_pred, percent_of_positive_data, num_negative_data, num_positive_data)
            self.result = self.result._append(stat, ignore_index=True)


    def main(self):
        if self.neural_network:
            self.result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])
            for c in self.percent_of_positive_data:
                for num_of_data in self.num_of_data_list:
                    print("c:", c, "\nnum_of_data:", num_of_data)
                    x_test, y_test = shuffle(self.x_test, self.y_test)
                    x_train, y_train, s_train, shape_size = convert_to_PU(self.x_train, self.y_train, c, num_of_data, len(self.x_train))
                    precision, recall, f1_score = cnn(x_train, s_train.ravel(), x_test, y_test.ravel(), shape_size)
                    stat = {
                        "c": c, "Num of data": int(num_of_data), "Precision": precision,
                        "Recall": recall, "F1-score": f1_score
                    }
                    self.result = self.result._append(stat, ignore_index=True)
        else:
            for c in self.percent_of_positive_data:
                self.get_estimates("PU learning in progress...", c)
            self.get_estimates("Regular learning in progress...")
        print(self.result)

if __name__ == '__main__':
    estimator = MnistEstimator()
    estimator.main()

