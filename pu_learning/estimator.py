import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from puLearning.adapter import PUAdapter

empty = np.zeros((0, 0))


def shuffle(x, y=empty, z=empty):
    order = np.random.permutation(x.shape[0])
    x_shuffled = x[order, :]
    y_flag = False
    z_flag = False

    if y.shape[0] > 0:
        assert y.shape[0] == x.shape[0], "Arrays must have the same length."
        y_shuffled = y[order, :]
        y_flag = True

    if z.shape[0] > 0:
        assert z.shape[0] == x.shape[0], "Arrays must have the same length."
        z_shuffled = z[order, :]
        z_flag = True

    # Accomodate different number of outputs
    if y_flag and z_flag:
        return x_shuffled, y_shuffled, z_shuffled
    elif y_flag and not z_flag:
        return x_shuffled, y_shuffled
    elif not y_flag and not z_flag:
        return x_shuffled

    warnings.filterwarnings("ignore")


def convert_to_PU(X, y, c, num_of_data, default_num_of_data):
    n, m = X.shape
    len_y = len(np.where(y == 1.0)[0])
    pos_size = int(c * len_y)
    # Separate positive and negative data
    y_reshaped = y.reshape(y.shape[0])

    pos_mask = y_reshaped == 1.0
    neg_mask = y_reshaped == -1.0

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
    y = np.concatenate((np.ones((pos.shape[0], 1)), np.full((neg.shape[0], 1), -1.0)))
    s = np.concatenate((np.ones((P.shape[0], 1)), np.full((U.shape[0], 1), -1.0)))

    if num_of_data == default_num_of_data:
        end_num_of_data = default_num_of_data
    else:
        end_num_of_data = num_of_data + pos_size
        X = X[:end_num_of_data]
        y = y[:end_num_of_data]
        s = s[:end_num_of_data]

    X, y, s = shuffle(X, y, s)
    return X, y, s, end_num_of_data


class Estimator:
    def __init__(self, data, estimator, neural_network):
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = self.get_train_test_data()
        self.data = data
        self.estimator = estimator
        self.neural_network = neural_network
        self.percent_of_positive_data = [0.3, 0.5, 0.7, 1]
        self.num_of_data_list = None
        self.result = pd.DataFrame(
            columns=[
                "c",
                "Num of negative data",
                "Num of positive data",
                "Total num of data",
                "Precision",
                "Recall",
                "F1-score",
            ]
        )

    @abstractmethod
    def get_train_test_data(self):
        pass

    @abstractmethod
    def main(self):
        pass

    def get_predicted_class(self, estimator, x_train, s_train, x_test):
        estimator.fit(x_train, s_train)
        return estimator.predict(x_test)

    def get_calculated_data(
        self, y_test, y_pred, c, num_of_negative_data, num_of_positive_data
    ):
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        stat = {
            "c": c,
            "Num of negative data": int(num_of_negative_data),
            "Num of positive data": int(num_of_positive_data),
            "Total num of data": int(num_of_negative_data) + int(num_of_positive_data),
            "Precision": round(precision[1], 3),
            "Recall": round(recall[1], 3),
            "F1-score": round(f1_score[1], 3),
        }
        return stat

    def get_estimates(self, inscription, percent_of_positive_data=1):
        for num_of_data in self.num_of_data_list:
            print(inscription)
            print(
                "percent of positive data:",
                percent_of_positive_data,
                "\nnum_of_data:",
                num_of_data,
            )
            x_test, y_test = shuffle(self.x_test, self.y_test)
            x_train, y_train, s_train, _ = convert_to_PU(
                self.x_train,
                self.y_train,
                percent_of_positive_data,
                num_of_data,
                len(self.x_train),
            )
            y_pred = self.get_predicted_class(
                PUAdapter(self.estimator), x_train, s_train.ravel(), x_test
            )
            conf_matrix = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()
            print("tn:", tn, "fp:", fp)
            print("fn:", fn, "tp:", tp)
            num_positive_data = len(np.where(s_train == 1.0)[0])
            num_negative_data = len(np.where(s_train == -1.0)[0])
            stat = self.get_calculated_data(
                y_test.ravel(),
                y_pred,
                percent_of_positive_data,
                num_negative_data,
                num_positive_data,
            )
            self.result = self.result._append(stat, ignore_index=True)
