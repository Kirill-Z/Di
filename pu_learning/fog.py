import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from pulearn import ElkanotoPuClassifier
from load_fog_data import get_data
import pandas as pd
from mnist import convert_to_PU, get_predicted_class, get_estimates, shuffle
from CNN import cnn_2


clear_data_path = "/home/kirill/PycharmProjects/Di/clear_data"

def main_for_fog():
    print("Loading dataset")
    x_true, x_false = get_data()
    y_true = np.ones((len(x_true), 1))
    y_false = np.full((len(x_false), 1), 0)

    x = np.concatenate((x_true, x_false))
    y = np.concatenate((y_true, y_false))
    x, y = shuffle(x, y)
    X, X_test_data, y, y_test_data = train_test_split(x, y, test_size=0.2)

    c_list = [0.3, 0.5, 0.7, 1]

    len_data = len(X)
    num_of_data_list = [len_data, len_data*0.75, len_data*0.5, len_data*0.25, len_data*0.1, len_data*0.02]
    #num_of_data_list = [620]

    result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])
    trad_result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])
    estimator = "cnn_2"
    if estimator == "cnn_2":
        for c in c_list:
            for num_of_data in num_of_data_list:
                print("c:", c, "\nnum_of_data:", num_of_data)
                X_test, y_test = shuffle(X_test_data, y_test_data)
                X_train, y_train, s_train, shape_size = convert_to_PU(X, y, c, num_of_data, len(X))
                precision, recall, f1_score = cnn_2(X_train, s_train.ravel(), X_test, y_test.ravel(), shape_size)
                stat = {
                    "c": c, "Num of data": int(num_of_data), "Precision": precision,
                    "Recall": recall, "F1-score": f1_score
                }
                result = result._append(stat, ignore_index=True)
        print(result)
    else:
        estimator = DecisionTreeClassifier()
        for c in c_list:
            for num_of_data in num_of_data_list:
                num_of_data = int(num_of_data)
                print("c", c, "\nnum_of_data:", num_of_data)
                X_test, y_test = shuffle(X_test_data, y_test_data)
                X_train, y_train, s_train, _ = convert_to_PU(X, y, c, num_of_data, len_data)

                print("PU learning in progress...")


                unique, counts = np.unique(s_train, return_counts=True)
                print(dict(zip(unique, counts)))

                y_pred = get_predicted_class(ElkanotoPuClassifier(estimator), X_train, s_train.ravel(), X_test)
                stat = get_estimates(y_test.ravel(), y_pred, c, num_of_data)
                result = result._append(stat, ignore_index=True)

        for num_of_data in num_of_data_list:
            num_of_data = int(num_of_data)
            print("c", 1, "\nnum_of_data:", num_of_data)
            X_test, y_test = shuffle(X_test_data, y_test_data)
            X_train, y_train, s_train, _ = convert_to_PU(X, y, 1, num_of_data, len_data)
            print("Regular learning in progress...")
            y_pred = get_predicted_class(estimator, X_train, s_train.ravel(), X_test)
            stat = get_estimates(y_test.ravel(), y_pred, 1, num_of_data)
            trad_result = trad_result._append(stat, ignore_index=True)

        print(result)
        print(trad_result)


if __name__ == '__main__':
    main_for_fog()