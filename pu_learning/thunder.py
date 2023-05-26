import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from pulearn import ElkanotoPuClassifier
from load_thunder_data import ThunderData
import pandas as pd
from mnist import convert_to_PU, get_predicted_class, get_estimates, shuffle
from CNN import cnn
from puLearning.adapter import PUAdapter

data = ThunderData()

def main():
    x_true, x_false = data.get_data()
    y_true = np.ones((len(x_true), 1))
    y_false = np.full((len(x_false), 1), -1.)

    x = np.concatenate((x_true, x_false))
    y = np.concatenate((y_true, y_false))

    X, y = shuffle(x, y)
    X, X_test_data, y, y_test_data = train_test_split(x, y, test_size=0.2)

    #c_list = [0.3, 0.5, 0.7, 1]
    c_list = [0.5]
    len_data = len(X)
    print("len data:", len_data)
    #num_of_data_list = [len_data, len_data * 0.75, len_data * 0.5, len_data * 0.25, len_data * 0.1, len_data * 0.02]
    num_of_data_list = [len_data*0.3]

    result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])
    trad_result = pd.DataFrame(columns=["c", "Num of data", "Precision", "Recall", "F1-score"])
    estimator = ""
    if estimator == "cnn_2":
        for c in c_list:
            for num_of_data in num_of_data_list:
                num_of_data = int(num_of_data)
                print("c:", c, "\nnum_of_data:", num_of_data)
                X_test, y_test = shuffle(X_test_data, y_test_data)
                X_train, y_train, s_train, shape_size = convert_to_PU(X, y, c, num_of_data, len(X))
                precision, recall, f1_score = cnn(X_train, s_train.ravel(), X_test, y_test.ravel(), shape_size)
                stat = {
                    "c": c, "Num of data": int(num_of_data), "Precision": precision,
                    "Recall": recall, "F1-score": f1_score
                }
                result = result._append(stat, ignore_index=True)
    else:

        estimator = RandomForestClassifier(max_depth=5, min_samples_leaf=5, n_jobs=4)
        """for c in c_list:
            for num_of_data in num_of_data_list:
                num_of_data = int(num_of_data)
                print("c:", c, "\nnum_of_data:", num_of_data)
                #X_test, y_test = shuffle(X_test_data, y_test_data)
                X_train, y_train, s_train, _ = convert_to_PU(X, y, c, num_of_data, len_data)

                y_pred = get_predicted_class(PUAdapter(estimator, hold_out_ratio=0.1), X_train, s_train)
                #conf_matrix = confusion_matrix(y_test, y_pred)
                #tn, fp, fn, tp = conf_matrix.ravel()
                #print(len(np.where(s_train == 1.)[0]))
                #print(conf_matrix)
                #print("tn:", tn, "fp:", fp)
                #print("fn:", fn, "tp:", tp)
                #stat = get_estimates(y_test.ravel(), y_pred, c, num_of_data)
                result = result._append(stat, ignore_index=True)"""

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
    main()