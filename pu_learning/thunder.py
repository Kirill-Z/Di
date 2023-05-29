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

    x, y = shuffle(x, y)
    X, X_test_data, y, y_test_data = train_test_split(x, y, test_size=0.33, random_state=42)

    c_list = [0.3, 0.5, 0.7, 1]
    len_data = len(X)
    num_of_data_list = [88546, 78386, 52257, 26128, 10451, 2090]

    result = pd.DataFrame(columns=["c", "Num of negative data", "Num of positive data", "Total num of data", "Precision", "Recall", "F1-score"])
    trad_result = pd.DataFrame(columns=["c", "Num of negative data",  "Num of positive data", "Total num of data","Precision", "Recall", "F1-score"])

    estimator = RandomForestClassifier(n_jobs=4)
    for c in c_list:
        for num_of_data in num_of_data_list:
            num_of_data = int(num_of_data)
            print("\nPU learning in progress...")
            print("c:", c, "\nnum_of_data:", num_of_data)
            X_test, y_test = shuffle(X_test_data, y_test_data)
            X_train, y_train, s_train, _ = convert_to_PU(X, y, c, num_of_data, len_data)

            #pu_estimator = PUAdapter(estimator)
            #pu_estimator.fit(X_train, s_train.ravel())
            #y_pred = pu_estimator.predict(X_test)

            y_pred = get_predicted_class(PUAdapter(estimator), X_train, s_train.ravel(), X_test)
            #conf_matrix = confusion_matrix(y_test, y_pred)
            #tn, fp, fn, tp = conf_matrix.ravel()
            #print(len(np.where(s_train == 1.)[0]))
            #print(conf_matrix)
            #print("tn:", tn, "fp:", fp)
            #print("fn:", fn, "tp:", tp)
            num_positive_data = len(np.where(s_train == 1.)[0])
            num_negative_data = len(np.where(s_train == -1.)[0])
            print("num_positive_data: ", num_positive_data)
            print("num_negative_data: ", num_negative_data)
            stat = get_estimates(y_test.ravel(), y_pred, c, num_negative_data, num_positive_data)
            result = result._append(stat, ignore_index=True)

    for num_of_data in num_of_data_list:
        num_of_data = int(num_of_data)
        X_test, y_test = shuffle(X_test_data, y_test_data)
        X_train, y_train, s_train, _ = convert_to_PU(X, y, 1, num_of_data, len_data)
        print("\nRegular learning in progress...")
        print("c", 1, "\nnum_of_data:", num_of_data)
        y_pred = get_predicted_class(estimator, X_train, s_train.ravel(), X_test)
        num_positive_data = len(np.where(s_train == 1.)[0])
        num_negative_data = len(np.where(s_train == -1.)[0])
        print("num_positive_data: ", num_positive_data)
        print("num_negative_data: ", num_negative_data)
        stat = get_estimates(y_test.ravel(), y_pred, 1, num_negative_data, num_positive_data)
        trad_result = trad_result._append(stat, ignore_index=True)

    print(result)
    #print(trad_result)


if __name__ == '__main__':
    main()

