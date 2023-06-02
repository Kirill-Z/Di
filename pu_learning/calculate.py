import matplotlib.pyplot as plt
import numpy as np
from pulearn import ElkanotoPuClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from estimator import convert_to_PU, get_estimates, get_predicted_class, shuffle

data = None


def get_one_param_data(x_data, num_calc_param):
    x_one_param = []
    for x in x_data:
        x_one_param.append(x[num_calc_param])
    x_one_param = np.array(x_one_param)
    return x_one_param.reshape(-1, 1)


def get_score(y_test, y_pred):
    return (
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
    )


def get_scores(y_test, y_pred):
    return (
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
    )


def del_min_f1_score_param(x_train, y_train, s_train, x_test, y_test, del_index):
    x_train = np.delete(x_train, del_index, axis=1)
    x_test = np.delete(x_test, del_index, axis=1)
    return x_train, y_train, s_train, x_test, y_test


def main():
    x_true, x_false = data.get_data()
    y_true = np.ones((len(x_true), 1))
    y_false = np.full((len(x_false), 1), -1.0)

    x = np.concatenate((x_true, x_false))
    y = np.concatenate((y_true, y_false))
    x, y = shuffle(x, y)
    X, X_test_data, y, y_test_data = train_test_split(x, y, test_size=0.2)

    c = 1
    len_data = len(X)
    num_of_data = len_data
    estimator = RandomForestClassifier()

    X_test, y_test = shuffle(X_test_data, y_test_data)
    X_train, y_train, s_train, _ = convert_to_PU(X, y, c, num_of_data, len_data)
    total_f1_score = []
    total_precision = []
    total_recall = []
    deleted_params = []
    params = list(range(34))
    while len(X_train[0]):
        print(len(X_train[0]))
        print(params)
        f1_score_for_params = []
        num_of_params = len(X_train[0])
        y_pred_total = get_predicted_class(
            ElkanotoPuClassifier(estimator), X_train, s_train.ravel(), X_test
        )
        prec, rec, f1 = get_scores(y_test.ravel(), y_pred_total)
        total_f1_score.append(f1)
        total_precision.append(prec)
        total_recall.append(rec)
        for num_calc_param in range(num_of_params):
            x_test_one_param = get_one_param_data(X_test, num_calc_param)
            x_train_one_param = get_one_param_data(X_train, num_calc_param)

            y_pred = get_predicted_class(
                ElkanotoPuClassifier(estimator),
                x_train_one_param,
                s_train.ravel(),
                x_test_one_param,
            )
            precision, recall, f1 = get_score(y_test.ravel(), y_pred)
            f1_score_for_params.append(f1)

        min_f1 = f1_score_for_params.index(min(f1_score_for_params))
        del params[min_f1]
        print("deleted param: ", min_f1, "\n")
        deleted_params.append(min_f1)
        X_train, y_train, s_train, X_test, y_test = del_min_f1_score_param(
            X_train, y_train, s_train, X_test, y_test, min_f1
        )

    print(total_f1_score)
    print(deleted_params)
    range_data = list(range(0, 34))
    plt.plot(range_data, total_recall, label="Оправдываемость")
    plt.plot(range_data, total_precision, label="Точность")
    plt.xlabel("Кол-во случаев")
    plt.ylabel("Оценка")
    plt.legend()
    plt.axis([0, 34, 0.0, 1])
    plt.xticks(list(range(34)))
    plt.grid(which="major")
    plt.show()


if __name__ == "__main__":
    main()
