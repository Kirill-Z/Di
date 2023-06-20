import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from CNN import cnn
from estimator import Estimator, convert_to_PU, shuffle


class MnistEstimator(Estimator):
    def __init__(self, data, estimator, neural_network):
        self.data = data
        super().__init__(data, estimator, neural_network)
        self.num_of_data_list = [60000, 45000, 30000, 15000, 5000]

    def convert_data_to_binary(self, x_train_all, y_train_all):
        Positive_class = 8
        Negative_class = [0, 1, 2, 3, 4, 5, 6, 7, 9]

        X_orig = x_train_all
        y_orig = y_train_all

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

    def get_train_test_data(self):
        (x_train, y_train), (x_test, y_test) = self.data.load_data()

        x_train, y_train = self.convert_data_to_binary(x_train, y_train)
        x_test, y_test = self.convert_data_to_binary(x_test, y_test)

        return x_train, x_test, y_train, y_test

    def main(self):
        if self.neural_network:
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
            for c in self.percent_of_positive_data:
                for num_of_data in self.num_of_data_list:
                    print("c:", c, "\nnum_of_data:", num_of_data)
                    x_test, y_test = shuffle(self.x_test, self.y_test)
                    x_train, y_train, s_train, shape_size = convert_to_PU(
                        self.x_train, self.y_train, c, num_of_data, len(self.x_train)
                    )
                    precision, recall, f1_score = cnn(
                        x_train, s_train.ravel(), x_test, y_test.ravel(), shape_size
                    )
                    num_of_positive_data = len(np.where(s_train == 1.0)[0])
                    num_of_negative_data = len(np.where(s_train == 0)[0])
                    stat = {
                        "c": c,
                        "Num of negative data": int(num_of_negative_data),
                        "Num of positive data": int(num_of_positive_data),
                        "Total num of data": int(num_of_negative_data) + int(num_of_positive_data),
                        "Precision": round(precision, 3),
                        "Recall": round(recall, 3),
                        "F1-score": round(f1_score, 3),
                    }
                    self.result = self.result._append(stat, ignore_index=True)
        else:
            for c in self.percent_of_positive_data:
                self.get_estimates("PU learning in progress...", c)
            self.get_estimates("Regular learning in progress...")
        print(self.result)
        x = self.result["Num of negative data"] + self.result["Num of negative data"]
        y = self.result["Num of positive data"] / self.result["Num of negative data"]
        z = self.result["Precision"]
        fontsize = 34
        points_whole_ax = 5 * 0.8 * 72
        radius = 1.2
        rad = 2 * radius / 1.0 * points_whole_ax

        text = [str(i) for i in self.result["c"]]

        plt.suptitle("Отношение точности к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("Точность", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()

        z = self.result["Recall"]

        plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
        for i in range(len(z)):
            plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -20), textcoords='offset points')
        plt.suptitle("Отношение оправдываемости к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("Оправдываемость", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()

        z = self.result["F1-score"]

        plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
        plt.subplots_adjust(left=0.13, bottom=0.114, right=0.983, top=0.926)
        for i in range(len(z)):
            plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -22), textcoords='offset points')
        plt.suptitle("Отношение F1 меры к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize, labelpad=10)
        plt.xlim((0.01, 1.25))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("F1 мера", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()


if __name__ == "__main__":
    estimator = MnistEstimator(
        data=tf.keras.datasets.mnist,
        estimator=DecisionTreeClassifier(),
        neural_network=False,
    )
    estimator.main()
