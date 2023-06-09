import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from estimator import Estimator, shuffle
from load_wind_data import WindData


class WindEstimator(Estimator):
    def __init__(self, data, estimator, neural_network):
        self.data = data
        super().__init__(data, estimator, neural_network)
        self.num_of_data_list = [639, 540, 360, 180, 72]

    def get_train_test_data(self):
        x_true, x_false = self.data.get_data()
        y_true = np.ones((len(x_true), 1))
        y_false = np.full((len(x_false), 1), -1.0)

        x = np.concatenate((x_true, x_false))
        y = np.concatenate((y_true, y_false))

        x, y = shuffle(x, y)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, random_state=42
        )
        return x_train, x_test, y_train, y_test

    def main(self):
        for c in self.percent_of_positive_data:
            self.get_estimates("PU learning in progress...", c)
        self.get_estimates("Regular learning in progress...")
        print(self.result)
        """print(self.result)
        x = self.result["Num of negative data"] + self.result["Num of negative data"]
        y = self.result["Num of positive data"] / self.result["Num of negative data"]
        z = self.result["Precision"]
        fontsize = 30
        points_whole_ax = 5 * 0.8 * 72  # 1 point = dpi / 72 pixels
        radius = 0.4
        rad = 2 * radius / 1.0 * points_whole_ax

        text = [str(i) for i in self.result["c"]]
        plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
        for i in range(len(z)):
            plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -20), textcoords='offset points')
        text = [str(i) for i in self.result["c"]]
        print(text)
        #for i in range(len(z)):
        #    plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -20), textcoords='offset points')
        plt.suptitle("Отношение точности к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xticks(np.arange(0.05, 1.2, 0.1), fontsize=28)
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
        #for i in range(len(z)):
        #    plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -20), textcoords='offset points')
        plt.suptitle("Отношение оправдываемости к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xticks(np.arange(0.05, 1.2, 0.1), fontsize=28)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("Оправдываемость", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()

        z = self.result["F1-score"]

        plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
        for i in range(len(z)):
            plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -20), textcoords='offset points')
        plt.suptitle("Отношение F1 меры к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xticks(np.arange(0.05, 1.2, 0.1), fontsize=28)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("F1 мера", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()"""


if __name__ == "__main__":
    data = WindData(garbage=True)
    estimator = WindEstimator(
        data=data, estimator=RandomForestClassifier(n_jobs=4), neural_network=False
    )
    estimator.main()
