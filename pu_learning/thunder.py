import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from load_thunder_data import ThunderData
from wind import WindEstimator


class ThunderEstimator(WindEstimator):
    def __init__(self, data, estimator, neural_network):
        super().__init__(data, estimator, neural_network)
        self.percent_of_positive_data = [0.3, 0.5, 0.7]
        self.num_of_data_list = [78386, 52257, 26128, 10451]

    def main(self):
        super().main()

        x = self.result["Num of negative data"] + self.result["Num of negative data"]
        y = self.result["Num of positive data"] / self.result["Num of negative data"]
        z = self.result["Precision"]
        fontsize = 34
        points_whole_ax = 5 * 0.8 * 72
        radius = 1.2
        rad = 2 * radius / 1.0 * points_whole_ax

        text = [str(i) for i in self.result["c"]]
        plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
        plt.subplots_adjust(left=0.13, bottom=0.114, right=0.983, top=0.926)
        for i in range(len(z)):
            plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -22), textcoords='offset points')
        text = [str(i) for i in self.result["c"]]
        print(text)

        plt.suptitle("Отношение точности к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xlim((0.01, 1.2))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("Точность", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()

        z = self.result["Recall"]

        plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
        plt.subplots_adjust(left=0.13, bottom=0.114, right=0.983, top=0.926)
        for i in range(len(z)):
            plt.annotate(text[i], (y[i], z[i]), xycoords='data', xytext=(-7, -22), textcoords='offset points')

        plt.suptitle("Отношение оправдываемости к балансировке обучающей выборки", fontsize=fontsize)
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xlim((0.01, 1.2))
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
        plt.xlabel("Отношение данных с положительной меткой к неразмеченным данным", fontsize=fontsize)
        plt.xlim((0.01, 1.2))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel("F1 мера", fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="Общее кол-во данных", size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        plt.show()



if __name__ == "__main__":
    data = ThunderData(garbage=False)
    estimator = ThunderEstimator(
        data=data, estimator=RandomForestClassifier(n_jobs=4), neural_network=False
    )
    estimator.main()
