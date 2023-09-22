import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from estimator import shuffle, get_positive_negative_data
from CNN import cnn
import pandas as pd


from load_thunder_data import ThunderData
from wind import WindEstimator


class ThunderEstimator(WindEstimator):
    def __init__(self, data, estimator, neural_network):
        super().__init__(data, estimator, neural_network)
        self.num_of_data_list = [104523, 78386, 52257, 26128, 10451]

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
                    x_train, s_train = get_positive_negative_data(
                        self.x_train,
                        self.y_train,
                        c,
                        num_of_data,
                        len(self.x_train)
                    )
                    shape_size = len(x_train)
                    """x_train, y_train, s_train, shape_size = convert_to_PU(
                        self.x_train, self.y_train, c, num_of_data, len(self.x_train)
                    )"""
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
                self.get_estimates("\nPU learning in progress...", c)
            #self.get_estimates("Regular learning in progress...")
        print(self.result)

        x = self.result["Num of negative data"] + self.result["Num of negative data"]
        y = self.result["Num of positive data"] / self.result["Num of negative data"]
        z = self.result["Precision"]
        fontsize = 34
        points_whole_ax = 5 * 0.8 * 72
        radius = 1.2
        rad = 2 * radius / 1.0 * points_whole_ax

        text = [str(i) for i in self.result["c"]]
        """plt.scatter(y, z, c=x, s=rad, cmap="rainbow")
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
        plt.show()"""
        x = self.result["c"][::5]
        y1 = self.result["F1-score"][::5]
        y2 = self.result["F1-score"][1::5]
        y3 = self.result["F1-score"][2::5]
        y4 = self.result["F1-score"][3::5]
        y5 = self.result["F1-score"][4::5]
        print("START")
        print(x)
        print(y3)
        print(len(x))
        print(len(y3))
        print("END")

        plt.plot(x, y1, label=int(self.result["Total num of data"][0]))
        plt.plot(x, y2, label=int(self.result["Total num of data"][1]))
        plt.plot(x, y3, label=int(self.result["Total num of data"][2]))
        plt.plot(x, y4, label=int(self.result["Total num of data"][3]))
        plt.plot(x, y5, label=int(self.result["Total num of data"][4]))
        plt.legend(prop={"size": 24})
        plt.xlabel(
            "Процент положительных данных от общего количества положительных данных",
            fontsize=fontsize,
            labelpad=10
        )
        plt.xticks(np.arange(0, 1.1, 0.1), fontsize=fontsize)
        #plt.suptitle("cnn", fontsize=fontsize)
        plt.ylabel("F1 мера", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    data = ThunderData(
        False,
        "/home/kirill/PycharmProjects/Di/thunder_data",
        "/home/kirill/PycharmProjects/Di/clear_thunder_data"
    )
    estimator = ThunderEstimator(
        data=data, estimator=SVC(probability=True), neural_network=False
    )
    estimator.main()
