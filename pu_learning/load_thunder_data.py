import os
import csv
import pandas as pd
import numpy as np
from load_fog_data import Data


def separate_false_and_true_data():
    dir_path = "/home/kirill/PycharmProjects/Di/thunder_data"
    clear_data_path = "/home/kirill/PycharmProjects/Di/clear_thunder_data"

    open("/home/kirill/PycharmProjects/Di/clear_thunder_data/true_data.csv", "w").close()
    open("/home/kirill/PycharmProjects/Di/clear_thunder_data/false_data.csv", "w").close()

    for file in os.listdir(dir_path):
        read_file = pd.read_csv(os.path.join(dir_path, file), usecols=[i for i in range(91)])
        true_data = read_file[read_file["TunderYN"] == 1.]
        false_data = read_file[read_file["TunderYN"] == 0.]
        true_data = true_data.replace([np.inf, -np.inf], np.nan)
        false_data = false_data.replace([np.inf, -np.inf], np.nan)
        true_data[true_data > 3.40282346638528859811704183484516925440e+30] = np.nan
        false_data[false_data > 3.40282346638528859811704183484516925440e+30] = np.nan
        true_data.dropna(inplace=True)
        false_data.dropna(inplace=True)
        true_data = true_data.iloc[:, 0:90]
        false_data = false_data.iloc[:, 0:90]
        true_data.to_csv(os.path.join(clear_data_path, "true_data.csv"), mode="a", header=None, index=False)
        false_data.to_csv(os.path.join(clear_data_path, "false_data.csv"), mode="a", header=None, index=False)

class ThunderData(Data):
    def __init__(self):
        super().__init__()
        self.dir_path = "/home/kirill/PycharmProjects/Di/thunder_data"
        self.clear_data_path = "/home/kirill/PycharmProjects/Di/clear_thunder_data"

    def get_data(self):
        file_path = os.path.join(self.clear_data_path, "true_data.csv")
        read_file = open(file_path, "r")
        true_data = list(csv.reader(read_file, delimiter=","))
        read_file.close()
        file_path = os.path.join(self.clear_data_path, "false_data.csv")
        read_file = open(file_path, "r")
        false_data = list(csv.reader(read_file, delimiter=","))
        read_file.close()
        return self.conver_str_to_float(true_data), self.conver_str_to_float(false_data)

    def conver_str_to_float(self, data):
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                data[i][j] = float(data[i][j])
        return np.array(data)

    def del_garbage(self, data):
        return data


#a = ThunderData()()
separate_false_and_true_data()
