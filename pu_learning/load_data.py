import csv
import os

import numpy as np
import pandas as pd


class Data:
    def __init__(self, garbage):
        self.dir_path = None
        self.clear_data_path = None
        self.garbage = garbage

    def __call__(self):
        true, false = self.read_txt()
        filename_true = self.txt_to_csv(true)
        filename_false = self.txt_to_csv(false)
        self.read_csv(filename_true, "true_data.csv")
        self.read_csv(filename_false, "false_data.csv")

    def del_garbage(self, data):
        pass

    def read_txt(self):
        true_files = []
        false_files = []
        for file in os.listdir(self.dir_path):
            if file.endswith("true.txt"):
                true_files.append(file)
            if file.endswith("false.txt"):
                false_files.append(file)

        return true_files, false_files

    def txt_to_csv(self, file):
        filename = []
        for f in file:
            read_file = pd.read_csv(
                os.path.join(self.dir_path, f), sep=r"\s+", header=None, dtype=str
            )
            new_file_name = os.path.splitext(f)[0] + ".csv"
            read_file.to_csv(os.path.join(self.dir_path, new_file_name), index=False)
            filename.append(new_file_name)
        return filename

    def save_data_to_csv(self, filename, data):
        with open(filename, "ab") as f:
            f.write(b"")
            np.savetxt(f, data, delimiter=",", fmt="%s")

    def read_csv(self, filename: list, new_file_name):
        path_for_clear_data = os.path.join(self.clear_data_path, new_file_name)
        open(path_for_clear_data, "w").close()
        for file in filename:
            file_path = os.path.join(self.dir_path, file)
            read_file = open(file_path, "r")
            data_list = list(csv.reader(read_file, delimiter=","))
            read_file.close()
            data_list = data_list[1:]
            data_list = list(map(list, zip(*data_list)))
            if self.garbage:
                clear_data = self.del_garbage(data_list)
            else:
                clear_data = data_list
            self.save_data_to_csv(path_for_clear_data, clear_data)

    def get_raw_data(self):
        file_path = os.path.join(self.clear_data_path, "true_data.csv")
        read_file = open(file_path, "r")
        true_data = list(csv.reader(read_file, delimiter=","))
        read_file.close()
        file_path = os.path.join(self.clear_data_path, "false_data.csv")
        read_file = open(file_path, "r")
        false_data = list(csv.reader(read_file, delimiter=","))
        read_file.close()
        return true_data, false_data

    def conver_str_to_num(self, data, data_type):
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                data[i][j] = data_type(data[i][j])
        return np.array(data)
