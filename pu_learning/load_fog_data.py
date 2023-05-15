import csv
import os
import pandas as pd
import numpy as np

class Data:
    def __init__(self):
        self.dir_path = "/home/kirill/PycharmProjects/Di/data"
        self.clear_data_path = "/home/kirill/PycharmProjects/Di/clear_data"

    def __call__(self):
        true, false = self.read_txt()
        filename_true = self.txt_to_csv(true)
        filename_false = self.txt_to_csv(false)
        self.read_csv(filename_true, "true_data.csv")
        self.read_csv(filename_false, "false_data.csv")

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
            read_file = pd.read_csv(os.path.join(self.dir_path, f), sep=r"\s+", header=None, dtype=str)
            new_file_name = os.path.splitext(f)[0] + ".csv"
            read_file.to_csv(os.path.join(self.dir_path, new_file_name), index=False)
            filename.append(new_file_name)
        return filename



    def save_data_to_csv(self, filename, data):
        with open(filename, "ab") as f:
            f.write(b"")
            np.savetxt(f, data, delimiter=',', fmt ='%s')


    def read_csv(self, filename: list, new_file_name):
        path_for_clear_data = os.path.join(self.clear_data_path, new_file_name)
        open(path_for_clear_data, 'w').close()
        for file in filename:
            file_path = os.path.join(self.dir_path, file)
            read_file = open(file_path, "r")
            data_list = list(csv.reader(read_file, delimiter=","))
            read_file.close()
            data_list = data_list[1:]
            data_list = list(map(list, zip(*data_list)))
            clear_data = self.del_garbage(data_list)
            self.save_data_to_csv(path_for_clear_data, clear_data)


    def del_garbage(self, data):
        len_data = len(data)
        i = 0
        while i < len_data:
            for j in range(len(data[i])):
                if data[i][j] == "" or "*" in data[i][j]:
                    data.pop(i)
                    len_data -= 1
                    i =+ 1
                    if i < len_data:
                        data.pop(i)
                        len_data -= 1
                        i =+ 1
                    break

            i += 1
        return data

    def get_data(self):
        file_path = os.path.join(self.clear_data_path, "true_data.csv")
        read_file = open(file_path, "r")
        true_data = list(csv.reader(read_file, delimiter=","))
        read_file.close()
        file_path = os.path.join(self.clear_data_path, "false_data.csv")
        read_file = open(file_path, "r")
        false_data = list(csv.reader(read_file, delimiter=","))
        read_file.close()
        return self.conver_str_to_int(true_data), self.conver_str_to_int(false_data)

    def conver_str_to_int(self, data):
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                data[i][j] = int(data[i][j])
        return np.array(data)
