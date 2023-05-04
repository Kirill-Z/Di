import csv
import glob
import pathlib
import os
import pandas as pd
dir_path = "/home/kirill/PycharmProjects/Di/data"


def read_txt():
    true_files = []
    false_files = []
    for file in os.listdir(dir_path):
        if file.endswith("true.txt"):
            true_files.append(file)
        if file.endswith("false.txt"):
            false_files.append(file)

    return true_files, false_files


def txt_to_csv(file):
    for f in file:
        read_file = pd.read_csv(os.path.join(dir_path, f), sep=r"\s+")
        new_file_name = os.path.splitext(f)[0] + ".csv"
        read_file.to_csv(os.path.join(dir_path, new_file_name), index=False)


def read_csv(file_name):
    file_path = os.path.join(dir_path, file_name)
    file = open(file_path, "r")
    data = list(csv.reader(file, delimiter=","))
    data = list(map(list, zip(*data)))
    file.close()
    return data


def del_garbage(data):
    for i in data:
        for j in i:
            if "*" in j:
                del i
    return data


true, false = read_txt()
txt_to_csv(true)
txt_to_csv(false)
print(true[0])
data = read_csv("20674_fog_false.csv")
print(len(data))
data = del_garbage(data)
print(len(data))
#for row in data:
#    print(row)



