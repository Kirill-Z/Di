import csv

def read_txt(file_name):
    file_path = "/home/kirill/PycharmProjects/Di/pu_learning/" + file_name
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return data