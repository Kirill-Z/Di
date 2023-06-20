import os

import numpy as np
import pandas as pd

from load_data import Data


class ThunderData(Data):
    def __init__(self, garbage, dir_path, clear_dir_path):
        super().__init__(garbage)
        self.dir_path = dir_path
        self.clear_data_path = clear_dir_path

    def get_data(self):
        true_data, false_data = self.get_raw_data()
        return self.conver_str_to_num(true_data, float), self.conver_str_to_num(
            false_data, float
        )

    def del_garbage(self, data):
        data = data.replace([np.inf, -np.inf], np.nan)
        data[data > 3.40282346638528859811704183484516925440e30] = np.nan
        data.dropna(inplace=True)
        return data

    def separate_false_and_true_data(self):
        open(os.path.join(self.clear_data_path, "true_data.csv"), "w").close()
        open(os.path.join(self.clear_data_path, "false_data.csv"), "w").close()
        for file in os.listdir(self.dir_path):
            read_file = pd.read_csv(
                os.path.join(self.dir_path, file), usecols=[i for i in range(91)]
            )
            true_data = read_file[read_file["TunderYN"] == 1.0]
            false_data = read_file[read_file["TunderYN"] == 0.0]
            true_data = self.del_garbage(true_data)
            false_data = self.del_garbage(false_data)
            true_data = true_data.iloc[:, 0:90]
            false_data = false_data.iloc[:, 0:90]
            true_data.to_csv(
                os.path.join(self.clear_data_path, "true_data.csv"),
                mode="a",
                header=None,
                index=False,
            )
            false_data.to_csv(
                os.path.join(self.clear_data_path, "false_data.csv"),
                mode="a",
                header=None,
                index=False,
            )
