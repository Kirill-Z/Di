import numpy as np
from sklearn.model_selection import train_test_split
from load_wind_data import WindData
from mnist import shuffle
from mnist import MnistEstimator


#clear_data_path = "/home/kirill/PycharmProjects/Di/clear_wind_data"
data = WindData()

class WindEstimator(MnistEstimator):
    def __init__(self):
        super().__init__()
        self.num_of_data_list = [964, 861, 574, 287, 114, 22]

    def get_train_test_data(self):
        x_true, x_false = data.get_data()
        y_true = np.ones((len(x_true), 1))
        y_false = np.full((len(x_false), 1), -1.)

        x = np.concatenate((x_true, x_false))
        y = np.concatenate((y_true, y_false))

        x, y = shuffle(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        return x_train, x_test, y_train, y_test


    def main(self):
        for c in self.percent_of_positive_data:
            self.get_estimates("PU learning in progress...", c)
        self.get_estimates("Regular learning in progress...")
        print(self.result)


def get_one_param_data(x_data, num_calc_param):
    x_one_param = []
    for x in x_data:
        x_one_param.append(x[num_calc_param])
    x_one_param = np.array(x_one_param)
    return x_one_param.reshape(-1, 1)

def get_many_param_data(x_data, params):
    x_params = []
    for x in x_data:
        x_params.append(list(x[i] for i in params))
    x_params = np.array(x_params)
    return x_params


if __name__ == '__main__':
    estimator = WindEstimator()
    estimator.main()