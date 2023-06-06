import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from estimator import Estimator, shuffle
from load_wind_data import WindData


class WindEstimator(Estimator):
    def __init__(self, data, estimator, neural_network):
        self.data = data
        super().__init__(data, estimator, neural_network)
        self.percent_of_positive_data = [0.7]
        self.num_of_data_list = [540]

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


if __name__ == "__main__":
    data = WindData(garbage=True)
    estimator = WindEstimator(
        data=data, estimator=RandomForestClassifier(n_jobs=4), neural_network=False
    )
    estimator.main()
