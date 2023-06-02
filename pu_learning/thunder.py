from sklearn.ensemble import RandomForestClassifier

from load_thunder_data import ThunderData
from wind import WindEstimator


class ThunderEstimator(WindEstimator):
    def __init__(self, data, estimator, neural_network):
        super().__init__(data, estimator, neural_network)
        self.num_of_data_list = [88546, 78386, 52257, 26128, 10451, 2090]


if __name__ == "__main__":
    data = ThunderData(garbage=False)
    estimator = ThunderEstimator(
        data=data, estimator=RandomForestClassifier(n_jobs=4), neural_network=False
    )
    estimator.main()
