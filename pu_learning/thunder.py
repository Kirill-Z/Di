from load_thunder_data import ThunderData
from wind import WindEstimator

data = ThunderData()

class ThunderEstimator(WindEstimator):
    def __init__(self):
        super().__init__()
        self.num_of_data_list = [88546, 78386, 52257, 26128, 10451, 2090]


if __name__ == '__main__':
    estimator = ThunderEstimator()
    estimator.main()

