import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from pulearn import ElkanotoPuClassifier
from load_thunder_data import ThunderData
import pandas as pd
from mnist import convert_to_PU, get_predicted_class, get_estimates, shuffle
from CNN import cnn
from puLearning.adapter import PUAdapter
from wind import WindEstimator

data = ThunderData()

class ThunderEstimator(WindEstimator):
    def __init__(self):
        super().__init__()
        self.num_of_data_list = [88546, 78386, 52257, 26128, 10451, 2090]


if __name__ == '__main__':
    estimator = ThunderEstimator()
    estimator.main()

