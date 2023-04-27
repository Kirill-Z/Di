import numpy as np


class PUAdapter(object):
    def __init__(self, estimator):
        self.estimator = estimator
        self.c = 1.0
        self.fit = self.__fit_no_precomputed_kernel
        self.estimator_fitted = False

    def __fit_no_precomputed_kernel(self, x_train, y_train, x_test):
        self.estimator.fit(x_train, y_train)
        predictions = self.estimator.predict_proba(x_test)

        try:
            predictions = predictions[:,1]
        except:
            pass

        c = np.mean(predictions)

        self.c = c

        self.estimator_fitted = True

    def predict_proba(self, x_train):
        if not self.estimator_fitted:
            raise Exception("The estimator must be fitted before calling predict_proba(...).")

        probablistic_predictions = self.estimator.predict_proba(x_train)

        try:
            probablistic_predictions = probablistic_predictions[:,1]
        except:
            pass

        return probablistic_predictions / self.c

    def predict(self, x_train, treshold=0.5):
        if not self.estimator_fitted:
            raise Exception("The estimator must be fitted before calling predict(...).")

        return np.array([1. if p > treshold else -1 for p in self.predict_proba(x_train)])

