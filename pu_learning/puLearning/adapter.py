import numpy as np


class PUAdapter(object):
    def __init__(self, estimator, hold_out_ratio=0.1):
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.fit = self.__fit_no_precomputed_kernel
        self.estimator_fitted = False

    def __fit_no_precomputed_kernel(self, x, y):
        positives = np.where(y == 1.)[0]

        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        x_hold_out = x[hold_out]
        x = np.delete(x, hold_out, 0)
        y = np.delete(y, hold_out)

        # Обучаем классификатор предсказывать вероятность того, что образец будет помечен P(s=1|x)
        self.estimator.fit(x, y)
        # Используем классификатор для предсказания вероятности того, что положительные образцы будут помечены P(s=1|y = 1)
        predictions = self.estimator.predict_proba(x_hold_out)

        # Получаем вероятность, что предсказана 1
        predictions = predictions[:,1]

        # Получаем среднюю вероятность
        c = np.mean(predictions)

        self.c = c
        #print("len positive data:", len(positives))
        #print(c)
        #exit()

        self.estimator_fitted = True
        return self.estimator, c

    def predict_proba(self, x):
        if not self.estimator_fitted:
            raise Exception("The estimator must be fitted before calling predict_proba(...).")

        probablistic_predictions = self.estimator.predict_proba(x)


        probablistic_predictions = probablistic_predictions[:,1]

        return probablistic_predictions / self.c

    def predict(self, x, threshold=1):
        if not self.estimator_fitted:
            raise Exception("The estimator must be fitted before calling predict(...).")

        return np.array([1. if p > threshold else -1 for p in self.predict_proba(x)])

