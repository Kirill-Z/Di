import numpy as np


class LabelingMechanism:
    def __init__(
        self,
        propensity_attributes,
        propensity_attributes_signs,
        min_prob=0.0,
        max_prob=1.0,
        power=1,
    ):
        assert len(propensity_attributes) == len(
            propensity_attributes_signs), "size of attributes and signs must be same"
        self.propensity_attributes = np.array(propensity_attributes)
        self.propensity_attributes_signs = np.array(propensity_attributes_signs)
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.power = power

        self.min_x = None
        self.max_x = None

    def fit(self, xs):
        xs_ = xs[:, self.propensity_attributes] * self.propensity_attributes_signs
        self.min_x = xs_.min(0)
        self.max_x = xs_.max(0)

    def propensity_score(self, xs):
        assert self.min_x is not None and self.max_x is not None, "run fit() before calculating propensity score"
        xs_ = xs[:, self.propensity_attributes] * self.propensity_attributes_signs
        scaled = self.min_prob + (((xs_ - self.min_x) / (self.max_x - self.min_x)) ** self.power) * (self.max_prob - self.min_prob)
        es = (scaled ** (1 / len(self.propensity_attributes))).prod(1)  # geometric mean
        return es

    @staticmethod
    def label_frequency(es, ys):
        es_pos = es[ys == 1]
        return es_pos.mean()


lm = LabelingMechanism([0], [1], min_prob=0.0, max_prob=1.0, power=4)
