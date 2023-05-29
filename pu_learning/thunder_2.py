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

data = ThunderData()

def evaluate_results(y_test, y_pred):
    print('Classification results:')
    f1 = f1_score(y_test, y_pred, average='micro')
    print("f1: %.2f%%" % (f1 * 100.0))
    rec = recall_score(y_test, y_pred, average='micro')
    print("recall: %.2f%%" % (rec * 100.0)) 
    prc = precision_score(y_test, y_pred, average='micro')
    print("precision: %.2f%%" % (prc * 100.0))


def fit_PU_estimator(X, y, hold_out_ratio, estimator):
    # find the indices of the positive/labeled elements
    assert (type(y) == np.ndarray), "Must pass np.ndarray rather than list as y"
    positives = np.where(y == 1.)[0]

    print("positives", len(positives))
    print("neg", len(np.where(y == -1.)[0]))
    # hold_out_size = the *number* of positives/labeled samples
    # that we will use later to estimate P(s=1|y=1)
    hold_out_size = int(np.ceil(len(positives) * hold_out_ratio))
    np.random.shuffle(positives)
    # hold_out = the *indices* of the positive elements
    # that we will later use  to estimate P(s=1|y=1)
    hold_out = positives[:hold_out_size]
    # the actual positive *elements* that we will keep aside
    X_hold_out = X[hold_out]
    # remove the held out elements from X and y
    X = np.delete(X, hold_out, 0)
    y = np.delete(y, hold_out)
    # Обучаем классификатор предсказывать вероятность того, что образец будет помечен P(s=1|x)
    estimator.fit(X, y)
    # Используем классификатор для предсказания вероятности того, что положительные образцы будут помечены P(s=1|y = 1)
    hold_out_predictions = estimator.predict_proba(X_hold_out)
    print(hold_out_predictions)
    # Получаем вероятность, что предсказана 1
    hold_out_predictions = hold_out_predictions[:, 1]
    # Получаем среднюю вероятность
    c = np.mean(hold_out_predictions)
    print("c= ", c)
    return estimator, c


def predict_PU_prob(X, estimator, prob_s1y1):
    predicted_s = estimator.predict_proba(X)
    predicted_s = predicted_s[:, 1]
    return predicted_s / prob_s1y1
    

def main():
    x_true, x_false = data.get_data()
    y_true = np.ones((len(x_true), 1))
    y_false = np.full((len(x_false), 1), -1.)

    x = np.concatenate((x_true, x_false))
    y = np.concatenate((y_true, y_false))

    x, y = shuffle(x, y)
    X, X_test_data, y_train, y_test_data = train_test_split(x, y, test_size=0.2)

    """model = RandomForestClassifier(n_jobs=4)
    model.fit(X, y_train.ravel())

    y_pred = model.predict(X_test_data)

    evaluate_results(y_test_data, y_pred)"""

    c_list = [0.5]
    len_data = len(X)
    num_of_data_list = [int(len_data*0.5)]

    X_test, y_test = shuffle(X_test_data, y_test_data)
    x_for_pu, y_for_pu, s_for_pu, _ = convert_to_PU(x, y, c_list[0], num_of_data_list[0], len_data)

    predicted = np.zeros(len(x))
    learning_iterations = 1
    #for index in range(learning_iterations):
    pu_estimator, probs1y1 = fit_PU_estimator(x_for_pu, s_for_pu, 0.2, RandomForestClassifier(n_jobs=4))
    predicted += predict_PU_prob(x, pu_estimator, probs1y1)
    #    if (index % 4 == 0):
    #        print(f'Learning Iteration::{index}/{learning_iterations} => P(s=1|y=1)={probs1y1}')

    y_predict = [1 if y > 0.5 else 0 for y in (predicted / learning_iterations)]
    print("positive predicted:", len(np.where(y_predict == 1)[0]))
    print("neg predicted:", len(np.where(y_predict == 0)[0]))
    evaluate_results(y, y_predict)

if __name__ == '__main__':
    main()







