import keras
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.metrics import Precision, Recall

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


def cnn_builder():
    model = Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(300, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def f1_m(history, precision, recall):
    precision = history.history[precision]
    recall = history.history[recall]
    f1 = []
    for prec, rec in zip(precision, recall):
        f1.append(2 * ((prec * rec) / (prec + rec + K.epsilon())))
    return f1


def cnn_2(x_train, y_train, x_test, y_test, shape_size):
    x_train = x_train.reshape((shape_size, 28, 28))
    x_test = x_test.reshape((10000, 28, 28))

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = cnn_builder()
    model.summary()
    history = model.fit(
        x=x_train, y=y_train,
        validation_split=0.1,
        epochs=5
    )

    metrics = pd.DataFrame(history.history)
    print(metrics)
    precision = keras.metrics.Precision()
    #print(precision(y_test))
    #test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    threshold = 0.5
    y_pred = model.predict(x_test)
    predictions = np.where(y_pred > threshold, 1, 0)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions)
    return precision[1], recall[1], f1_score[1]


def cnn(x_train, y_train, x_test, y_test):
    num_classes = 2

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test / 255.0

    model = Sequential()
    model.add(Dense(units=128, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation="softmax"))
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=[Precision(), Recall()])

    batch_size = 512
    epoch = 10
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(x_test, y_test),
        verbose=False
    )

    predictions_prob = model.predict(x_test)
    predictions = np.argmax(predictions_prob, axis=1)

    print(classification_report(y_test, predictions))



