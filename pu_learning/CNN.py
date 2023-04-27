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

from sklearn.metrics import classification_report


def cnn_builder():
    model = Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(2),
    ])


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def cnn_2(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape((60000, 28, 28))
    x_test = x_test.reshape((10000, 28, 28))

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = cnn_builder()
    model.summary()
    history = model.fit(x=x_train, y=y_train,
                        validation_data=(x_test, y_test),
                        epochs=10, batch_size=32)

    metrics = pd.DataFrame(history.history)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print(metrics.head())
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions_prob = probability_model.predict(x_test)
    predictions = np.argmax(predictions_prob, axis=1)
    print(predictions)
    print(classification_report(y_test, predictions))
    exit()


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

    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

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



