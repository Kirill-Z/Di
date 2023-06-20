import numpy as np
from keras import layers
from keras.models import Sequential
from sklearn.metrics import precision_recall_fscore_support


def cnn_builder():
    model = Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(300, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def cnn(x_train, y_train, x_test, y_test, shape_size):
    x_train = x_train.reshape((shape_size, 28, 28))
    x_test = x_test.reshape((10000, 28, 28))

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = cnn_builder()
    model.summary()
    model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=5)

    threshold = 0.5
    y_pred = model.predict(x_test)
    predictions = np.where(y_pred > threshold, 1, 0)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, predictions
    )
    return precision[1], recall[1], f1_score[1]
