import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist


def to_binary_classification(dataset, target_true):
    for i, y in enumerate(dataset):
        if y == target_true:
            dataset[i] = 1
        else:
            dataset[i] = 0
    return dataset


def create_dense(layer_sizes):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))

    for s in layer_sizes[1:]:
        model.add(Dense(units=s, activation='sigmoid'))

    model.add(Dense(units=1, activation='sigmoid'))
    return model


def evaluate(model, batch_size=128, epochs=5):
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

    """plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()
    """

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the images
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

num_classes = 2
y_train = to_binary_classification(y_train, 8)
y_test = to_binary_classification(y_test, 8)

y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))


image_size = 784


model = create_dense([2048])
evaluate(model)
