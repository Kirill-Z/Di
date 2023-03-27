from keras.datasets import mnist
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras import backend as K
from scar import scar


def decision_tree_classifier():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #y_train_8 = (y_train == 8)
    #y_test_8 = (y_test == 8)

    x_train, y_train, s_train, len_true_data = scar(x_train, y_train, 8, 1)
    x_test, y_test, s_test, _ = scar(x_test, y_test, 8, 1)

    num_of_data = 0
    if num_of_data != 0:
        end_num_of_data = num_of_data + len_true_data
        if end_num_of_data > 60000:
            print("Слишком большой запрос на количество данных")
            exit(0)
        x_train = x_train[:end_num_of_data]
        y_train = y_train[:end_num_of_data]

    image_vector_size = 28 * 28
    print(len(x_train))
    print(len(y_train))
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)

    sgd_clf = DecisionTreeClassifier()
    sgd_clf.fit(x_train, y_train)

    y_test_pred = cross_val_predict(sgd_clf, x_test, y_test, cv=3)
    y_test_hat = sgd_clf.predict(x_test)
    print(len(y_test))
    print(len(y_test_pred))

    print("precision: ", precision_score(y_test, y_test_hat))
    print("recall: ", recall_score(y_test, y_test_hat))
    print("f1 score: ", f1_score(y_test, y_test_hat))


def random_forest_classifier():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #y_train_8 = (y_train == 8)
    #y_test_8 = (y_test == 8)

    x_train, y_train, s_train, len_true_data = scar(x_train, y_train, 8, 1)
    x_test, y_test, s_test, _ = scar(x_test, y_test, 8, 1)

    num_of_data = 0
    if num_of_data != 0:
        end_num_of_data = num_of_data + len_true_data
        if end_num_of_data > 60000:
            print("Слишком большой запрос на количество данных")
            exit(0)
        x_train = x_train[:end_num_of_data]
        y_train = y_train[:end_num_of_data]

    image_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)

    sgd_clf = RandomForestClassifier()
    sgd_clf.fit(x_train, y_train)

    y_test_pred = cross_val_predict(sgd_clf, x_test, y_test, cv=3)
    y_test_hat = sgd_clf.predict(x_test)

    print("precision: ", precision_score(y_test, y_test_hat))
    print("recall: ", recall_score(y_test, y_test_hat))
    print("f1 score: ", f1_score(y_test, y_test_hat))


def sgd_classifier():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #y_train_8 = (y_train == 8)
    #y_test_8 = (y_test == 8)

    x_train, y_train, s_train, len_true_data = scar(x_train, y_train, 8, 1)
    x_test, y_test, s_test, _ = scar(x_test, y_test, 8, 1)

    num_of_data = 45000
    if num_of_data != 0:
        end_num_of_data = num_of_data + len_true_data
        if end_num_of_data > 60000:
            print("Слишком большой запрос на количество данных")
            exit(0)
        x_train = x_train[:end_num_of_data]
        y_train = y_train[:end_num_of_data]

    image_vector_size = 28*28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)

    sgd_clf = SGDClassifier()
    sgd_clf.fit(x_train, y_train)

    y_test_pred = cross_val_predict(sgd_clf, x_test, y_test, cv=3)
    y_test_hat = sgd_clf.predict(x_test)
    print(len(y_test))
    print(len(y_test_pred))

    print("precision: ", precision_score(y_test, y_test_hat))
    print("recall: ", recall_score(y_test, y_test_hat))
    print("f1 score: ", f1_score(y_test, y_test_hat))


def neural_network_classifier():
    def create_dense(layer_sizes):
        model = Sequential()
        model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))

        for s in layer_sizes[1:]:
            model.add(Dense(units=s, activation='sigmoid'))

        model.add(Dense(units=1, activation='sigmoid'))
        return model

    def evaluate(model, batch_size=128, epochs=40):
        def f1_m(precision, recall):
            precision = history.history[precision]
            recall = history.history[recall]
            f1 = []
            for prec, rec in zip(precision, recall):
                f1.append(2 * ((prec * rec) / (prec + rec + K.epsilon())))
            return f1
        model.summary()
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[Precision(), Recall()])
        history = model.fit(x_train, y_train_8, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_8), verbose=False)
        # y_train_pred = cross_val_predict(model, x_train, y_train_8, cv=3)

        print("precision: ", history.history)
        # print("f1 score: ", f1_score(y_train_8, y_train_pred))
        print("f1: ", f1_m("precision", "recall"))
        print("val_ f1: ", f1_m("val_precision", "val_recall"))
        f1 = f1_m("precision", "recall")
        val_f1 = f1_m("val_precision", "val_recall")

        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('model precision')
        plt.ylabel('precision')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('model recall')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

        plt.plot(f1)
        plt.plot(val_f1)
        plt.title('model f1 score')
        plt.ylabel('f1 score')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train_8 = (y_train == 8)
    y_test_8 = (y_test == 8)
    image_vector_size = 28*28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)
    image_size = 784
    model = create_dense([256])
    evaluate(model)


if __name__ == '__main__':
    sgd_classifier()
