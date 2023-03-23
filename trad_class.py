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


def decision_tree_classifier():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train_8 = (y_train == 8)
    y_test_8 = (y_test == 8)
    image_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)

    sgd_clf = DecisionTreeClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train_8)

    y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_8, cv=3)
    print(len(y_train_8))
    print(len(y_train_pred))

    print("precision: ", precision_score(y_train_8, y_train_pred))
    print("recall: ", recall_score(y_train_8, y_train_pred))
    print("f1 score: ", f1_score(y_train_8, y_train_pred))


def random_forest_classifier():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train_8 = (y_train == 8)
    y_test_8 = (y_test == 8)
    image_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)

    sgd_clf = RandomForestClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train_8)

    y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_8, cv=3)

    print("precision: ", precision_score(y_train_8, y_train_pred))
    print("recall: ", recall_score(y_train_8, y_train_pred))
    print("f1 score: ", f1_score(y_train_8, y_train_pred))


def sgd_classifier():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train_8 = (y_train == 8)
    y_test_8 = (y_test == 8)
    image_vector_size = 28*28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    some_digit = x_train[-1]
    some_digit_image = some_digit.reshape(28, 28)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x_train, y_train_8)

    y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_8, cv=3)

    print("precision: ", precision_score(y_train_8, y_train_pred))
    print("recall: ", recall_score(y_train_8, y_train_pred))
    print("f1 score: ", f1_score(y_train_8, y_train_pred))


def neural_network_classifier():
    def create_dense(layer_sizes):
        model = Sequential()
        model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))

        for s in layer_sizes[1:]:
            model.add(Dense(units=s, activation='sigmoid'))

        model.add(Dense(units=1, activation='sigmoid'))
        return model

    def evaluate(model, batch_size=128, epochs=5):
        def f1_m(precision, recall):
            precision = history.history[precision]
            recall = history.history[recall]
            f1 = []
            for prec, rec in zip(precision, recall):
                f1.append(2 * ((prec * rec) / (prec + rec + K.epsilon())))
            return f1
        model.summary()
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[Precision(), Recall()])
        history = model.fit(x_train, y_train_8, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
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
    model = create_dense([2048])
    evaluate(model)


if __name__ == '__main__':
    decision_tree_classifier()
