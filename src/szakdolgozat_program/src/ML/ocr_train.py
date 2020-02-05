import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class Ocr_model(object):
    """
    Class for training the gesture recogniser model
    using Random Forest Classifier
    """
    def build_dataset(self, source_path='./training_datas/ocr_datas/'):
        """
        Building the dataset from files
        :param source_path: The path of the data's folder
        """
        self._X = np.empty((0, 165), dtype=np.uint8)
        self._y = np.empty((0, ), dtype=np.uint8)
        self._labels = os.listdir(source_path)

        for label in self._labels:
            for file in os.listdir(source_path+label):
                img = cv2.imread(source_path+label+"/"+file, 0)

                feature_vector = np.array(img).flatten()

                self._X = np.append(self._X, [feature_vector], axis=0)
                self._y = np.append(self._y, [label], axis=0)

    def shuffle_dataset_order(self):
        """
        Randomize dataset order
        """
        shuffle_index = np.random.permutation(len(self._y))
        self._X, self._y = self._X[shuffle_index], self._y[shuffle_index]

    def split_dataset(self, test_size, seed):
        """
        Splitting dataset to train and test partitions
        :param test_size: Size of the test partition in percentage
        :param seed: random_state value for splitting
        """
        self._X_train, self._X_test, self._Y_train, self._Y_test = \
            train_test_split(self._X, self._y,
                             test_size=test_size,
                             random_state=seed)
        print("Training dataset size:", len(self._Y_train))
        print("Testing dataset size:", len(self._Y_test))

    def train_classifier(self, clf_args):
        """
        Fitting the training data to the classifier
        :param clf_args: Args for the classifier as a dictionary
        """
        self._clf = RandomForestClassifier(random_state=42)
        self._clf.fit(self._X_train, self._Y_train)

    def save_model(self, filename='./trained_models/ocr_model.sav'):
        """
        Saving the model for later use
        :param filename: Name of the file
        """
        pickle.dump(self._clf, open(filename, 'wb'))
        print("Saved to:", filename)

    @property
    def X_train(self):
        return self._X_train

    @property
    def Y_train(self):
        return self._Y_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def clf(self):
        return self._clf


class Ocr_performance(object):

    @staticmethod
    def cross_val_accuracy(clf, X, Y, folding=3):
        """
        Calculating Cross-Validation accuracy scores
        """
        scores = cross_val_score(clf,
                                 X, Y,
                                 cv=3, scoring="accuracy")
        return scores

    @staticmethod
    def calc_confusion_matrix(clf, X, Y, folding=3):
        """
        Calculating confusion matrix
        """
        y_pred = cross_val_predict(clf, X, Y, cv=3)
        confusion = confusion_matrix(Y, y_pred)
        return confusion

    @staticmethod
    def normalize_confusion_matrix(conf_mx):
        """
        Normalising confusion matrix
        """
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        return norm_conf_mx

    @staticmethod
    def plot_confusion_matrix(confusion_matrix):
        """
        Plotting confusion matrix
        """
        plt.matshow(confusion_matrix, cmap=plt.cm.gray)
        plt.show()

    @staticmethod
    def test_prediction(clf, X_test, index=19):
        """
        testing prediction on a specified element
        """
        some_element = X_test[index]

        some_element_prediction = clf.predict([some_element])
        some_element_decision = clf.predict_proba([some_element])
        print(some_element_prediction)
        print(some_element_decision)
        while True:
            cv2.imshow("se", some_element.reshape(11, 15))
            k = cv2.waitKey(100) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    classifier_args = {
        "randomstate": 42
    }
    ocr = Ocr_model()
    ocr.build_dataset()
    ocr.shuffle_dataset_order()
    ocr.split_dataset(0.3, 7)
    ocr.train_classifier(classifier_args)

    perf = Ocr_performance()
    train_score = perf.cross_val_accuracy(ocr.clf, ocr.X_train, ocr.Y_train)
    test_score = perf.cross_val_accuracy(ocr.clf, ocr.X_test, ocr.Y_test)

    train_conf_mx = perf.calc_confusion_matrix(ocr.clf,
                                               ocr.X_train, ocr.Y_train)
    test_conf_mx = perf.calc_confusion_matrix(ocr.clf,
                                              ocr.X_test, ocr.Y_test)

    normalized_train_conf_mx = perf.normalize_confusion_matrix(train_conf_mx)
    normalized_test_conf_mx = perf.normalize_confusion_matrix(test_conf_mx)

    perf.test_prediction(ocr.clf, ocr.X_test)

    # perf.plot_confusion_matrix(train_conf_mx)
    # perf.plot_confusion_matrix(test_conf_mx)

    print("\n----Performance Measures----")
    print("Cross-validation accuracy scores on training data:\n", train_score,
          "\n")
    print("Cross-validation accuracy scores on test data:\n", test_score,
          "\n")
    print("Confusion matrix for training data:\n", train_conf_mx, "\n")
    print("Normalized confusion matrix for training data:\n",
          normalized_train_conf_mx,
          "\n")
    print("Confusion matrix for test data:\n", test_conf_mx, "\n")
    print("Normalized confusion matrix for test data:\n",
          normalized_test_conf_mx,
          "\n")

    to_save = input("Save this model? (y/n)")
    if to_save == "y":
        ocr.save_model()
