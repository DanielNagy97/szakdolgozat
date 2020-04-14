import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


class Model(object):
    """
    Class for Random Forest Classifier model
    """
    def build_ocr_dataset(self, feature_lenght,
                          source_path='./training_datas/ocr_datas/'):
        """
        Building the dataset from files
        :param source_path: The path of the data's folder
        """
        self._X = np.empty((0, feature_lenght), dtype=np.uint8)
        self._y = np.empty((0, ), dtype=np.uint8)
        self._labels = os.listdir(source_path)

        for label in self._labels:
            for file in os.listdir(source_path+label):
                img = cv2.imread(source_path+label+"/"+file, 0)

                feature_vector = np.array(img).flatten()

                self._X = np.append(self._X, [feature_vector], axis=0)
                self._y = np.append(self._y, [label], axis=0)

    def build_grab_dataset(self, feature_lenght,
                           source_path='./training_datas/grab_datas/'):
        """
        Building the dataset from files
        :param source_path: The path of the data's folder
        """
        self._X = np.empty((0, feature_lenght), dtype=np.uint8)
        self._y = np.empty((0, ), dtype=np.uint8)
        self._labels = os.listdir(source_path)

        for label in self._labels:
            for file in os.listdir(source_path+label):
                feature_vector = \
                    pickle.load(open(source_path+label+"/"+file, 'rb'))
                feature_vector = np.array(feature_vector,
                                          dtype=np.float32).flatten()
                self._X = np.append(self._X, [feature_vector], axis=0)
                self._y = np.append(self._y, [label], axis=0)

        print(self._y.shape)

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
        :param clf_args: Args for the 42 as a dictionary
        """
        self._clf = RandomForestClassifier(**clf_args)
        self._clf.fit(self._X_train, self._Y_train)

    def save_model(self, filename='./trained_models/grab_model.sav'):
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


class Performance(object):

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
    def calc_confusion_matrix(model):
        """
        Calculating confusion matrix
        """
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(model.clf, model.X_test, model.Y_test,
                                         display_labels=model._labels,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix, "\n")

        plt.show()


if __name__ == "__main__":
    classifier_args = dict(random_state=42)
    grab = Model()
    grab.build_grab_dataset(306)
    grab.shuffle_dataset_order()
    grab.split_dataset(0.2, 20)
    grab.train_classifier(classifier_args)

    grab_perf = Performance()
    train_score = grab_perf.cross_val_accuracy(grab.clf,
                                               grab.X_train, grab.Y_train)
    test_score = grab_perf.cross_val_accuracy(grab.clf,
                                              grab.X_test, grab.Y_test)

    print("\n----grab_performance Measures----")
    print("Cross-validation accuracy scores on training data:\n", train_score,
          "\n")
    print("Cross-validation accuracy scores on test data:\n", test_score,
          "\n")

    grab_perf.calc_confusion_matrix(grab)

    to_save = input("Save this model? (y/n)")
    if to_save == "y":
        grab.save_model()

    # ------------

    classifier_args = dict(random_state=42)
    ocr = Model()
    ocr.build_ocr_dataset(165)
    ocr.shuffle_dataset_order()
    ocr.split_dataset(0.3, 7)
    ocr.train_classifier(classifier_args)

    perf = Performance()
    train_score = perf.cross_val_accuracy(ocr.clf, ocr.X_train, ocr.Y_train)
    test_score = perf.cross_val_accuracy(ocr.clf, ocr.X_test, ocr.Y_test)

    print("\n----Performance Measures----")
    print("Cross-validation accuracy scores on training data:\n", train_score,
          "\n")
    print("Cross-validation accuracy scores on test data:\n", test_score,
          "\n")

    perf.calc_confusion_matrix(ocr)

    to_save = input("Save this model? (y/n)")
    if to_save == "y":
        ocr.save_model()