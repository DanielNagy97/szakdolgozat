import cv2
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import pickle

# Reading datas and making the dataset
path = "./training_datas/ocr_datas/"
X = []
y = []
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)

        # reading image in grayscale mode
        img = cv2.imread(path+directory+"/"+file, 0)

        feature_vector = np.array(img).flatten()

        X.append(feature_vector)
        y.append(directory)

# Randomize dataset order
shuffle_index = np.random.permutation(len(y))
X, y = np.array(X)[shuffle_index], np.array(y)[shuffle_index]

# Splitting dataset to train and test partitions
X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(X, y,
                                     test_size=0.33,
                                     random_state=7)

# Defining classifier and fitting training data to it
clf = SGDClassifier(random_state=42)
clf.fit(X_train, Y_train)

# score by cross-validation
scores = cross_val_score(clf, X_test, Y_test)
print(scores.mean())

# testing prediction on a specified element
# testing on the 20th element of the test dataset
some_element = X_test[20]
some_element_prediction = clf.predict([some_element])
some_element_decision = clf.decision_function([some_element])
print(some_element_prediction)
print(some_element_decision)

# saving the trained model
filename = './trained_models/ocr_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# loading the model for later use
"""
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
"""
