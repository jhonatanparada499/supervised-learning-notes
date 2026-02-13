# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm, datasets
from sklearn.model_selection import train_test_split

# Import image processing library 2026-02-10
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

# Import os to handle local files
import os


PATH = "datasets/archive/"

# production
categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# testing
# categories = ["0", "1"]

images = []
labels = []

# Iterates through every directory in archive/
for index, category in enumerate(categories):
    for file in os.listdir(os.path.join(PATH, category)):
        img_path = os.path.join(PATH, category, file)
        img = imread(img_path)
        gray_img = rgb2gray(img)
        resized_img = resize(gray_img, (8, 8), anti_aliasing=True)

        # Apply Data Transformation
        inverted_img = 1.0 - resized_img
        final_img = inverted_img * 16

        if inverted_img.max() > 0:
            final_img = (inverted_img / inverted_img.max()) * 16
        else:
            final_img = inverted_img
        final_img = np.round(final_img, 0)  # 0 means the number of decimals

        # Populate Data(after flattening) and Label values
        images.append(final_img)

        # Task: get
        labels.append(index)

# print(images[0])
# print()
# print(labels[0])
#
#
# flatten the images
n_samples = len(images)
# data = images.reshape((n_samples, -1)) # returns error bc reshape is not a memeber of built-in list
data = np.array([img.ravel() for img in images])

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.5,
    shuffle=True,  # Originally was False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


digits = datasets.load_digits()

n_samples = len(digits.images)
data2 = digits.images.reshape((n_samples, -1))

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data2,
    digits.target,
    test_size=0.5,
    shuffle=False,
)

predicted2 = clf.predict(X_test2)

print(
    f"Classification report for Kaggle dataset:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

print(
    f"Classification report for Scikit-learn dataset:\n"
    f"{metrics.classification_report(y_test2, predicted2)}\n"
)
