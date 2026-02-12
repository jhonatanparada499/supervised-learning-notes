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

PATH = "archive/"

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

        # Scaling pixel values inside img array
        # img_resize = img_resize / np.max(img_resize)
        # img_resize = 16 - (img_resize * 16)  # assigns pixels a value btw 0 - 16
        # img_resize = np.round(img_resize, 0)  # 0 means the number of decimals
        #
        # # Populate Data(after flattening) and Label values
        # images.append(img_resize)
        #
        # # Task: get
        # labels.append(index)

# print(images[0])
# print()
# print(labels[0])
#

# flatten the images
n_samples = len(images)
# data = images.reshape((n_samples, -1)) # returns error bc reshape is not a memeber of built-in list
data = np.array([img.ravel() for img in images])

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.01)

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

print(
    f"Classification report for 50% of Kaggle dataset"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# In This section my model trained with a Kaggle digits dataset
# makes predictions on Scikit-learn digits dataset.

digits = datasets.load_digits()

print(digits.images[0])
n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
# Split data into 50% train and 50% test subsets
# X_train2, X_test2, y_train2, y_test2 = train_test_split(
#     data,
#     digits.target,
#     test_size=0.5,
#     shuffle=False,
# )

sample_index = 0
single_sample = digits.data.reshape(n_samples, -1)[0]

print("sample from scikit learn")
print(single_sample)
print("sample from custom")
print(data[0])

# Predict the value of the digit on the test subset
predicted2 = clf.predict([single_sample])

print(predicted2)

# print(
#     f"Classification report for 50% of Scikit-learn dataset"
#     f"{metrics.classification_report(y_test2, predicted2)}\n"
# )
