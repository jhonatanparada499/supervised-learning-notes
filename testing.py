# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
# from matplotlib.pyplot import imread

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Source: phttps://medium.com/@sreuniversity/unlocking-image-classification-with-scikit-learn-a-journey-into-computer-vision-af2cdc881ad
import os
from skimage.io import imread
from skimage.transform import resize

digits = datasets.load_digits()
print(digits.images[0])
print()

# # flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
print(data[0])
print()

IMG_PATH = "Zero_full (1).jpg"
img = imread(IMG_PATH)
img_resize = resize(img, (8, 8))

print(img)

#
# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)
#
# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )
#
# # Learn the digits on the train subset
# clf.fit(X_train, y_train)
#
# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)
