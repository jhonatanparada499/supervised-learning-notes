# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Source: phttps://medium.com/@sreuniversity/unlocking-image-classification-with-scikit-learn-a-journey-into-computer-vision-af2cdc881ad
import os  # used later to process thousends of images
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

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
gray_img = rgb2gray(img)
resized_img = resize(gray_img, (8, 8), anti_aliasing=True)

# Scaling values inside the img array
resized_img = resized_img / np.max(
    resized_img
)  # calcs pixel intensity relative to higher one in the img
resized_img = 16 - (resized_img * 16)  # assigns pixels a value btw 0 - 16
resized_img = np.round(resized_img, 0)  # 0 means the number of decimals

print(resized_img)

# img = resize(img, (8, 8))
# # img_arr = np.array(img)
# # img_arr = 16 - (img_arr / 16.0)
# flat_data = img.flatten()
# flat_data = np.clip(flat_data, 0, 16)

# print(flat_data)

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
