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

PATH = "archive/"

# production
# categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# testing
categories = ["0", "1"]

data = []
labels = []

# Iterates through every directory in archive/
for index, category in enumerate(categories):
    for file in os.listdir(os.path.join(PATH, category)):
        img_path = os.path.join(PATH, category, file)
        img = imread(img_path)
        gray_img = rgb2gray(img)
        img_resize = resize(gray_img, (8, 8), anti_aliasing=True)

        # Scaling pixel values inside img array
        img_resize = img_resize / np.max(img_resize)
        img_resize = 16 - (img_resize * 16)  # assigns pixels a value btw 0 - 16
        img_resize = np.round(img_resize, 0)  # 0 means the number of decimals

        # Populate Data(after flattening) and Label values
        data.append(img_resize.flatten())

        # Task: get
        labels.append(index)

print(data[0])
print()
print(datasets.load_digits().images[0])
# print(labels)

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
