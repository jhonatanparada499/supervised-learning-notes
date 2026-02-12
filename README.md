# Supervised Learning Notes

Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning.

## Background knowledge
(Done)[Jupyter Notebook](https://docs.jupyter.org/en/latest/what_is_jupyter.html)

(Done) [Google Introduction to Machine Learning](https://developers.google.com/machine-learning/intro-to-ml)

(Neural networks) [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/linear-regression)

(Neural networks: Multi-class classificatio) [Scikit Learn](https://scikit-learn.org/stable/modules/linear_model.html)

(Done) [Support Vector Machine (SVM) Algorithm](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)

## Google Colab Links
(Done) [linear_regression_taxi.ipynb](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_taxi.ipynb#scrollTo=pkuQNjgoAKYt)

(Done) [binary_classification_rice.ipynb](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/binary_classification_rice.ipynb#scrollTo=qvpUsZF1LDWM)
## Scikit-learn Examples
(Digits dataset) [Recognizing hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

## Requirements
- python3
- pip3
- python3.12-venv

## Environment activation
```bash
source sklearn-env/bin/activate  # activate
```

Source for more detailed installation steps: [Click here](https://scikit-learn.org/stable/install.html#installation-instructions)

## Datasets
[21,600 Handwritten Digits 0-9 from Kaggle](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9)

## Toy ML Examples
### Recognizing Handwritten digits

Dataset: [21,600 Handwritten Digits 0-9 from Kaggle](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9)

Performance Metrics Comparison after Modifying Gamma Hyperparam of the SVM:

gamma = 0.001
Classification report for classifier SVC(gamma=0.001):
              precision    recall  f1-score   support

           0       0.73      0.78      0.75      1139
           1       0.57      0.80      0.66      1117
           2       0.67      0.62      0.64      1132
           3       0.52      0.59      0.55      1100
           4       0.58      0.61      0.59      1075
           5       0.54      0.48      0.51      1045
           6       0.62      0.68      0.65      1053
           7       0.64      0.54      0.59      1056
           8       0.64      0.50      0.56      1039
           9       0.60      0.46      0.52      1022

    accuracy                           0.61     10778
   macro avg       0.61      0.61      0.60     10778
weighted avg       0.61      0.61      0.61     10778

gamma = 0.01
```
```
Classification report for classifier SVC(gamma=0.01):
              precision    recall  f1-score   support

           0       0.90      0.90      0.90      1112
           1       0.80      0.91      0.85      1109
           2       0.86      0.82      0.84      1130
           3       0.76      0.78      0.77      1094
           4       0.75      0.81      0.78      1079
           5       0.84      0.72      0.77      1104
           6       0.77      0.89      0.83      1081
           7       0.84      0.85      0.84      1051
           8       0.82      0.73      0.77      1045
           9       0.80      0.70      0.75       973

    accuracy                           0.81     10778
   macro avg       0.81      0.81      0.81     10778
weighted avg       0.82      0.81      0.81     10778
```
```
