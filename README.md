# Supervised Learning Notes

Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning.

## Background knowledge
(Done)[Jupyter Notebook](https://docs.jupyter.org/en/latest/what_is_jupyter.html)

(Done) [Google Introduction to Machine Learning](https://developers.google.com/machine-learning/intro-to-ml)

(Neural networks) [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/linear-regression)

(Neural networks: Multi-class classificatio) [Scikit Learn](https://scikit-learn.org/stable/modules/linear_model.html)

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

## Notes
I started using Google Colab, but professor challenged me to run ML models locally. I copied the code from Colab to a local python script, installed the required modules, but the display behavior seem in Colab did not work locally. I read that I was missing the show() method from plt, but when I did it, another problem/error showed up. So, I read that Jupyter notebooks could handle this problem. It was easy to install and set the web interface for Jupyter notebook and I was able to run the models.
