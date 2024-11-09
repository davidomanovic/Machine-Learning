import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

def generate_data(n, a, variance):
    X = np.random.normal(size=(n,))
    y = X * a + np.sqrt(variance) * np.random.normal(size=(n,))
    return X.reshape((-1,1)), y

def linear_regression(X, y):
    """
    arguments:
        - X : input data matrix
        - y : output
    returns:
        - w : the least square estimator
    """
    lambda_ = 1.0
    X = np.concatenate((X, np.ones((len(X), 1))), axis=1) # Add column for bias
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def gen_data_poly(p, std_dev, n):
    """
    arguments:
        - p  : list of coefficients representing the polynom (polynome[i] multiplies X^i)
        - std_dev : standard deviation of the gaussian noise
        - n         : number of data points
    returns: 
        - (X, y)    : list of n points with y = P(x) + gaussian noise
    """
    d = len(p) - 1 #degree of polynomial
    samples = np.random.uniform(-2, 2, size=n)
    X = np.copy(samples)
    samples = np.ones((n, d + 1)).T *samples
    exp = np.power(samples.T, np.arange(0, d+1, 1)) # exponentials
    y = np.sum(exp * p, axis=1) + np.random.normal(0, std_dev, size=n)
    return X, y