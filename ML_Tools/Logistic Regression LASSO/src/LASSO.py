import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle, os
from urllib.request import urlopen 
from sklearn.linear_model import LinearRegression,LogisticRegression, Ridge, Lasso, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_step_logistic(X, y, w_, lr=0.01):
    """One step gradient descent for logistic regression"""
    #return w - lr * np.einsum('ij,j->i', X.T,(sigmoid(y * (X @ w)) - 1.0)*y)
    return w_ - lr * (X.T@((sigmoid(y*(X@w_))-1)*y))

def gradient_descent_logistic(X, y, w_init, lr=0.01, n_iter=100):
    """Gradient descent for logistic regression with multiple iterations"""
    w = w_init
    ws = []
    for i in range(n_iter):
        w = gradient_step_logistic(X, y, w, lr) #gradient of logistic loss with y = -1 or y = 1
        ws.append(w)
    return ws

def stochastic_gradient_descent_logistic(X, y, w_init, lr=0.01, n_iter=100, batch_size = 10):
    w = w_init
    ws = []
    for i in range(n_iter):
        indices = np.random.randint(0, X.shape[0], size=batch_size)
        w = gradient_step_logistic(X[indices], y[indices], w, lr)
        ws.append(w)
    return ws

def plot_decision_boundaries(X_test,y_test,ridge_classifier, logistic_regression, accuracy_ridge,accuracy_logistic):
    """
    arguments : 
        - X_test, y_test : test data
        - ridge_classifier : instance of the class RidgeClassifier
        - logistic_regression : instance of the class LogisticRegression
        - accuracy_ridge : accuracy of the ridge classifier
        - accuracy_logistic : accuracy of the logistic regression
    """
    # Plot the decision boundary for Ridge Classifier
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
    plt.title(f"Ridge Classifier\nAccuracy: {accuracy_ridge:.2f}")
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = ridge_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    
    # Plot the decision boundary for Logistic Regression
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
    plt.title(f"Logistic Regression\nAccuracy: {accuracy_logistic:.2f}")
    ax = plt.gca()
    Z = logistic_regression.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)
    
    plt.show()

    def ridge_train(X,y,alpha):
        n_samples, n_features = X.shape
        X_augmented = np.column_stack((X, np.ones(n_samples)))  # Add a bias term

        identity_matrix = np.eye(n_features + 1)
        coefficients = np.linalg.inv(X_augmented.T @ X_augmented + alpha * identity_matrix) @ X_augmented.T @ y

        coef_ = coefficients[:-1]
        intercept_ = coefficients[-1]
        return coef_, intercept_

def ridge_predict(X,coef_,intercept_):
    preds = X @ coef_ + intercept_
    return np.sign(preds)

#Load Ising data
def load_data():
    """
    Loads the data of the Ising model. The labels correspond to ordered (1) and disordered states (0).
    """
    # url to data
    url_main = 'https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/';
    ######### LOAD DATA
    # The data consists of 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25):
    data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 
    # The labels are obtained from the following file:
    label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"
    #DATA
    data = pickle.load(urlopen(url_main + data_file_name)) # pickle reads the file and returns the Python object (1D array, compressed bits)
    data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
    data=data.astype('int')
    data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

    #LABELS (convention is 1 for ordered states and 0 for disordered states)
    labels = pickle.load(urlopen(url_main + label_file_name)) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)
    return data, labels


def linear_regression_pi(X, y):
    """
    arguments:
    X : data matrix
    y : output
    returns:
    the least square estimator "w"
    """
    w = X.T@np.linalg.inv(X.dot(X.T)).dot(y)
    return w

def linear_regression_ridge(X, y, reg=1):
    """
    arguments:
    X : data matrix
    y : output
    returns:
    the least square estimator "w"
    """
    w = (np.linalg.inv(X.T.dot(X)+reg*np.eye(X.shape[1]))@X.T).dot(y)
    return w

def gradient_descent_step(X,y,w,eta):
    """
    Do one gradient step of OLS
    """
    grad = X.T@(X@w-y)
    w = w - eta * grad 
    return w

def train(n_iter,X,y,w,eta):
    for i in tqdm(range(n_iter)):
        w = gradient_descent_step(X,y,w,eta)
    return w

def linear_regression(X, y):
    """
    X : data matrix
    y : output

    return:
    w : the least square estimator
    """
    w = np.linalg.inv(X.T.dot(X))@X.T.dot(y)
    return w