import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        for _ in range(self.epochs):
            gradients = X.T.dot(X.dot(self.theta) - y) / self.m
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        return X.dot(self.theta)


class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        for _ in range(self.epochs):
            for i in range(self.m):
                xi = X[i].reshape(1, -1)
                yi = y[i]
                gradient = xi.T.dot(xi.dot(self.theta) - yi)
                self.theta -= self.learning_rate * gradient

    def predict(self, X):
        return X.dot(self.theta)
