"""
Linear Regression Class
"""
import numpy as np
from regression import RegressionBase


class LinearRegression(RegressionBase):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self._coeff = None
        self._bias = None
        self.history = []

    def fit(self, X, y, eta=0.0001, max_iteration=1000):
        """

        :param X: Covariate matrix with size (num_samples, num_dimensions). Numpy Array
        :param y: Response vector with size (num_samples). Numpy array
        :return: None
        """
        assert X.shape[0] == y.shape[0], "Invalid sample size in X and y : {}, {}".format(X.shape, y.shape)

        self._theta = self._gd(X, y, eta, max_iteration)
        self._coeff, self._bias = self._theta[1:], self._theta[0]


    def _gd(self, X, y, eta, max_iter):
        """
        Fits a linear regression model using gradient descent
        :param X:
        :param y:
        :return:
        """
        X_padded = np.append(np.ones((X.shape[0],1)), X, axis = 1)
        theta = np.random.rand(X_padded.shape[1]).reshape((X_padded.shape[1], 1))
        self.history = []
        for iter in range(max_iter):
            grad = self._gradient(X_padded, y, theta)
            theta -= eta*grad

            loss = self._score(X_padded, y, theta)
            self.history.append({'score': loss,
                            'gradient': np.sqrt(np.sum(grad**2))})
        return theta

    def _gradient(self, X_padded, y, theta):
        return X_padded.T.dot(X_padded.dot(theta) - y)/float(X_padded.shape[0])

    def _score(self, X_padded, y, theta):
        return np.sum((X_padded.dot(theta) - y)**2)/float(X_padded.shape[0])

    def predict(self, X):
        return X.dot(self._coeff) + self._bias






