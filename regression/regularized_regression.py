"""
Regularized Linear Regression.
Currently supports l1-norm, l2-norm regression.
"""
import numpy as np
from regression import RegressionBase


class RegularizedRegression(RegressionBase):
    def __init__(self, regularization=None):
        super(RegularizedRegression, self).__init__()
        self._coeff = None
        self._bias = None
        self.history = []
        self._regularization_coef = None
        if regularization is None:
            print("No regularization method provided, using L2 by default")
            self._regularization = "l2"
        else:
            self._regularization = regularization
        assert regularization in ["l1", "l2"], "Unknown regularization method {}".format(regularization)

    def fit(self, X, y, regularization_coef=0.01, eta=0.0001, max_iteration=1000):
        assert X.shape[0] == y.shape[0], "Invalid sample size in X and y : {}, {}".format(X.shape, y.shape)
        assert regularization_coef > 0, "Only positive values of regularization allowed"
        self._regularization_coef = regularization_coef

        self._theta = self._proximal_gd(X, y, eta, max_iteration)
        self._coeff, self._bias = self._theta[1:], self._theta[0]

    def _proximal_gd(self, X, y, eta, max_iteration):
        """
        Projected Gradient Descent for optimizing regularized linear regression
        :param X: Covariate Matrix, shape (num_samples, dimensions+1)
        :param y: Response vector
        :param eta: learning rate
        :param max_iteration: maximum iteration
        :return: theta: learned coefficient vectoe
        """
        X_padded = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        theta = np.random.rand(X_padded.shape[1]).reshape((X_padded.shape[1], 1))
        self.history = []

        for iter in range(max_iteration):
            grad = self._gradient(X_padded, y, theta)
            theta -= eta * grad

            theta = self._proximal(theta)

            loss = self._score(X_padded, y, theta)
            self.history.append({'score': loss,
                                 'gradient': np.sqrt(np.sum(grad ** 2))})
        return theta

    def _gradient(self, X_padded, y, theta):
        return X_padded.T.dot(X_padded.dot(theta) - y)/float(X_padded.shape[0])

    def _proximal(self, theta):
        '''Proximal operator with respect to Norm ball'''
        if self._regularization == "l2":
            return theta/(self._regularization_coef + 1.0)

        if self._regularization == "l1":
            return self._soft_threshold(theta)

        if self._regularization == "k_support_norm":
            pass

    def _score(self, X_padded, y, theta):
        return np.sum((X_padded.dot(theta) - y)**2)/float(X_padded.shape[0])

    def predict(self, X):
        return X.dot(self._coeff) + self._bias

    def _soft_threshold(self, theta):
        ret = np.zeros(theta.shape)
        ret[0] = theta[0]
        for index, value in enumerate(theta):
            if index > 0: # do not regularize bias term
                if np.abs(value) <= self._regularization_coef:
                    continue
                elif value > self._regularization_coef:
                    ret[index] = value - self._regularization_coef
                else:
                    ret[index] = value + self._regularization_coef
        return ret
