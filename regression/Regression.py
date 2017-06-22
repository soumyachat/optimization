'''
Base class for Regression
'''

class RegressionBase(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def loss(self, y_predicted, y):
        pass
