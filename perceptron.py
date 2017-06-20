import numpy as np

class Perceptron(object) :
    def __init__(self, learningRate = 0.01, iteration=10):
        self.learningRate = learningRate
        self.iteration = iteration

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []

        for _ in range(self.iteration):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learningRate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
