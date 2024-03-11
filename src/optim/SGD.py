import numpy as np


class SGD:
    def __init__(self, model, stopping_condition, learning_rate=.001, batch_size=1):
        self.model = model
        self.stopping_condition = stopping_condition
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def optimize(self, x, y):
        while not self.stopping_condition(model=self.model, x=x, y=y):
            sample = np.random.choice(range(len(y)), self.batch_size)
            probs = self.model.predict_probs(x[sample, :])
            self.model.weights = self.model.weights - self.learning_rate * np.mean(
                np.mean(probs - y[sample]) * x[sample], axis=0)
        return self.model
