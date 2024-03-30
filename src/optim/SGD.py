import numpy as np


class SGD:
    def __init__(self, model, stop_condition, learning_rate=.001, batch_size=1):
        self.model = model
        self.stop_condition = stop_condition
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def optimize(self, x, y):
        logliks = []
        while not self.stop_condition(model=self.model, x=x, y=y):
            permutation = np.random.permutation(len(y))
            for ix in range(0, len(y), self.batch_size):
                sample = permutation[ix:ix + self.batch_size]
                probs = self.model.predict_probs(x[sample, :])
                self.model.weights = self.model.weights - self.learning_rate * np.mean(
                    np.mean(probs - y[sample]) * x[sample], axis=0)
            prediction = np.clip(self.model.predict(x), 1e-10, 1 - 1e-10)
            loglik = - np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
            logliks.append(loglik)
        return self.stop_condition.best_model if hasattr(self.stop_condition, 'best_model') else self.model, logliks
