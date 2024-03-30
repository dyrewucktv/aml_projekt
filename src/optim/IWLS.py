import numpy as np


class IWLS:
    """
        def optimize(self, x, y):
        while not self.stop_condition(model=self.model, x=x, y=y):
            p = self.model.predict_probs(x)
            w = np.diag(np.maximum(p * (1 - p), self.delta))
            z = x @ self.model.weights + np.linalg.inv(w) @ (y - p)
            b_new = np.linalg.inv(
                (1 - self.lambda_) * (x.T @ w @ x)
                + self.lambda_ * np.identity(self.model.weights.shape[0])) @ x.T @ w @ z
            self.model.weights = b_new
        return self.model
    """

    def __init__(self, model, stop_condition, delta=1e-5, lambda_=1e-5):
        self.model = model
        self.stop_condition = stop_condition
        self.delta = delta
        self.lambda_ = lambda_

    def optimize(self, x, y):
        while not self.stop_condition(model=self.model, x=x, y=y):
            p = self.model.predict_probs(x)
            w = np.maximum(p * (1 - p), self.delta)
            z = (x @ self.model.weights + ((1 / w) * (y - p))[:, np.newaxis].T).T
            b_new = np.linalg.inv(
                (1 - self.lambda_) * (x.T @ (x * w[:, np.newaxis]))
                + self.lambda_ * np.identity(self.model.weights.shape[0])) @ x.T @ (z * w[:, np.newaxis])
            self.model.weights = np.squeeze(b_new)
        return self.stop_condition.best_model if hasattr(self.stop_condition, 'best_model') else self.model
