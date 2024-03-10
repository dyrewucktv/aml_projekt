import numpy as np


class IWLS:
    def __init__(self, model, stop_condition, delta=1e-5):
        self.model = model
        self.stop_condition = stop_condition
        self.delta = delta

    def optimize(self, x, y):
        while not self.stop_condition(model=self.model, x=x, y=y):
            p = self.model.predict_probs(x)
            w = np.diag(np.maximum(p * (1 - p), self.delta))
            z = x @ self.model.weights + np.linalg.inv(w) @ (y - p)
            b_new = np.linalg.inv(x.T @ w @ x) @ x.T @ w @ z
            self.model.weights = b_new
        return self.model
