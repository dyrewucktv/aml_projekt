import numpy as np


class MaxIterationCondition:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration = 0

    def __call__(self, **kwargs):
        self.iteration += 1
        if self.iteration > self.max_iterations:
            return True
        return False


class NoLogLikImprovementCondition:
    def __init__(self, threshold=1e-6):
        self.last_scores = []
        self.epoch = -2
        self.threshold = threshold

    def __call__(self, model=None, x=None, y=None, **kwargs):
        self.epoch += 1
        if self.epoch < 0:
            return False
        loglik = np.sum(-np.log(np.exp(model.weights @ x.T)) + y * (model.weights @ x.T))
        if self.last_scores:
            if loglik - self.last_scores[-1] <= self.threshold:
                return True
        self.last_scores.append(loglik)
        return False


class NoLogLikOrMaxIterCondition(NoLogLikImprovementCondition):
    def __init__(self, max_iterations=10, threshold=1e-6):
        super().__init__(threshold)
        self.max_iterations = max_iterations

    def __call__(self, model=None, x=None, y=None, **kwargs):
        if self.epoch > self.max_iterations:
            return True
        super().__call__(model=model, x=x, y=y, **kwargs)
