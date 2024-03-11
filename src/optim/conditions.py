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
    def __init__(self, no_loglik_improvement_iters=1):
        self.no_loglik_improvement_iters = no_loglik_improvement_iters
        self.last_scores = []
        self.epoch = -2

    def __call__(self, model=None, x=None, y=None, **kwargs):
        self.epoch += 1
        if self.epoch < 0:
            return False
        loglik = np.sum(-np.log(np.exp(model.weights @ x.T)) + y * (model.weights @ x.T))
        if self.last_scores:
            if loglik <= np.mean(self.last_scores) and len(self.last_scores) >= self.no_loglik_improvement_iters:
                return True
            if len(self.last_scores) == self.no_loglik_improvement_iters:
                self.last_scores.pop(0)
        self.last_scores.append(loglik)
        return False
