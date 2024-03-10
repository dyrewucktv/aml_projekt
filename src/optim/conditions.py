

class MaxIterationCondition:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration = 0

    def __call__(self, **kwargs):
        self.iteration += 1
        if self.iteration > self.max_iterations:
            return True
        return False
