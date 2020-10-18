import numpy as np
from scipy.optimize import minimize
from typing import Callable

from scipy.stats import t
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))
sys.path.insert(1, os.path.dirname(sys.path[0]))

from .optimizer import Optimizer


class Powell(Optimizer):

    def __init__(self, theta, evaluationFunction):
        self.name = "Powell"
        self.theta = theta
        self.evaluationFunction = evaluationFunction

    def name(self):
        return self.name

    def run_optimizer(self, verbose=True) -> np.ndarray:
        # Chooses the black-box optimizer we will use (Powell)
        minimizer_method = 'Powell'
        minimizer_options = {'disp': False, 'maxfev': 1, 'maxiter':1}

        # Use Powell to get a candidate solution that tries to maximize candidateObjective
        res = minimize(self.evaluationFunction, x0=self.theta, method=minimizer_method, options=minimizer_options, tol=0.001)

        # Return the candidate solution we believe will pass the safety test
        print(f'x_min inside powell {res.x}')
        print(f'function value at x_min is {self.evaluationFunction(res.x)}')
        return res.x


def func(x):
    f = x * x + 4
    return np.squeeze(f)


def test():
    # pass
    cem = Powell(5.0, func)
    cem.run_optimizer()


if __name__ == "__main__":
    test()
