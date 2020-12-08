import numpy as np
from scipy.optimize import minimize
from typing import Callable
import cma
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
        print(self.name)


    def name(self):
        return self.name

    def run_optimizer(self, verbose=True) -> np.ndarray:
        # Chooses the black-box optimizer we will use (Powell)
        minimizer_method = 'Powell'
        minimizer_options = {'disp': False, 'maxfev': 1000, 'maxiter':1000}

        # Use Powell to get a candidate solution that tries to maximize candidateObjective
        res = minimize(self.evaluationFunction, x0=self.theta, method=minimizer_method, options=minimizer_options, tol=0.001)

        # Return the candidate solution we believe will pass the safety test
        print(f'x_min inside powell {res.x}')
        print(f'function value at x_min is {self.evaluationFunction(res.x)}, Message is {res.message}')
        return res.x


class BFGS(Optimizer):

    def __init__(self, theta, evaluationFunction):
        self.name = "BFGS"
        self.theta = theta
        self.evaluationFunction = evaluationFunction
        print(self.name)


    def name(self):
        return self.name

    def run_optimizer(self, verbose=True) -> np.ndarray:
        # Chooses the black-box optimizer we will use (Powell)
        minimizer_method = 'BFGS'
        minimizer_options = {'disp': True, 'maxfev': 1000, 'maxiter':1000}

        # Use Powell to get a candidate solution that tries to maximize candidateObjective
        res = minimize(self.evaluationFunction, x0=self.theta, method=minimizer_method, options=minimizer_options, tol=0.001)

        # Return the candidate solution we believe will pass the safety test
        print(f'x_min inside bfgs {res.x}')
        print(f'function value at x_min is {self.evaluationFunction(res.x)}, Message is {res.message}')
        return res.x



class CMA(Optimizer):

    def __init__(self, theta, evaluationFunction):
        self.name = "CMA"
        self.theta = theta
        self.evaluationFunction = evaluationFunction
        self.sigma = 0.5
        self.iterations = 5
        print(self.name)


    def name(self):
        return self.name

    def run_optimizer(self, verbose=True) -> np.ndarray:
        # Chooses the black-box optimizer we will use (Powell)
        candidate_solution = cma.fmin(self.evaluationFunction, self.theta.flatten(), self.sigma, options={'maxiter': self.iterations})[0]
        # print(obj_function(candidate_solution))


        # Return the candidate solution we believe will pass the safety test
        print(f'x_min inside cma {candidate_solution}')
        print(f'function value at x_min is {self.evaluationFunction(candidate_solution)}')
        return candidate_solution


def func(x):
    f = x * x + 4
    return np.squeeze(f)


def test():
    # pass
    cem = Powell(5.0, func)
    cem.run_optimizer()


if __name__ == "__main__":
    test()
