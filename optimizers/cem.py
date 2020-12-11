import numpy as np
from multiprocessing import Pool
from typing import Callable

from scipy.stats import t
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from .optimizer import Optimizer

"""
Cross entropy method for optimization
"""


class CEM(Optimizer):

    def __init__(self, theta, evaluationFunction):
        sigma = 2
        popSize = 20
        numElite = int(0.10*popSize)
        epsilon = 0.0001

        self.name = "Cem"
        self.theta = theta
        self.Sigma = sigma * np.eye(self.theta.size)
        self.popSize = popSize
        self.numElite = numElite
        self.epsilon = epsilon
        self.evaluationFunction = evaluationFunction
        self.iterations = 500
        print(self.name)


    def name(self):
        return self.name


    def run_optimizer(self, verbose=True) -> np.ndarray:
        for iter_count in range(self.iterations):
            values = []
            thetas_to_try = []
            for k in range(1, self.popSize + 1, 1):
                theta_k = np.random.multivariate_normal(self.theta.flatten(), self.Sigma)
                # multiprocessing use
                thetas_to_try.append(theta_k)

            with Pool(8) as p:
                J_dash_list=(p.map(self.evaluationFunction, thetas_to_try))

            values = [(thetas_to_try[i], J_dash_list[i]) for i in range(0, len(thetas_to_try))]

            sorted_values = sorted(values, key=lambda second: second[1])
            theta_values = np.asarray([i[0] for i in sorted_values])[0: self.numElite]
            J_values = np.asarray([i[1] for i in sorted_values])[0: self.numElite]
            self.theta = np.sum(theta_values, axis=0) / self.numElite

            new_J = np.sum(J_values, axis=0) / self.numElite
            dot_theta = 0
            for i in range(self.numElite):
                dot_theta += np.dot(np.reshape(theta_values[i] - self.theta, (-1, 1)),
                                    np.reshape(np.transpose(theta_values[i] - self.theta), (1, -1)))

            self.Sigma = (self.epsilon * np.eye(self.Sigma.shape[0]) + dot_theta) / (
                    self.epsilon + self.numElite)


            if iter_count % 50 == 0 and verbose:
                print(f'At iteration count{iter_count} best objective is {J_values[0]}')
                print(f'Theta Best value is {theta_values[0]}')
                sys.stdout.flush()

        return theta_values[0]




def func(x, e):
    x = x.reshape(-1, 1)
    f = 4 + np.dot(x.T, x)
    return np.squeeze(f)[()]


def test():
    cem = CEM(np.ones((12,1)), func)
    cem.run_optimizer()


if __name__ == "__main__":
    test()
