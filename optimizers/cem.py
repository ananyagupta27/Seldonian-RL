import numpy as np

from typing import Callable

from scipy.stats import t
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))
sys.path.insert(1, os.path.dirname(sys.path[0]))
from .optimizer import Optimizer


class CEM(Optimizer):

    def __init__(self, theta, evaluationFunction):
        sigma = 0.5
        popSize = 10
        numElite = 5
        epsilon = 0.0001

        self.name = "Cem"
        self.theta = theta
        self.Sigma = sigma * np.eye(self.theta.size)
        self.popSize = popSize
        self.numElite = numElite
        self.epsilon = epsilon
        self.evaluationFunction = evaluationFunction
        self.iterations = 200


    def name(self):
        return self.name

    # def setNumEpisodes(self, numEpisodes):
    #     self._numEpisodes = numEpisodes

    def run_optimizer(self, verbose=True) -> np.ndarray:
        for iter_count in range(self.iterations):
            values = []
            for k in range(1, self.popSize + 1, 1):
                theta_k = np.random.multivariate_normal(self.theta.flatten(), self.Sigma)
                J_dash = self.evaluationFunction(theta_k)
                values.append((theta_k, J_dash))
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

            #
            if iter_count % 10 == 0 and verbose:
                print(f'At iteration count{iter_count} best objective is {J_values[0]}')
                # print(f'Theta value is {theta_values[0]}')
                sys.stdout.flush()

        return theta_values[0]






def main():
    episodes = int(sys.argv[1])
    for _ in range(10):
        theta = np.zeros((256 * 2))
        #     theta = np.zeros((25, 4))
        evaluationFunction = CandidateCartpolePDIS
        cem = CEM(theta, evaluationFunction)
        # print(evaluationFunction(cmaes.generation_loop()))
        for it in range(1000):
            x_min = cem.run_optimizer(episodes)

            # if it % 100 == 0:
            #     print("episodes", episodes)
            #     # print("x_min:=", x_min)
            #     print("f_min:=", CandidateCartpolePDIS(x_min, episodes, multiplier=1))
            #     sys.stdout.flush()
        print("--------------------------")
        print("x_min:=", x_min)
        print("f_min:=", evaluationFunction(x_min, episodes, multiplier=1))
        sys.stdout.flush()
    # # cmaes = CEM(np.zeros((12, 1)), func)
    # episodes = int(sys.argv[1])
    # for _ in range(10):
    #     theta = np.zeros((25, 4))
    #     sigma = 0.5
    #
    #     popSize = 10
    #     numElite = 5
    #     evaluationFunction = CandidateGridworldPDIS
    #     cem = CEM(theta, sigma, popSize, numElite, episodes, evaluationFunction)
    #
    #     for it in range(1000):
    #         x_min = cem.train()
    #         if it % 100 == 0:
    #
    #             print("episodes", episodes)
    #     # print("x_min:=", x_min)
    #             print("f_min:=", CandidateGridworldPDIS(x_min, episodes, multiplier=1))
    #             sys.stdout.flush()
    #     print("--------------------------")
    #     print("x_min:=", x_min)
    #     print("f_min:=", CandidateGridworldPDIS(x_min, episodes, multiplier=1))
    #     sys.stdout.flush()


def func(x, e):
    x = x.reshape(-1, 1)
    f = 4 + np.dot(x.T, x)
    return np.squeeze(f)[()]


def test():
    cem = CEM(np.ones((12,1)), func)
    cem.run_optimizer()


if __name__ == "__main__":
    test()
