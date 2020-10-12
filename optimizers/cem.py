import numpy as np

from typing import Callable


from scipy.stats import t
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))
sys.path.insert(1, os.path.dirname(sys.path[0]))

from optimize_with_bbo import CandidateGridworld, CandidateGridworldPDIS
from optimize_cartpole import  CandidateCartpole
class CEM:
    """
    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta: np.ndarray, sigma: float, popSize: int, numElite: int, numEpisodes: int,
                 evaluationFunction: Callable, epsilon: float = 0.0001):
        self._name = "Cem"
        self._theta = theta
        self._Sigma = sigma * np.eye(self._theta.size)
        self._popSize = popSize
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._epsilon = epsilon
        self._placeholder_theta = theta
        self._placeholder_Sigma = self._Sigma
        self._evaluationFunction = evaluationFunction

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        return self._theta.flatten()

    def setNumEpisodes(self, numEpisodes):
        self._numEpisodes = numEpisodes

    def train(self) -> np.ndarray:

        values = []
        for k in range(1, self._popSize + 1, 1):
            theta_k = np.random.multivariate_normal(self._theta.flatten(), self._Sigma)
            #             print(theta_k)
            # J_dash = self._evaluationFunction(theta_k.reshape(25, 4), self._numEpisodes, multiplier=1)
            J_dash = self._evaluationFunction(theta_k)
            #             print(k, " ", J_dash)
            if J_dash > -9999:

                values.append((theta_k, J_dash))
        sorted_values = sorted(values, key=lambda second: second[1], reverse=True)
        if len(values) < 1:
            return self._theta
        theta_values = np.asarray([i[0] for i in sorted_values])[0: min(self._numElite, len(values))]
        J_values = np.asarray([i[1] for i in sorted_values])[0: min(self._numElite, len(values))]
        # if len(J_values) > 0:
        #     print(J_values[0])
        self._theta = np.sum(theta_values, axis=0) / min(self._numElite, len(values))

        new_J = np.sum(J_values, axis=0) / min(self._numElite, len(values))

        dot_theta = 0
        for i in range(min(self._numElite, len(values))):
            dot_theta += np.dot(np.reshape(theta_values[i] - self._theta, (-1, 1)),
                                np.reshape(np.transpose(theta_values[i] - self._theta), (1, -1)))

        self._Sigma = (self._epsilon * np.eye(self._Sigma.shape[0]) + dot_theta) / (
                    self._epsilon + min(self._numElite, len(values)))

        return theta_values[0]

    def reset(self) -> None:
        self._theta = self._placeholder_theta
        self._Sigma = self._placeholder_Sigma


def func(x):
    x = x.reshape(-1,1)
    f = 4 - np.dot(x.T,x)
    # print(f)
    # exit()
    return np.squeeze(f)[()]


def main():
    # cmaes = CEM(np.zeros((12, 1)), func)
    episodes = int(sys.argv[1])
    for _ in range(10):
        theta = np.zeros((25, 4))
        sigma = 0.5

        popSize = 10
        numElite = 5
        evaluationFunction = CandidateGridworldPDIS
        cem = CEM(theta, sigma, popSize, numElite, episodes, evaluationFunction)

        for it in range(1000):
            x_min = cem.train()
            if it % 100 == 0:
                
                print("episodes", episodes)
        # print("x_min:=", x_min)
                print("f_min:=", CandidateGridworldPDIS(x_min, episodes, multiplier=1))
                sys.stdout.flush()
        print("--------------------------")
        print("x_min:=", x_min)
        print("f_min:=", CandidateGridworldPDIS(x_min, episodes, multiplier=1))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
