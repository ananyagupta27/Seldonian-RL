import numpy as np
from helper import *
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworld import Gridworld
from environments.cartpole import Cartpole
from environments.mountaincar import Mountaincar


class Dataset:

    def __init__(self, episodes, env):
        self.env = env
        self.episodes = episodes


    def generate_dataset(self, theta):
        choices = self.env.getNumActions()
        print(choices)
        N = self.episodes
        theta = theta.reshape(-1, choices)

        states = [[] for _ in range(N)]
        actions = [[] for _ in range(N)]
        rewards = [[] for _ in range(N)]
        pi_b = [[] for _ in range(N)]


        for i in range(N):
            self.env.reset()
            s = self.env.state
            while True:
                states[i].append(s)
                s_transformed = get_transformed_state(self.env, s, theta)
                pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
                pi_b[i].append(pi)
                a = get_action(pi)
                actions[i].append(a)
                _, r, ended = self.env.step(a)
                rewards[i].append(r)

                if ended:
                    break
                s_prime = self.env.state
                s = s_prime
        dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b':pi_b}
        return dataset


def test():
    env = Cartpole()
    data = Dataset(2, env)
    dataset = data.generate_dataset(np.zeros((256, 2)))
    print(dataset)


if __name__ == "__main__":
    test()
