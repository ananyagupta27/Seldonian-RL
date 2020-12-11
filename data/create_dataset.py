from utils.helper import *
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworldv2 import Gridworldv2

"""
This class is responsible for data generation by either running the environment for given number of episodes 
or reading data from file and saving the episodes and timesteps of data including states, actions, rewards, behavior policy
"""


class Dataset:


    def __init__(self, episodes=None, env=None):
        self.env = env
        self.episodes = episodes

    def generate_dataset(self, theta):
        """
        :param theta: parameterized initial behavior policy to be run to get the data from the environment
        :return: dataset
        """
        choices = self.env.getNumActions()
        N = self.episodes
        theta = theta.reshape(-1, choices)
        states = [[] for _ in range(N)]
        actions = [[] for _ in range(N)]
        rewards = [[] for _ in range(N)]
        pi_b = [[] for _ in range(N)]
        ended = False

        for i in range(N):
            self.env.reset()
            s = self.env.state
            while True:
                states[i].append(s)

                s_transformed = get_transformed_state(self.env, s, theta)
                pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))


                a = get_action(pi)
                pi_b[i].append(pi[0][a])
                actions[i].append(a)


                # For mountain car take step with same action 4 times as described in the paper
                # https://arxiv.org/pdf/1511.03722.pdf
                if self.env.name is "Mountaincar":
                    r = 0
                    for _ in range(4):
                        _, r, ended = self.env.step(a)
                    rewards[i].append(4 * r)
                else:
                    _, r, ended = self.env.step(a)
                    rewards[i].append(r)

                if ended:
                    break
                s_prime = self.env.state
                s = s_prime
        dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b}

        return dataset


    def get_dataset_from_file(self, filename):
        """

        :param filename: name of the file from which to read data of the format

        episodes
        timesteps
        state, action, reward, pi_b
        state, action, reward, pi_b
        ....................

        :return: dataset
        """
        fileForInput = open(filename, 'r')
        no_of_episodes = int(fileForInput.readline())

        states = [[] for _ in range(no_of_episodes)]
        actions = [[] for _ in range(no_of_episodes)]
        rewards = [[] for _ in range(no_of_episodes)]
        pi_b = [[] for _ in range(no_of_episodes)]

        for episode in range(no_of_episodes):
            timesteps = int(fileForInput.readline())
            for timestep in range(timesteps):
                cur_list = fileForInput.readline().rstrip('\n').split(',')
                states[episode].append(int(cur_list[0]))
                actions[episode].append(int(cur_list[1]))
                rewards[episode].append(int(cur_list[2]))
                pi_b[episode].append(float(cur_list[3]))

        dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b}
        return dataset


def test():
    env = Gridworldv2()
    data = Dataset(20, env)
    dataset = data.generate_dataset(np.zeros((16, 4)))
    print(dataset)



if __name__ == "__main__":
    test()
