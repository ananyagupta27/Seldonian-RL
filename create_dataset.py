import numpy as np
from helper import *
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworld import Gridworld
from environments.gridworldv2 import Gridworldv2
from environments.cartpole import Cartpole
from environments.mountaincar import Mountaincar


class Dataset:

    def __init__(self, episodes, env):
        self.env = env
        self.episodes = episodes

    def generate_dataset(self, theta):
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

                if type(self.env) is Mountaincar:
                    r = 0
                    for _ in range(5):
                        _, r, ended = self.env.step(a)
                    rewards[i].append(5 * r)
                else:
                    _, r, ended = self.env.step(a)
                    rewards[i].append(r)

                if ended:
                    break
                s_prime = self.env.state
                s = s_prime
        dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b}

        return dataset



class Model:

    def __init__(self, env, dataset, episodes, numStates, numActions, L):
        self.env = env
        self.dataset = dataset
        self.numStates = numStates - 1
        self.numActions = numActions
        self.L = L
        self.episodes = episodes

    def makeMLEModel(self):
        states = self.dataset['states']
        actions = self.dataset['actions']
        rewards = self.dataset['rewards']
        pi_b = self.dataset['pi_b']

        # Set everything to zero
        stateActionCounts = np.zeros((self.numStates, self.numActions))
        stateActionStateCounts = np.zeros((self.numStates, self.numActions, self.numStates + 1))

        p = np.zeros((self.numStates, self.numActions, self.numStates + 1))
        R = np.zeros((self.numStates, self.numActions, self.numStates + 1))
        d0 = np.zeros(self.numStates)

        # Compute all of the counts and set R to the sum of rewards from [s][a][sprime] transitions
        for i in range(self.episodes):
            trajectoryLength = len(states[i])
            for j in range(trajectoryLength):
                # s = np.argmax(states[i][j])
                s = self.env.getDiscreteState(states[i][j])
                a = actions[i][j]
                r = rewards[i][j]
                if j == trajectoryLength - 1:
                    sPrime = self.numStates
                    # if sPrime ==0 :
                    #     print("s=",s,"a=",a,"sp",sPrime)
                else:
                    # sPrime = np.argmax(states[i][j + 1])
                    sPrime = self.env.getDiscreteState(states[i][j + 1])
                # print("state is", s, "action is", a, "sprime is", sPrime)

                if j != self.L - 1:
                    stateActionStateCounts[s][a][sPrime] += 1
                    stateActionCounts[s][a] += 1
                else:
                    if sPrime != self.numStates:
                        print("sPrime not equal to numStates: Error ", sPrime)
                        exit()

                R[s][a][sPrime] += r

        for i in range(self.episodes):
            # d0[np.argmax(states[i][0])] += 1.0 / self.episodes
            d0[self.env.getDiscreteState(states[i][0])] += 1.0 / self.episodes

        rMin = rewards[0][0]
        for i in range(self.episodes):
            trajectoryLength = len(states[i])
            for j in range(trajectoryLength):
                rMin = min(rewards[i][j], rMin)

        # Compute P and R
        print("p and R")
        for s in range(self.numStates):
            for a in range(self.numActions):
                for sPrime in range(self.numStates + 1):
                    if stateActionCounts[s][a] == 0:
                        p[s][a][sPrime] = 1 if sPrime == self.numStates else 0
                    else:
                        p[s][a][sPrime] = stateActionStateCounts[s][a][sPrime] / stateActionCounts[s][a]

                    if stateActionStateCounts[s][a][sPrime] == 0:
                        R[s][a][sPrime] = 0
                    else:
                        R[s][a][sPrime] /= stateActionStateCounts[s][a][sPrime]
                    # print("s ", s, " a ", a, " sprime ", sPrime, "P", p[s][a][sPrime], " R ", R[s][a][sPrime])
        trajectories = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b, 'p': p, 'R': R,
                        'd0': d0}
        print("done modeling")
        return trajectories


def test():
    env = Gridworldv2()
    data = Dataset(20, env)
    dataset = data.generate_dataset(np.zeros((16, 4)))
    print(dataset)
    model = Model(dataset, 20, 16, 4, 100)
    model.makeMLEModel()



if __name__ == "__main__":
    test()
