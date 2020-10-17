import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
import sys
import os
np.seterr(all='raise')

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworld import Gridworld
from environments.cartpole import Cartpole
from environments.mountaincar import Mountaincar

from optimizers.cem import CEM
from optimizers.powell import Powell
from optimizers.cmaes import CMAES
from helper import *
from create_dataset import Dataset

# dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b}


def IS(theta, dataset, episodes, env):
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    is_estimates = []
    average_estimate = 0
    for episode in range(episodes):

        G_h_l = 0
        is_current = 0
        num = 1
        den = 1

        for timestep in range(len(states[episode])):

            s = states[episode][timestep]
            a = actions[episode][timestep]
            r = rewards[episode][timestep]
            pi_b_cur = pi_b[episode][timestep]

            G_h_l += r

            s_transformed = get_transformed_state(env, s, theta)
            pi_e = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            num *= pi_e[0][a]
            den *= pi_b_cur[0][a]

        is_current = G_h_l * num / den
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)



def PDIS(theta, dataset, episodes, env):
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    is_estimates = []
    average_estimate = 0


    for episode in range(episodes):

        G_h_l = 0
        is_current = 0
        num = 1
        den = 1

        for timestep in range(len(states[episode])):
            s = states[episode][timestep]
            a = actions[episode][timestep]
            r = rewards[episode][timestep]
            pi_b_cur = pi_b[episode][timestep]

            G_h_l += r

            s_transformed = get_transformed_state(env, s, theta)
            pi_e = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            try:
                num *= pi_e[0][a]
                den *= pi_b_cur[0][a]


                is_current += (env.gamma**timestep) * (rewards[episode][timestep] * (num / den))
            except:
                break
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)



def WIS(theta, dataset, episodes, env):
    pass


def total_return(theta, dataset, episodes, env):

    states = dataset['states']
    rewards = dataset['rewards']

    is_estimates = []
    for episode in range(episodes):

        G_h_l = 0
        for timestep in range(len(states[episode])):
            r = rewards[episode][timestep]
            G_h_l += r


        is_current = G_h_l
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)


def test():
    env = Gridworld()
    datasetGenerator = Dataset(100, env)
    theta = np.zeros((env.getStateDims(), env.getNumActions()))
    dataset = datasetGenerator.generate_dataset(theta)
    avg, arr = PDIS(theta, dataset, 100, env)
    print(avg)


if __name__ == "__main__":
    test()
