import numpy as np
from helper import *

# dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b}


def IS(theta, dataset, episodes, env):
    theta = theta.reshape(256, 2)
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
    theta = theta.reshape(256, 2)
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

            is_current += (env.gamma**timestep) * rewards[episode][timestep] * num / den

        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)



def WIS(theta, dataset, episodes=10):
    pass



def test():
    pass


if __name__ == "__main__":
    test()
