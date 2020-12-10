import numpy as np
# from scipy.stats import t
import sys
import os

np.seterr(all='raise')

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from helper import *


# dataset = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b}


def IS(theta, dataset, episodes, env):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    theta = theta.reshape(env.getStateDims(), env.getNumActions())

    is_estimates = []
    average_estimate = 0
    for episode in range(episodes):

        G_h_l = 0
        is_current = 0

        frac = 1

        for timestep in range(len(states[episode])):
            s = states[episode][timestep]
            a = actions[episode][timestep]
            r = rewards[episode][timestep]
            pi_b_cur = pi_b[episode][timestep]

            G_h_l += r

            s_transformed = get_transformed_state(env, s, theta)
            pi_e = normalize(np.dot(s_transformed.T, theta))
            # pi_e = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))

            frac *= pi_e[0][a] / pi_b_cur

        is_current = G_h_l * frac


        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    # print(is_estimates)
    return average_estimate, np.array(is_estimates)

def PDIS(theta, dataset, episodes, env):
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    # print(theta)
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
        # frac= 1

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
                den *= pi_b_cur
                # frac *= pi_e[0][a] / pi_b_cur[0][a]

                is_current += (env.gamma**timestep) * (rewards[episode][timestep] * (num / den))
            except:
                break
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    print(average_estimate)
    sys.stdout.flush()
    #exit(0)
    return average_estimate, np.array(is_estimates)

# def PDIS(theta, dataset, episodes, env):
#     states = dataset['states']
#     actions = dataset['actions']
#     rewards = dataset['rewards']
#     pi_b = dataset['pi_b']
#
#     theta = theta.reshape(env.getStateDims(), env.getNumActions())
#
#
#     is_estimates = []
#     average_estimate = 0
#
#     for episode in range(episodes):
#
#         is_current = 0
#         frac = 1
#         try:
#             for timestep in range(len(states[episode])):
#                 s = states[episode][timestep]
#                 a = actions[episode][timestep]
#                 r = rewards[episode][timestep]
#                 pi_b_cur = pi_b[episode][timestep]
#
#                 s_transformed = get_transformed_state(env, s, theta)
#                 pi_e = normalize(np.dot(s_transformed.T, theta))
#
#                 frac *= pi_e[0][a] / pi_b_cur
#             is_current += (env.gamma ** timestep) * (rewards[episode][timestep] * frac)
#         except:
#             continue
#
#         is_estimates.append(is_current)
#     # print(np.mean(is_estimates))
#     average_estimate = np.mean(is_estimates)
#     return average_estimate, np.array(is_estimates)


def WIS(theta, dataset, episodes, env):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    theta = theta.reshape(env.getStateDims(), env.getNumActions())

    is_estimates = []
    average_estimate = 0
    norm = 0
    for episode in range(episodes):

        G_h_l = 0
        is_current = 0

        frac = 1
        for timestep in range(len(states[episode])):
            s = states[episode][timestep]
            a = actions[episode][timestep]
            r = rewards[episode][timestep]
            pi_b_cur = pi_b[episode][timestep]

            G_h_l += r

            s_transformed = get_transformed_state(env, s, theta)
            pi_e = normalize(np.dot(s_transformed.T, theta))
            frac *= pi_e[0][a] / pi_b_cur

        is_current = G_h_l * frac
        norm += frac

        is_estimates.append(is_current)
    is_estimates = is_estimates * episodes / norm
    average_estimate = np.mean(is_estimates)
    # print(np.mean(is_estimates))
    return average_estimate, np.array(is_estimates)


def DR(theta, dataset, episodes, env):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']
    p = dataset['p']
    R = dataset['R']

    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    max_exp = np.max(theta, axis=1).reshape(theta.shape[0], 1)
    pi_theta = np.exp(theta - max_exp) / np.sum(np.exp(theta - max_exp), axis=1).reshape(env.getStateDims(), 1)
    # print(pi_theta)
    Q, V = loadEvalPolicy(pi_theta, episodes, p, R, env)
    # print(Q.shape, V.shape)
    is_estimates = []
    average_estimate = 0

    # s_transformed = get_transformed_state(env, s, theta)
    # pi_e = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))

    L = 0
    for i in range(episodes):
        L = max(L, len(states[i]))

    rho = np.zeros((L, episodes))
    pi_e = [[] for _ in range(episodes)]

    for i in range(episodes):
        for timestep in range(len(states[i])):
            s_transformed = get_transformed_state(env, states[i][timestep], theta)
            pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            pi_e[i].append(pi[0][actions[i][timestep]])

    for i in range(episodes):
        rho[0][i] = pi_e[i][0] / pi_b[i][0]

    for timestep in range(1, L):
        for i in range(episodes):
            rho[timestep][i] = rho[timestep - 1][i] * (pi_e[i][timestep] / pi_b[i][timestep]) if timestep < len(
                states[i]) else rho[timestep - 1][i]

    gamma = env.gamma
    for i in range(episodes):
        is_current = 0
        curGamma = 1
        for timestep in range(min(L, len(states[i]))):
            is_current += curGamma * rho[timestep][i] * rewards[i][timestep]
            weightV = 1 if timestep == 0 else rho[timestep - 1][i]
            weightQ = rho[timestep - 1][i]
            is_current -= curGamma * (
                    weightQ * Q[0][env.getDiscreteState(states[i][timestep])][actions[i][timestep]] - weightV
                    * V[0][env.getDiscreteState(states[i][timestep])])
            curGamma *= gamma
        is_estimates.append(is_current)
    # print(is_estimates)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)


def DR_hat(theta, dataset, episodes, env):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']
    # p = dataset['p']
    # R = dataset['R']
    Q = dataset['Q']
    V = dataset['V']
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    max_exp = np.max(theta, axis=1).reshape(theta.shape[0], 1)
    pi_theta = np.exp(theta - max_exp) / np.sum(np.exp(theta - max_exp), axis=1).reshape(env.getStateDims(), 1)
    # print(pi_theta)
    # Q, V = loadEvalPolicy(pi_theta, episodes, p, R, env)
    # print(Q.shape, V.shape)
    is_estimates = []
    average_estimate = 0

    # s_transformed = get_transformed_state(env, s, theta)
    # pi_e = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))

    L = 0
    for i in range(episodes):
        L = max(L, len(states[i]))

    rho = np.zeros((L, episodes))
    pi_e = [[] for _ in range(episodes)]

    for i in range(episodes):
        for timestep in range(len(states[i])):
            s_transformed = get_transformed_state(env, states[i][timestep], theta)
            pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            pi_e[i].append(pi[0][actions[i][timestep]])

    for i in range(episodes):
        rho[0][i] = pi_e[i][0] / pi_b[i][0]

    for timestep in range(1, L):
        for i in range(episodes):
            rho[timestep][i] = rho[timestep - 1][i] * (pi_e[i][timestep] / pi_b[i][timestep]) if timestep < len(
                states[i]) else rho[timestep - 1][i]

    gamma = env.gamma
    for i in range(episodes):
        is_current = 0
        curGamma = 1
        for timestep in range(min(L, len(states[i]))):
            is_current += curGamma * rho[timestep][i] * rewards[i][timestep]
            weightV = 1 if timestep == 0 else rho[timestep - 1][i]
            weightQ = rho[timestep - 1][i]
            # is_current -= curGamma * (weightQ * Q[0][np.argmax(states[i][timestep])][actions[i][timestep]] - weightV
            #                           * V[0][np.argmax(states[i][timestep])])
            is_current -= curGamma * (
                    weightQ * Q[0][env.getDiscreteState(states[i][timestep])][actions[i][timestep]] - weightV
                    * V[0][env.getDiscreteState(states[i][timestep])])
            curGamma *= gamma
        is_estimates.append(is_current)
    # print(is_estimates)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)


def loadEvalPolicy(pi_e, episodes, p, R, env):
    numStates = env.getNumDiscreteStates()
    numActions = env.getNumActions()
    actionProbabilities = np.zeros((numStates, numActions))
    L = env.horizonLength

    numStates -= 1

    for s in range(numStates):
        actionProbabilities[s] = pi_e[s]

    Q = np.zeros((L, numStates + 1, numActions))
    V = np.zeros((L, numStates + 1))
    # print(L,"L",numStates,numActions)

    for t in range(L - 1, -1, -1):
        for s in range(numStates):
            for a in range(numActions):
                for sPrime in range(numStates + 1):
                    Q[t][s][a] += p[s][a][sPrime] * R[s][a][sPrime]
                    if sPrime != numStates and t != L - 1:
                        Q[t][s][a] += p[s][a][sPrime] * np.dot(actionProbabilities[sPrime], (Q[t + 1][sPrime]))

    for t in range(L):
        for s in range(numStates):
            V[t][s] = np.dot(actionProbabilities[s], (Q[t][s]))

    # for s in range(numStates):
    #     print("state:", s, " value func:", V[0][s])
    #
    # for s in range(numStates):
    #     for a in range(4):
    #         print("action q function", s, " ", a, " ", Q[0][s][a])
    return Q, V


def total_return(dataset, episodes):
    states = dataset['states']
    rewards = dataset['rewards']

    is_estimates = []
    for episode in range(episodes):

        G_h_l = 0
        for timestep in range(len(rewards[episode])):
            r = rewards[episode][timestep]
            G_h_l += r

        is_current = G_h_l
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    # print(is_estimates)
    return average_estimate, np.array(is_estimates)

# def test():
#     env = Mountaincar()
#     datasetGenerator = Dataset(100, env)
#     theta = np.zeros((env.getStateDims(), env.getNumActions()))
#     dataset = datasetGenerator.generate_dataset(theta)
#     avg, arr = total_return(theta, dataset, 100, env)
#     print(avg)
#     print(arr)
#
#
# def test2():
#     env = Gridworldv2()
#     data = Dataset(200, env)
#     dataset = data.generate_dataset(np.zeros((16, 4)))
#     print(dataset)
#     model = Model(dataset, 200, 16, 4, 100)
#     dataset = model.makeMLEModel()
#     loadEvalPolicy(np.ones((16,4))*0.25, 100, dataset['p'], dataset['R'], env)
#
#


# def test2():
#     env = Gridworldv2()
#     data = Dataset(20, env)
#     dataset = data.generate_dataset(np.zeros((16, 4)))
#
#
#     model = Model(dataset, 20, 16, 4, 100)
#     dataset = model.makeMLEModel()
#
#     isEstimatesObj = ISEstimates(dataset, 20, env)
#     isEstimatesObj.DR(np.zeros((16, 4)))
#
#
# if __name__ == "__main__":
#     test2()
