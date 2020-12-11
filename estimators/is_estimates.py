# from scipy.stats import t
from utils.helper import *

np.seterr(all='raise')

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))



"""
This file contains various importance sampling estimators described in depth in Phil's thesis
https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
"""


def IS(theta, dataset, episodes, env):
    """

    :param theta: current evaluation policy parameter
    :param dataset: lists of states, actions, rewards, behavior policy
    :param episodes: number of episodes of data
    :param env: environment running
    :return: importance sampling estimate
    """
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    theta = theta.reshape(env.getStateDims(), env.getNumActions())

    is_estimates = []
    for episode in range(episodes):

        G_h_l = 0
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
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)


def PDIS(theta, dataset, episodes, env):
    """

    :param theta: current evaluation policy parameter
    :param dataset: lists of states, actions, rewards, behavior policy
    :param episodes: number of episodes of data
    :param env: environment running
    :return: per decision importance sampling estimate
    """
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    is_estimates = []

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
                den *= pi_b_cur
                is_current += (env.gamma ** timestep) * (rewards[episode][timestep] * (num / den))
            except:
                break
        is_estimates.append(is_current)
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)


def WIS(theta, dataset, episodes, env):
    """

    :param theta: current evaluation policy parameter
    :param dataset: lists of states, actions, rewards, behavior policy
    :param episodes: number of episodes of data
    :param env: environment running
    :return: weighted importance sampling estimate
    """
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    theta = theta.reshape(env.getStateDims(), env.getNumActions())

    is_estimates = []
    norm = 0
    for episode in range(episodes):

        G_h_l = 0

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
    return average_estimate, np.array(is_estimates)


def DR(theta, dataset, episodes, env):
    """
    Nan Jiang's doubly robust estimator  https://arxiv.org/pdf/1511.03722.pdf
    Also described in Phil's complete paper https://people.cs.umass.edu/~pthomas/papers/Thomas2016.pdf - Equations are taken from here
    Note- This function considers the states to be discrete
    :param theta: current evaluation policy parameter
    :param dataset: lists of states, actions, rewards, behavior policy
    :param episodes: number of episodes of data
    :param env: environment running
    :return: doubly robust estimate which has much less variance

    """
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']
    p = dataset['p']
    R = dataset['R']


    # calculating the current parameterized evaluation policy
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    max_exp = np.max(theta, axis=1).reshape(theta.shape[0], 1)
    pi_theta = np.exp(theta - max_exp) / np.sum(np.exp(theta - max_exp), axis=1).reshape(env.getStateDims(), 1)


    # getting the value function estimates for the current policy considering p and R are calculated using the MLE modeling
    Q, V = loadEvalPolicy(pi_theta, episodes, p, R, env)
    is_estimates = []

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

    # calculating the importance weights
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
    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)


def DR_hat(theta, dataset, episodes, env):
    """

    Nan Jiang's doubly robust estimator  https://arxiv.org/pdf/1511.03722.pdf
    Also described in Phil's complete paper https://people.cs.umass.edu/~pthomas/papers/Thomas2016.pdf - Equations are taken from here

    Note- This function considers the states to be discrete

    DR hat does not calculate the Q, V value functions for the current evaluation policy but calculates these functions once
    using the data from the behavior policy and uses those values for all iterations (is therefore much faster,
    but not as good as above DR estimator)

    :param theta: current evaluation policy parameter
    :param dataset: lists of states, actions, rewards, behavior policy
    :param episodes: number of episodes of data
    :param env: environment running
    :return: doubly robust estimate which has much less variance
    """
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    Q = dataset['Q']
    V = dataset['V']
    theta = theta.reshape(env.getStateDims(), env.getNumActions())
    max_exp = np.max(theta, axis=1).reshape(theta.shape[0], 1)
    is_estimates = []

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

    average_estimate = np.mean(is_estimates)
    return average_estimate, np.array(is_estimates)



def loadEvalPolicy(pi_e, episodes, p, R, env):
    """
    Calculates the Q and V value functions for the current evaluation policy using transition and reward functions
    using dynamic programming

    :param pi_e:
    :param episodes:
    :param p:
    :param R:
    :param env:
    :return:
    """
    numStates = env.getNumDiscreteStates()
    numActions = env.getNumActions()
    actionProbabilities = np.zeros((numStates, numActions))
    L = env.horizonLength

    numStates -= 1

    for s in range(numStates):
        actionProbabilities[s] = pi_e[s]

    Q = np.zeros((L, numStates + 1, numActions))
    V = np.zeros((L, numStates + 1))

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
    #     for a in range(numActions):
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
    return average_estimate, np.array(is_estimates)
