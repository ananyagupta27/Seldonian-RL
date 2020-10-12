import numpy as np
# Run BBO - CEM and CMAES on all environments
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworld import Gridworld
from environments.cartpole import Cartpole
from environments.mountaincar import Mountaincar

# from optimizers.cem import CEM
# from optimizers.cmaes import CMAES
from optimizers.utils import getCountlist, fourierBasis

delta = 0.05


class RandAgent:
    def __init__(self, states, actions, gamma, alpha):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha

    #         self.pi = np.zeros([states, actions]) + 0.25
    def get_action(self, state):
        return np.random.choice(4, 1)[0]


def tinv(p, nu):
    return t.ppf(p, nu)


def get_transformed_state(env, state, theta, order=3):
    for i, item in enumerate(env.observation_space.low):
        state[i] = (state[i] - env.observation_space.low[i]) / (
                env.observation_space.high[i] - env.observation_space.low[i])

    count_list = getCountlist(state.shape[0], order)
    phi = fourierBasis(state, count_list)
    return phi


def get_action(actPr):
    actions = actPr.shape[1]
    temp = np.random.uniform(0, 1)
    sum_pr = 0
    # print(actPr, actions)

    for i in range(actions):
        sum_pr += actPr[0][i]
        if temp <= sum_pr:
            return i
    return actions - 1


def CandidateCartpole(theta, param=10, verbose=0, multiplier=1):
    theta = theta.reshape(256, 2)
    env = Cartpole()
    total = 0
    no_of_episodes = param
    sol = []
    for episode in range(no_of_episodes):
        env.reset()
        G = 0
        s = env.state

        while True:

            s_transformed = get_transformed_state(env, s, theta)
            pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            a = get_action(pi)
            ss, r, isended = env.step(a)
            if env.isEnd:
                break
            G += r

            s_prime = env.state
            s = s_prime

        sol.append(G)
        total += G
        if G > 1000:
            print(env.terminal())
            print(env.t)
    ans = total / no_of_episodes
    print("cartpole", ans, sol)
    return ans


def CandidateMountaincarPDIS(theta, param=10, verbose=0, multiplier=1):
    theta = theta.reshape(16, 3)
    env = Mountaincar()
    total = 0
    no_of_episodes = param
    sol = []
    for episode in range(no_of_episodes):
        env.reset()
        G = 0
        s = env.state
        while True:

            s_transformed = get_transformed_state(env, s, theta)
            pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            a = get_action(pi)
            for _ in range(5):
                ss, r, isended = env.step(a)
            if env.isEnd:
                break
            G += r * 5

            s_prime = env.state
            s = s_prime

        sol.append(G)
        total += G
    ans = total / no_of_episodes
    print("mountaincar", ans, sol)
    return ans


def CandidateCartpolePDIS(theta, param=10, verbose=0, multiplier=2):
    theta = theta.reshape(256, 2)
    env = Cartpole()
    total = 0
    delta = 0.05
    no_of_episodes = param
    sol = []
    L = 1000
    G = 0
    for episode in range(1, no_of_episodes):
        temp_sum = 0
        r = 0
        for horizon_iterator in range(1, L):
            env.reset()
            s = env.state
            num = 1
            den = 1
            for j in range(horizon_iterator):

                s_transformed = get_transformed_state(env, s, theta)
                pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
                a = np.random.choice(2,1)
                num = num * pi[0][a]
                den = den * 0.5
                ss, r, isended = env.step(a)
                if env.isEnd:
                    r = 0
                    break

                s_prime = env.state
                s = s_prime
            if env.isEnd:
                break

            temp_sum += (env.gamma ** (horizon_iterator - 1)) * r * num / den
        G += temp_sum
        sol.append(temp_sum)

    G /= no_of_episodes
    # sample_mean1 = np.mean(np.array(sol) + 1)
    sample_mean2 = np.mean(20 - np.array(sol))
    # std1 = np.std(np.array(sol) + 1, ddof=1)
    std2 = np.std(20 - np.array(sol), ddof=1)

    # g1 = sample_mean1 + (multiplier * std1 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
    g2 = sample_mean2 + (multiplier * std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
    if g2 >= 0:
        G += -100000
    # print("G=",G, sol)
    return G


def CandidateMountaincar(theta, param=10, verbose=0, multiplier=1):
    theta = theta.reshape(16, 3)
    env = Mountaincar()
    total = 0
    no_of_episodes = param
    sol = []
    for episode in range(no_of_episodes):
        env.reset()
        G = 0
        s = env.state
        while True:

            s_transformed = get_transformed_state(env, s, theta)
            pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            a = get_action(pi)
            for _ in range(5):
                ss, r, isended = env.step(a)
            if env.isEnd:
                break
            G += r * 5

            s_prime = env.state
            s = s_prime

        sol.append(G)
        total += G
    ans = total / no_of_episodes
    print("mountaincar", ans, sol)
    return ans


'''
def CandidateCartpole(theta, param=10, verbose=0, multiplier=1):
    theta = theta.reshape(256, 2)
    delta = 0.05
    env = Cartpole()

    episode = 0
    total = 0
    no_of_episodes = param
    sol = []
    G = 0
    for i in range(1, no_of_episodes):
        temp_sum = 0
        r = 0

        for t_count in range(1, 1000):
            # loop for L
            env.reset()
            s = env.state
            num = 1
            den = 1
            for j in range(t_count):
                s_transformed = get_transformed_state(env, s, theta)
                pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
                a = np.random.choice(2)
                # a = get_action(pi)
                ss, r, isended = env.step(a)
                num *= pi[0][a]
                den *= 0.5

                if env.isEnd:
                    break

                s_prime = env.state
                s = s_prime
            temp_sum += (env.gamma ** (t_count-1)) * r * num / den

        G += temp_sum
        sol.append(temp_sum)

    G /= no_of_episodes
    # sample_mean1 = np.mean(np.array(sol) + 1)
    sample_mean2 = np.mean(20 - np.array(sol))
    # std1 = np.std(np.array(sol) + 1, ddof=1)
    std2 = np.std(20 - np.array(sol), ddof=1)

    # g1 = sample_mean1 + (multiplier * std1 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
    g2 = sample_mean2 + (multiplier * std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
    if g2 >= 0:
        G += -100000

    # print(G,"g2",g2, sol)
    return G
'''


def getCandidateSolution():
    # Chooses the black-box optimizer we will use (Powell)
    minimizer_method = 'Powell'
    minimizer_options = {'disp': False}

    theta = np.zeros((256, 2))
    sigma = 0.5

    popSize = 10
    numElite = 5
    numEpisodes = 2
    evaluationFunction = CandidateCartpole
    multiplier = 2
    # Initial solution given to Powell: simple linear fit we'd get from ordinary least squares linear regression
    # initialSolution = leastSq(candidateData_X, candidateData_Y)

    # Use Powell to get a candidate solution that tries to maximize candidateObjective
    res = minimize(evaluationFunction, x0=theta, method=minimizer_method, options=minimizer_options,
                   args=(numEpisodes, multiplier))

    print(CandidateCartpole(res.x, numEpisodes, multiplier=1))
    # Return the candidate solution we believe will pass the safety test
    return res.x


def main():
    getCandidateSolution()


if __name__ == "__main__":
    main()
