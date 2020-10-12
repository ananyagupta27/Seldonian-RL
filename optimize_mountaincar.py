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


def get_transformed_state(state, theta, order=3):
    count_list = getCountlist(state.shape[0], order)
    phi=fourierBasis(state, count_list)
    return phi


def get_action(actPr):
    actions = actPr.shape[1]
    temp = np.random.uniform(0,1)
    sum_pr = 0
    # print(actPr, actions)

    for i in range(actions):
        sum_pr += actPr[0][i]
        if temp <= sum_pr:
            return i
    return actions - 1


def CandidateCartpole(theta, param=10, verbose=0, multiplier=1):
    #     pi = np.exp(pi)/np.sum(np.exp(pi), axis=1).reshape(4,1)
    theta = theta.reshape(256, 2)
    alpha = 0.1
    delta = 0.05
    env = Cartpole()
    #     qagent = QAgent(env.get_num_states(), env.get_num_actions(), 0.9, alpha)
    # print("entered: theta shape", theta.shape)
    episode = 0
    total = 0
    no_of_episodes = param
    sol = []
    while True:
        env.reset()
        G = 0
        s = env.state

        t = 0
        num = 1
        den = 1
        while True:

            s_transformed = get_transformed_state(s, theta)
            # print(s_transformed.shape, "stransformed shape", theta.shape, "thetashape")
            pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
            #             print(actions)
            # num *= pi[a]
            # #             num *= pi[s,a]
            # den *= 0.5

            # a = np.random.choice(2)
            a = get_action(pi)

            for _ in range(2):
                ss, r, isended = env.step(a)
                G += r

            if env.isEnd:
                break

            s_prime = env.state
            s = s_prime

            t += 1
        episode += 1
        #         print("episode=", episode, " G", G*num/den)
        sol.append(G * num / den)
        # total += G * num / den
        total += G
        #         print(total)
        #         if G == -8:
        #             print(qagent.get_pi())
        #             break
        if episode > no_of_episodes:
            break
    #     print("total=",total)
    #     print("Average = ", total/no_of_episodes)
    #     if verbose:
    #         print(pi)
    ans = total / no_of_episodes
    print(ans)
    # #     sample_mean = np.mean(sol)
    # #     std = np.std(sol, ddof=1)
    #
    # #     bound = (std*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
    # #     ub = np.array(sample_mean) + bound
    # #     lb = np.array(sample_mean) - bound
    # #     sample_mean1 = np.mean(np.array(sol) - 120)
    # sample_mean2 = np.mean(25 - np.array(sol))
    # #     std1 = np.std(np.array(sol) -120 , ddof=1)
    # std2 = np.std(25 - np.array(sol), ddof=1)
    #
    # #     g1 = sample_mean1 + (std1*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
    # g2 = sample_mean2 + (std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
    # if g2 >= 0:
    #     ans += -10000
    # #     print(ans)
    return -ans


def getCandidateSolution():
    # Chooses the black-box optimizer we will use (Powell)
    minimizer_method = 'Powell'
    minimizer_options = {'disp': False}

    theta = np.zeros((256, 2))
    sigma = 0.5

    popSize = 10
    numElite = 5
    numEpisodes = 100
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