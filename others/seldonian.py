# import numpy as np
# # Run BBO - CEM and CMAES on all environments
# import numpy as np
# from scipy.optimize import minimize
# from helper import *
# import sys
# import os
#
# sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
# sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))
#
# from environments.gridworld import Gridworld
# from environments.cartpole import Cartpole
# from environments.mountaincar import Mountaincar
#
# # from optimizers.cem import CEM
# from optimizers.cmaes import CMAES
# from optimizers.utils_dir import getCountlist, fourierBasis
#
# delta = 0.05
#
#
# class RandAgent:
#     def __init__(self, states, actions, gamma, alpha):
#         self.states = states
#         self.actions = actions
#         self.gamma = gamma
#         self.alpha = alpha
#
#     #         self.pi = np.zeros([states, actions]) + 0.25
#     def get_action(self, state):
#         return np.random.choice(4, 1)[0]
#
#
#
#
# def CandidateCartpolePDIS(theta, dataset, param=10, verbose=0, multiplier=2):
#     theta = theta.reshape(256, 2)
#     env = Cartpole()
#     total = 0
#     delta = 0.05
#     no_of_episodes = param
#     sol = []
#     L = 1000
#     G = 0
#     for episode in range(0, no_of_episodes):
#         temp_sum = 0
#         r = 0
#         for horizon_iterator in range(0, len(dataset['states'][episode])):
#             num = 1
#             den = 1
#             for j in range(horizon_iterator):
#                 s = dataset['states'][episode][j]
#                 s_transformed = get_transformed_state(env, s, theta)
#                 pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
#                 a = dataset['actions'][episode][j]
#
#                 num = num * np.squeeze(pi[0][a])
#                 den = den * 0.5
#
#             r_t = dataset['rewards'][episode][horizon_iterator]
#             temp_sum += (env.gamma ** horizon_iterator) * r_t * num / den
#         G += temp_sum
#         sol.append(temp_sum)
#
#     G /= no_of_episodes
#     sample_mean1 = np.mean(np.array(sol) + 1)
#     # sample_mean2 = np.mean(15 - np.array(sol))
#     # # std1 = np.std(np.array(sol) + 1, ddof=1)
#     # std2 = np.std(15 - np.array(sol), ddof=1)
#     #
#     # # g1 = sample_mean1 + (multiplier * std1 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     # g2 = sample_mean2 + (multiplier * std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     # if g2 >= 0:
#     #     G += -100000
#     print("G=",G, sol)
#     return G
#
#
#
# def getCandidateSolution(dataset, param=100):
#     # Chooses the black-box optimizer we will use (Powell)
#     minimizer_method = 'Powell'
#     minimizer_options = {'disp': False}
#
#     theta = np.zeros((256, 2))
#     sigma = 0.5
#
#     popSize = 10
#     numElite = 5
#     numEpisodes = param
#     evaluationFunction = CandidateCartpolePDIS
#     multiplier = 2
#     # Initial solution given to Powell: simple linear fit we'd get from ordinary least squares linear regression
#     # initialSolution = leastSq(candidateData_X, candidateData_Y)
#
#     # Use Powell to get a candidate solution that tries to maximize candidateObjective
#     res = minimize(evaluationFunction, x0=theta, method=minimizer_method, options=minimizer_options,
#                    args=(dataset, numEpisodes, multiplier), tol=0.01)
#
#     print(CandidateCartpolePDIS(res.x, dataset,numEpisodes, multiplier=1))
#     # Return the candidate solution we believe will pass the safety test
#     return res.x
#
#
# def generate_dataset(env, choices, N):
#     states = [[] for i in range(N)]
#     actions = [[] for i in range(N)]
#     rewards = [[] for i in range(N)]
#     for i in range(N):
#         env.reset()
#         s = env.state
#         while True:
#             states[i].append(s)
#             a = np.random.choice(choices, 1)
#             actions[i].append(a)
#             _, r, ended = env.step(a)
#             rewards[i].append(r)
#
#             if ended:
#                 break
#             s_prime = env.state
#             s = s_prime
#     dataset = {'states': states, 'actions': actions, 'rewards': rewards}
#     return dataset
#
#
# def main():
#     episodes = int(sys.argv[1])
#
#     for _ in range(10):
#         env = Cartpole()
#         theta = np.zeros((256* 2))
#         evaluationFunction = CandidateCartpolePDIS
#         dataset_candidate = generate_dataset(env, 2, episodes)
#         dataset_safety = generate_dataset(env, 2, episodes)
#         print("done dataset creation")
#         cmaes = CMAES(theta, evaluationFunction)
#         # print(evaluationFunction(cmaes.generation_loop()))
#         x_min = cmaes.run_optimizer(episodes, dataset_candidate)
#         # x_min = getCandidateSolution(dataset_candidate,  episodes)
#         print("episodes", episodes)
#         print("x_min:=", x_min)
#         print("f_min:=", CandidateCartpolePDIS(x_min, dataset_safety,episodes, multiplier=1))
#         sys.stdout.flush()
#
#
# if __name__ == "__main__":
#     main()
