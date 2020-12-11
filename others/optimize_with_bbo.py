# import numpy as np
# # Run BBO - CEM and CMAES on all environments
# import numpy as np
# from scipy.optimize import minimize
# from scipy.stats import t
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
# # from optimizers.cmaes import CMAES
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
# def tinv(p, nu):
#     return t.ppf(p, nu)
#
#
# # def get_sample_stats(list_of_n):
# #     sample_means = []
# #     sample_std   = []
# #     for n in list_of_n:
# #         X = sample_from_beta_distribution(n, alpha, beta)
# #         sample_means.append(np.mean(X))
# #         sample_std.append(get_stddev(X))
# #     return sample_means, sample_std
#
# # sample_means, sample_std = get_sample_stats(N)
# # bounds = [(sample_std[i]*tinv(1-delta, n-1))/np.sqrt(n) for i,n in enumerate(N)]
# # upper_bounds = np.array(sample_means) + np.array(bounds)
# # lower_bounds = np.array(sample_means) - np.array(bounds)
# def CandidateCartpole(pi, param=10, verbose=0, multiplier=1):
#     #     pi = np.exp(pi)/np.sum(np.exp(pi), axis=1).reshape(4,1)
#     alpha = 0.1
#     delta = 0.05
#     env = Cartpole()
#     #     qagent = QAgent(env.get_num_states(), env.get_num_actions(), 0.9, alpha)
#
#     episode = 0
#     total = 0
#     no_of_episodes = param
#     sol = []
#     while True:
#         env.reset()
#         G = 0
#         s = env.state
#
#         t_count = 0
#         num = 1
#         den = 1
#         while True:
#             a = np.random.choice(2)
#             ss, r, isended = env.step(a)
#             G += r
#             actions = np.exp(np.dot(s, pi)) / np.sum(np.exp(np.dot(s, pi)))
#             #             print(actions)
#             num *= actions[a]
#             #             num *= pi[s,a]
#             den *= 0.5
#             if env.isEnd:
#                 break
#
#             s_prime = env.state
#             s = s_prime
#
#             t_count += 1
#             if t_count > 100:
#                 break
#         episode += 1
#         #         print("episode=", episode, " G", G*num/den)
#         sol.append(G * num / den)
#         total += G * num / den
#         #         print(total)
#         #         if G == -8:
#         #             print(qagent.get_pi())
#         #             break
#         if episode > no_of_episodes:
#             break
#     #     print("total=",total)
#     #     print("Average = ", total/no_of_episodes)
#     #     if verbose:
#     #         print(pi)
#     ans = total / no_of_episodes
#     # #     sample_mean = np.mean(sol)
#     # #     std = np.std(sol, ddof=1)
#     #
#     # #     bound = (std*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
#     # #     ub = np.array(sample_mean) + bound
#     # #     lb = np.array(sample_mean) - bound
#     # #     sample_mean1 = np.mean(np.array(sol) - 120)
#     # sample_mean2 = np.mean(25 - np.array(sol))
#     # #     std1 = np.std(np.array(sol) -120 , ddof=1)
#     # std2 = np.std(25 - np.array(sol), ddof=1)
#     #
#     # #     g1 = sample_mean1 + (std1*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
#     # g2 = sample_mean2 + (std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     # if g2 >= 0:
#     #     ans += -10000
#     # #     print(ans)
#     return ans
#
#
# def CandidateGridworld(pi, param=100, verbose=0, multiplier=1):
#     pi = pi.reshape(25, 4)
#     pi = np.exp(pi) / np.sum(np.exp(pi), axis=1).reshape(25, 1)
#     alpha = 0.1
#     env = Gridworld()
#     #     qagent = QAgent(env.get_num_states(), env.get_num_actions(), 0.9, alpha)
#     ragent = RandAgent(4, 2, 1, alpha)
#     episode = 0
#     total = 0
#     no_of_episodes = param
#     sol = []
#     while True:
#         env.reset()
#         G = 0
#         s = env.state
#
#         t = 0
#         num = 1
#         den = 1
#         while True:
#             a = ragent.get_action(s)
#             # a = np.argmax(pi[s])
#             _, r, _ = env.step(a)
#             G += r
#             num *= pi[s, a]
#             den *= 0.5
#             if env.terminal():
#                 break
#
#             s_prime = env.get_state()
#             s = s_prime
#
#             t += 1
#             if t > 100:
#                 break
#         episode += 1
#         #         print("episode=", episode, " G", G*num/den)
#         sol.append(G * num / den)
#         total += G * num / den
#         # total += G
#         #         if G == -8:
#         #             print(qagent.get_pi())
#         #             break
#         if episode > no_of_episodes:
#             break
#     #     print("total=",total)
#     #     print("Average = ", total/no_of_episodes)
#     #     if verbose:
#     #         print(pi)
#     ans = total / no_of_episodes
#     # #     sample_mean = np.mean(sol)
#     # #     std = np.std(sol, ddof=1)
#     #
#     # #     bound = (std*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
#     # #     ub = np.array(sample_mean) + bound
#     # #     lb = np.array(sample_mean) - bound
#     # #     sample_mean1 = np.mean(np.array(sol) + 1)
#     # sample_mean2 = np.mean(-20 - np.array(sol))
#     # #     std1 = np.std(np.array(sol) +1 , ddof=1)
#     # std2 = np.std(-20 - np.array(sol), ddof=1)
#     #
#     # #     g1 = sample_mean1 + (std1*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
#     # g2 = sample_mean2 + (std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     # if g2 >= 0:
#     #     ans += -10000
#     return ans
#
#
# def CandidateGridworldPDIS(pi, param=10, multiplier=1):
#     delta = 0.05
#     pi = pi.reshape(25, 4)
#     pi = np.exp(pi) / np.sum(np.exp(pi), axis=1).reshape(25, 1)
#     alpha = 0.1
#     env = Gridworld()
#     #     qagent = QAgent(env.get_num_states(), env.get_num_actions(), 0.9, alpha)
#     ragent = RandAgent(4, 4, 1, alpha)
#     episode = 0
#     total = 0
#     no_of_episodes = param
#     sol = []
#
#     L = 50
#     pho = 0
#
#     for i in range(1, no_of_episodes):
#         temp_sum = 0
#         norm = 0
#         r = 0
#         for horizon_iterator in range(1, L):
#             env.reset()
#             s = env.get_state()
#             num = 1
#             den = 1
#             for j in range(horizon_iterator):
#                 a = ragent.get_action(s)
#                 _, r, _ = env.step(a)
#                 num *= pi[s, a]
#                 den *= 0.25
#                 s_prime = env.get_state()
#                 s = s_prime
#
#             # print(temp_sum, "tempsum", "r=", r, "num", num, "den", den)
#             temp_sum += (env.gamma ** (horizon_iterator - 1)) * r * num / den
#
#         pho += temp_sum
#         sol.append(temp_sum)
#
#     pho /= no_of_episodes
#     sample_mean1 = np.mean(np.array(sol) + 1)
#     sample_mean2 = np.mean(-45 - np.array(sol))
#     std1 = np.std(np.array(sol) + 1, ddof=1)
#     std2 = np.std(-45 - np.array(sol), ddof=1)
#
#     g1 = sample_mean1 + (multiplier * std1 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     g2 = sample_mean2 + (multiplier * std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     if g2 >= 0 or g1 >= 0:
#         pho += -100000
#
#     # print(g1, "g1", g2, "g2", pho,"pho",sol)
#     # print(pho)
#     return pho
#
#
# def getCandidateSolution(param=100):
#     # Chooses the black-box optimizer we will use (Powell)
#     minimizer_method = 'Powell'
#     minimizer_options = {'disp': False}
#
#     theta = np.zeros((25, 4))
#     sigma = 0.5
#
#     popSize = 10
#     numElite = 5
#     numEpisodes = param
#     evaluationFunction = CandidateGridworldPDIS
#     multiplier = 2
#     # Initial solution given to Powell: simple linear fit we'd get from ordinary least squares linear regression
#     # initialSolution = leastSq(candidateData_X, candidateData_Y)
#
#     # Use Powell to get a candidate solution that tries to maximize candidateObjective
#     res = minimize(evaluationFunction, x0=theta, method=minimizer_method, options=minimizer_options,
#                    args=(numEpisodes, multiplier), tol=0.01)
#
#     print(CandidateGridworldPDIS(res.x, numEpisodes, multiplier=1))
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
#     for _ in range(10):
#         x_min = getCandidateSolution(episodes)
#         print("episodes", episodes)
#         # print("x_min:=", x_min)
#         print("f_min:=", CandidateGridworldPDIS(x_min, episodes, multiplier=1))
#         sys.stdout.flush()
#
# if __name__ == "__main__":
#     main()