# import numpy as np
# # Run BBO - CEM and CMAES on all environments
# import numpy as np
#
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
# from optimizers.cem import CEM
# from optimizers.cmaes import CMAES
# from optimizers.utils import getCountlist, fourierBasis
#
#
# def CandidateSel(pi, param=100, verbose=0, multiplier=1):
#     pi = np.exp(pi) / np.sum(np.exp(pi), axis=1).reshape(4, 1)
#     alpha = 0.1
#     env = Cartpole()
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
#             _, r, _ = env.step(a)
#             G += r
#             num *= pi[s, a]
#             den *= 0.5
#             if env.inTAS():
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
#     #     sample_mean = np.mean(sol)
#     #     std = np.std(sol, ddof=1)
#
#     #     bound = (std*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
#     #     ub = np.array(sample_mean) + bound
#     #     lb = np.array(sample_mean) - bound
#     #     sample_mean1 = np.mean(np.array(sol) + 1)
#     sample_mean2 = np.mean(-20 - np.array(sol))
#     #     std1 = np.std(np.array(sol) +1 , ddof=1)
#     std2 = np.std(-20 - np.array(sol), ddof=1)
#
#     #     g1 = sample_mean1 + (std1*tinv(1-delta, no_of_episodes-1))/np.sqrt(no_of_episodes)
#     g2 = sample_mean2 + (std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     if g2 >= 0:
#         ans += -10000
#     return ans