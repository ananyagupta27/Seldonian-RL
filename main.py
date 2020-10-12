# def CandidateCartpolePDIS(theta, param=10, verbose=0, multiplier=2):
#     theta = theta.reshape(256, 2)
#     env = Cartpole()
#     total = 0
#     delta = 0.05
#     no_of_episodes = param
#     sol = []
#     L = 1000
#     G = 0
#     for episode in range(1, no_of_episodes):
#         temp_sum = 0
#         r = 0
#         for horizon_iterator in range(1, L):
#             env.reset()
#             s = env.state
#             num = 1
#             den = 1
#             for j in range(horizon_iterator):
#
#                 s_transformed = get_transformed_state(env, s, theta)
#                 pi = np.exp(np.dot(s_transformed.T, theta)) / np.sum(np.exp(np.dot(s_transformed.T, theta)))
#                 a = get_action(pi)
#                 num = num * pi[0][a]
#                 den = den * 0.5
#                 ss, r, isended = env.step(a)
#                 if env.isEnd:
#                     r = 0
#                     break
#
#                 s_prime = env.state
#                 s = s_prime
#             temp_sum += (env.gamma ** (horizon_iterator - 1)) * r * num / den
#         G += temp_sum
#         sol.append(temp_sum)
#
#     G /= no_of_episodes
#     # sample_mean1 = np.mean(np.array(sol) + 1)
#     sample_mean2 = np.mean(20 - np.array(sol))
#     # std1 = np.std(np.array(sol) + 1, ddof=1)
#     std2 = np.std(20 - np.array(sol), ddof=1)
#
#     # g1 = sample_mean1 + (multiplier * std1 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     g2 = sample_mean2 + (multiplier * std2 * tinv(1 - delta, no_of_episodes - 1)) / np.sqrt(no_of_episodes)
#     if g2 >= 0:
#         G += -100000
#
#     return G