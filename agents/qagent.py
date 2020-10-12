import numpy as np

import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))

from gridworld import Gridworld


class QAgent:

    def __init__(self, states, actions, gamma, alpha):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.zeros([states, actions])
        self.pi = np.zeros([states, actions])

    def train(self, state, action, reward, s_prime):
        delta = reward + self.gamma * self.maxQ(s_prime) - self.q[state][action]
        self.q[state][action] += self.alpha * delta

    def train_inf(self, state, action, reward):
        delta = reward - self.q[state][action]
        self.q[state][action] += self.alpha * delta

    def new_episode(self):
        pass

    def get_action(self, state):
        act = self.q[state]
        actPr = np.exp(act) / np.sum(np.exp(act))
        temp = np.random.uniform(0, 1)
        sum_pr = 0
        for i in range(self.actions):
            sum_pr += actPr[i]
            if temp <= sum_pr:
                return i
        return self.actions - 1

    def get_pi(self):
        for state in range(self.states):
            act = self.q[state]
            self.pi[state] = np.exp(act) / np.sum(np.exp(act))
        return self.pi

    def maxQ(self, state):
        return np.max(self.q[state])


def get_pi_eval():
    alpha = 0.1
    env = Gridworld()
    agent = QAgent(25, 4, 1, alpha)
    episode = 0
    while True:
        env.reset()
        agent.new_episode()
        G = 0
        s = env.get_state()

        t = 0
        while True:
            a = agent.get_action(s)
            _, r, _ = env.step(a)
            G += r

            if env.terminal():
                agent.train_inf(s, a, r)
                break

            s_prime = env.get_state()
            agent.train(s, a, r, s_prime)
            s = s_prime

            t += 1
        episode += 1
        print("episode=", episode, " G", G)
        if episode > 800:
            break
    return agent.get_pi()


def main():
    get_pi_eval()


if __name__ == "__main__":
    main()
