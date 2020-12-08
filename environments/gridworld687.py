from abc import ABC

import numpy as np

from gym import spaces
import copy
from .environment import Environment


class Gridworld687(Environment):

    def __init__(self, size=5, gamma=0.9, horizonLength=200, discrete=True):
        self.size = int(size)
        self._gamma = gamma
        self._horizonLength = horizonLength
        self.numStates = size ** 2
        self.numActions = 4
        self.observation_space = spaces.Box(low=np.zeros(self.numActions), high=np.ones(self.numActions),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.numActions)
        self.x = 0
        self.y = 0
        self.count = 0
        self.discrete = discrete

    @property
    def isEnd(self):
        pass

    @property
    def state(self):
        return self.getState()

    @property
    def gamma(self):
        return self._gamma

    @property
    def horizonLength(self):
        return self._horizonLength

    @property
    def threshold(self):
        return -0.6

    def getNumActions(self):
        return self.numActions

    def getStateDims(self):
        return self.numStates

    def successful_action(self, a):

        if a == 0 and ((self.x == 2 and self.y == 3) or (self.x == 3 and self.y == 3)):
            return
        if a == 1 and ((self.x == 2 and self.y == 1) or (self.x == 3 and self.y == 1)):
            return
        if a == 2 and self.x == 4 and self.y == 2:
            return
        if a == 3 and self.x == 1 and self.y == 2:
            return
        if a == 0:
            self.y -= 1
        elif a == 1:
            self.y += 1
        elif a == 2:
            self.x -= 1
        elif a == 3:
            self.x += 1
        else:
            raise Exception("Action out of range! Must be in [0,3]: " + a)

        self.x = int(np.clip(self.x, 0, self.size - 1))
        self.y = int(np.clip(self.y, 0, self.size - 1))

    def veer(self, action, right):
        # right=0 => veer left else veer right
        # action attempted

        if action == 0:
            if right == 0:
                a = 3  # go down
            else:
                a = 2  # go up
        elif action == 1:
            if right == 0:
                a = 2
            else:
                a = 3
        elif action == 2:
            if right == 0:
                a = 0
            else:
                a = 1
        elif action == 3:
            if right == 0:
                a = 1
            else:
                a = 0
        return a

    def step(self, action):
        self.count += 1
        a = int(action)

        s = self.state
        prob = np.random.rand()

        if prob <= 0.8:  # 80% of time
            self.successful_action(a)
        elif 0.8 < prob <= 0.85:  # 5% of time
            # veer right
            new_a = self.veer(a, 1)  # changes depending on where robot is trying to go
            self.successful_action(new_a)
        elif 0.85 < prob <= 0.9:  # 5% of time
            # veer left!
            new_a = self.veer(a, 0)
            self.successful_action(new_a)

        reward = 0
        if self.x == self.size - 1 and self.y == self.size - 1:
            reward = 10
        elif self.x == 4 and self.y == 2:
            reward = -10

        return self.getState(), reward, self.terminal()

    def nextState(self, state, action):
        pass

    def reset(self):
        self.x = 0
        self.y = 0
        self.count = 0
        return self.getState()

    def R(self, state, action, nextState):
        pass

    def getState(self):
        x = np.zeros(self.numStates, dtype=np.float32)
        x[self.x * self.size + self.y] = 1
        return x

    def getDiscreteState(self, state):
        return np.argmax(state)

    def getNumDiscreteStates(self):
        return self.getStateDims()

    def terminal(self):
        return (self.x == self.size - 1 and self.y == self.size - 1) or (self.count >= self.horizonLength - 1)


def test():
    env = Gridworld687()
    avr = 0
    for i in range(100):
        G = 0
        env.reset()
        t = 0
        while True:
            a = np.random.choice(4, 1)
            s, r, isEnd = env.step(a)
            G += (env.gamma ** t) * r

            if isEnd:
                break

            t += 1
        print("episode=", i, " G", G, " t=", t, "count", env.count)
        avr += G
    print(avr / 100, "avr")


if __name__ == "__main__":
    test()
