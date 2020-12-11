from abc import ABC

import numpy as np

from gym import spaces
import copy
from .environment import Environment


class Gridworldv2(Environment):


    def __init__(self, size=4, gamma=1, horizonLength=100, discrete=True):
        self.size = int(size)
        self._gamma = gamma
        self._horizonLength = horizonLength
        self.numStates = size**2
        self.numActions = 4
        self.observation_space = spaces.Box(low=np.zeros(self.numActions), high=np.ones(self.numActions), dtype=np.float32)
        self.action_space = spaces.Discrete(self.numActions)
        self.x = 0
        self.y = 0
        self.count = 0
        self.discrete = discrete

    @property
    def name(self):
        return "Gridworldv2"

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
        return 30


    def getNumActions(self):
        return self.numActions


    def getStateDims(self):
        return self.numStates


    def step(self, action):
        self.count += 1
        a = int(action)

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


        if self.x == self.size - 1 and self.y == self.size - 1:
            reward = 100
        else:
            reward = -1

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