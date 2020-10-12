from abc import ABC

import numpy as np

from gym import spaces
import copy



class Gridworld(object):
    def __init__(self, size=5, gamma=1):
        self.size = int(size)
        self.x = int(0)
        self.y = int(0)
        self.count = 0
        nums = self.size ** 2
        self.nums = nums
        self.numa = 4
        self.observation_space = spaces.Box(low=np.zeros(nums), high=np.ones(nums), dtype=np.float32)
        self.action_space = spaces.Discrete(self.numa)
        self._P = None
        self._R = None
        self._gamma = gamma

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def isEnd(self) -> bool:
        return self.terminal()

    @property
    def state(self) -> np.ndarray:
        return np.array([self.get_state()])

    def reset(self):
        self.x = 0
        self.y = 0
        self.count = 0
        return self.get_state()

    def step(self, action):

        a = int(action)
        s = np.random.uniform(0, 1)
        if s < 0.1:
            return self.get_state(), -1, self.terminal()
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
        self.count += 1
        reward = -1.0

        if self.terminal():
            reward = 0

        return self.get_state(), reward, self.terminal()

    def get_state(self):
        x = np.zeros(self.nums, dtype=np.float32)
        x[self.x * self.size + self.y] = 1
        return self.x * self.size + self.y

    def terminal(self):
        return (self.x == self.size - 1 and self.y == self.size - 1) or (self.count > 500)


