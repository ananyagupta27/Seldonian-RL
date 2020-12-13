import numpy as np

from .environment import Environment

from typing import Tuple

from gym import spaces


class Mountaincar(Environment):
    """
    https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2020/687_F20.pdf
    The mountaincar environment as described in the 687 course material.
     The agent must learn to exit the valley in minimum timesteps.
    Actions: left, right and no action
    Reward: -1 always
    """

    def __init__(self, discrete=False):
        self._name = "MountainCar"
        self.numActions = 3
        self._action = None
        self._reward = 0
        self._isEnd = False

        self._gamma = 1.0
        ranges = np.zeros((2, 2))
        ranges[0, :] = [-1.2, 0.5]
        ranges[1, :] = [-0.07, 0.07]
        self.observation_space = spaces.Box(ranges[:, 0], ranges[:, 1])
        self.action_space = spaces.Discrete(3)

        self._x = 0.  # horizontal position of car
        self._v = 0.  # horizontal velocity of the car

        # dynamics
        self._g = 0.0025  # gravity coeff
        self._ucoef = 0.001  # gravity coeff
        self._h = 3.0  # cosine frequency parameter
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable
        self.discrete = discrete


    @property
    def name(self):
        return "Mountaincar"

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v])

    @property
    def gamma(self) -> float:
        return self._gamma


    @property
    def horizonLength(self):
        return 100

    @property
    def threshold(self):
        """
        The threshold performance
        """
        return -390

    def getNumActions(self):
        return self.numActions

    def getStateDims(self):
        if self.discrete:
            return self.getNumDiscreteStates()
        return 16


    def getNumDiscreteStates(self):
        return int((pow(2,3)+1) * (pow(2,4)+1))

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Dynamics equations to get the next state
        """
        u = float(action - 1)
        x, v = state
        v += self._ucoef * u - self._g * np.cos(self._h * x)
        v = np.clip(v, self.observation_space.low[1], self.observation_space.high[1])
        x += v

        state = np.array([x, v])

        # capping the states
        if state[0] < self.observation_space.low[0]:
            state[0] = self.observation_space.low[0]
            state[1] = 0
        elif state[0] > self.observation_space.high[0]:
            state[0] = self.observation_space.high[0]

        return state

    def R(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        returns a reward for the transition (state,action,next_state) -1 always
        """
        return -1.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        self._action = action
        state = np.array([self._x, self._v])
        next_state = self.nextState(state, action)
        self._x, self._v = next_state

        self._reward = self.R(state, action, next_state)
        self._t += 1.0
        self._isEnd = self.terminal()

        return self.state, self._reward, self.isEnd

    def reset(self):
        """
        resets the state of the environment to the initial configuration
        """
        self._x = -0.5
        self._v = 0.
        self._isEnd = False
        self._t = 0.0
        self._action = None

    def terminal(self) -> bool:
        """
        terminates the episode if:
            timesteps are greater that 400 seconds
            mountain car exits the valley and reaches top right
        """
        # 5000
        if self._t >= 400:
            return True
        if self._x >= 0.05:
            return True
        return False


    def getDiscreteState(self, state):
        state_copy = state.copy()
        state_copy = self.normalizeState(state_copy)
        discreteX = int(state_copy[0] * pow(2, 3))
        discreteV = int(state_copy[1] * pow(2, 4))
        return int(discreteX * pow(2, 4) + discreteV)

    def normalizeState(self, state):
        for i, item in enumerate(self.observation_space.low):
            if state[i] < self.observation_space.low[i]:
                state[i] = self.observation_space.low[i]
            elif state[i] > self.observation_space.high[i]:
                state[i] = self.observation_space.high[i]

        for i, _ in enumerate(range(2)):
            state[i] = (state[i] - self.observation_space.low[i]) / (
                    self.observation_space.high[i] - self.observation_space.low[i])
        return state


# utility function to test the average return for mountaincar
def test():
    env = Mountaincar()
    avr = 0
    for i in range(1000):
        G = 0
        env.reset()
        t = 0
        while True:
            a = np.random.choice(3, 1)

            r = 0
            for _ in range(4):
                s, r, isEnd = env.step(a)
            r = r*4
            G += (env.gamma ** t) * r

            if isEnd:
                break

            t += 1
        print("episode=", i, " G", G, " t=", t, "count", t)
        avr += G
    print(avr / 1000, "avr")


if __name__ == "__main__":
    test()