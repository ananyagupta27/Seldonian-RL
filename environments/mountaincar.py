import numpy as np

from .environment import Environment

from typing import Tuple

from gym import spaces


class Mountaincar(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.
    Actions: left (0) and right (1)
    Reward: 1 always
    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    Borrowed from Phil Thomas's RL course Fall 2019. Written by Blossom Metevier and Scott Jordan
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
        self.observation_space = spaces.Box(ranges[:, 0], ranges[:, 1], dtype=np.float64)
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
        return 1000

    @property
    def threshold(self):
        """
        The threshold performance
        """
        return 58.5

    def getNumActions(self):
        return self.numActions

    def getStateDims(self):
        if self.discrete:
            return self.getNumDiscreteStates()
        return 16


    def getNumDiscreteStates(self):
        return int((pow(2,4)+1) * (pow(2,4)+1))

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        u = float(action - 1)
        x, v = state
        v += self._ucoef * u - self._g * np.cos(self._h * x)
        v = np.clip(v, self.observation_space.low[1], self.observation_space.high[1])
        x += v

        state = np.array([x, v])

        if state[0] < self.observation_space.low[0]:
            state[0] = self.observation_space.low[0]
            state[1] = 0
        elif state[0] > self.observation_space.high[0]:
            state[0] = self.observation_space.high[0]

        return state

    def R(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        returns a reward for the transition (state,action,next_state)
        """
        if self.terminal() and self._x >= 0.05:
            return 200
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
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # 5000
        if self._t >= 1000:
            return True
        if self._x >= 0.05:
            return True
        return False


    def getDiscreteState(self, state):
        # print(state)
        state = self.normalizeState(state)
        discreteX = int(state[0] * pow(2, 4))
        discreteV = int(state[1] * pow(2, 4))
        # print(state, discreteX, discreteV, int(discreteX * pow(2, 4) + discreteV))

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
            for _ in range(5):
                s, r, isEnd = env.step(a)
            r = r*5
            # s, r, isEnd = env.step(a)
            # print(s, r, isEnd)
            G += (env.gamma ** t) * r

            if isEnd:
                break

            t += 1
        print("episode=", i, " G", G, " t=", t, "count", t)
        avr += G
    print(avr / 1000, "avr")


if __name__ == "__main__":
    test()