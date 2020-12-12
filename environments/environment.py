from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Implement this class for any additional environments
    """

    @property
    @abstractmethod
    def name(self):
        """
        Name of the environment
        :return:
        """
        pass

    @property
    @abstractmethod
    def gamma(self):
        """
        The reward discount factor.
        """
        pass

    @property
    @abstractmethod
    def horizonLength(self):
        """
        The maximum horizon length
        """
        pass


    @property
    @abstractmethod
    def threshold(self):
        """
        The threshold for performance to be satisfied with high confidence
        """
        pass

    @property
    @abstractmethod
    def state(self):
        """
        The current state of the environment.
        """
        pass

    @property
    @abstractmethod
    def isEnd(self):
        """
        True if the environment needs to be reset and False
        otherwise.
        """
        pass

    @abstractmethod
    def getNumActions(self):
        """
        :return: number of actions possible
        """
        pass

    @abstractmethod
    def getStateDims(self):
        """
        :return: the dimenstions of the states
        if using fourier basis
        """
        pass

    @abstractmethod
    def getDiscreteState(self, state):
        """
        Takes state as input and discretizes it
        :param state:
        :return: discretized state
        """
        pass

    @abstractmethod
    def getNumDiscreteStates(self):
        """
        :return: total the number of discrete states
        """
        pass



    @abstractmethod
    def step(self, action):
        """
        Take the given action in the environment to transition to next state
        :param action:
        :return:
        """
        pass

    @abstractmethod
    def nextState(self, state, action):
        """
        According to the transition function take action in the given state
        :param state:
        :param action:
        :return: next state
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the environment and all states
        :return:
        """
        pass

    @abstractmethod
    def R(self, state, action, nextState):
        """
        return reward for this particular transition
        :param state:
        :param action:
        :param nextState:
        :return:
        """
        pass


