from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Implement this class for any additional environments
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def isEnd(self):
        """
        True if the environment needs to be reset and False
        otherwise.
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
    def gamma(self):
        """
        The reward discount factor.
        """
        pass

    @property
    @abstractmethod
    def horizonLength(self):
        """
        The reward discount factor.
        """
        pass

    @property
    @abstractmethod
    def threshold(self):
        """
        The threshold performance.
        """
        pass


    @abstractmethod
    def getNumActions(self):
        pass

    @abstractmethod
    def getStateDims(self):
        pass

    @abstractmethod
    def step(self, action):
        """

        :param action:
        :return:
        """
        pass

    @abstractmethod
    def nextState(self, state, action):
        """

        :param state:
        :param action:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """

        :return:
        """
        pass

    @abstractmethod
    def R(self, state, action, nextState):
        """

        :param state:
        :param action:
        :param nextState:
        :return:
        """
        pass


    @abstractmethod
    def getDiscreteState(self, state):
        """

        :param state:
        :return:
        """
        pass

    @abstractmethod
    def getNumDiscreteStates(self):
        """

        :return:
        """
        pass