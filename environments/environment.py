from abc import ABC, abstractmethod


class Environment(ABC):
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

    @abstractmethod
    def getNumActions(self):
        pass

    @abstractmethod
    def getStateDims(self):
        pass

    @abstractmethod
    def step(self, action):
        """
        An action is taken in the environment and the next
        state is entered.
        output:
            state -- the next state
            reward -- the reward from taking the action
            isEnd -- True if environment reset is required
        """
        pass

    @abstractmethod
    def nextState(self, state, action):
        """
        Provides the next state of the environment given an environment state
        and an intended action.
        output:
            nextState: the next state
        """
        pass

    @abstractmethod
    def reset(self):
        """
        The environment is reset.
        """
        pass

    @abstractmethod
    def R(self, state, action, nextState):
        """
        The reward function. Defines the signal sent to the
        learning agent as it interacts in the environment.
        output:
            reward -- the reward resulting from taking the
                        last action in the environment.
        """
        pass
