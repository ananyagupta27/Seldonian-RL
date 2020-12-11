from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Must extend this class and implement the following functions to create a new optimizer
    """

    @abstractmethod
    def name(self):
        pass


    @abstractmethod
    def run_optimizer(self, verbose):
        pass
