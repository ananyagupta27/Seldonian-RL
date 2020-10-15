from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def name(self):
        pass


    @abstractmethod
    def run_optimizer(self, verbose):
        pass
