from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod

class EEA(ABC, BaseEstimator):
    def __init__(self, n_endmembers:int):
        super(EEA, self).__init__()
        self.n_endmembers = n_endmembers

    @abstractmethod
    def endmembers(self):
        pass