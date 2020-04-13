
from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    @abstractmethod
    def initialize(self, reachability_form):
        pass

    def __repr__(self):
        return type(self).__name__

class Updater(ABC):
    @abstractmethod
    def update(self, x):
        pass

    def __repr__(self):
        return type(self).__name__

class AllOnesInitializer(Initializer):
    def initialize(self, reachability_form):
        return np.ones(reachability_form.P.shape[1])

class InverseResultUpdater(Updater):
    def update(self, x):
        C = np.max(x) + 1e8
        return np.array([1/v if v > 0 else C for v in x])