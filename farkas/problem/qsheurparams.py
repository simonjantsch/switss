
from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    @abstractmethod
    def initialize(self, reachability_form, mode):
        pass

    def __repr__(self):
        return type(self).__name__

class Updater(ABC):
    @abstractmethod
    def update(self, x, mode):
        pass

    def __repr__(self):
        return type(self).__name__

class AllOnesInitializer(Initializer):
    def initialize(self, indicator_keys):
        return [(i,1) for i in indicator_keys]

class InverseResultUpdater(Updater):
    def update(self, x, indicator_keys):
        C = np.max(x) + 1e8
        objective = []
        for i in indicator_keys:
            objective.append((i, 1/x[i] if x[i] > 0 else C))
        return objective
