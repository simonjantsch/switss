
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
    def initialize(self, reachability_form, mode):
        assert mode in ["max","min"]
        # max-form -> requires action-state-dimension (C)
        # min-form -> requires state-dimension (N)
        dim = { "max" : 0, "min" : 1}[mode]
        return np.ones(reachability_form.P.shape[dim])

class InverseResultUpdater(Updater):
    def update(self, x, mode):
        assert mode in ["max","min"]
        C = np.max(x) + 1e8
        return np.array([1/v if v > 0 else C for v in x])