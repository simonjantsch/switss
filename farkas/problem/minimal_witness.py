import numpy as np

class MinimalWitness:
    def __init__(self, system, result):
        assert result.shape[0] in system.P.shape, "result shape must be either [amount of states] or [amount of state-action pairs]."
        assert ((0 <= result) + (result <= 1)).all(), "result has faulty values."
        self.__system = system
        self.__result = result
        self.__subsystem_mask = None

    @property
    def witnesstype(self):
        if self.__result.shape[0] == self.__system.P.shape[1]:
            return "state"
        elif self.__result.shape[0] == self.__system.P.shape[0]:
            return "state-action"

    @property
    def subsystem_mask(self):
        if self.__subsystem_mask is None:
            if self.witnesstype == "state-action":
                self.__subsystem_mask = np.zeros(self.__system.P.shape[1])
                for index in range(self.__system.P.shape[0]):
                    if self.__result[index] > 0:
                        (st,_) = self.__system.index_by_state_action.inv[index]
                        self.__subsystem_mask[st] = True
            elif self.witnesstype == "state":
                self.__subsystem_mask = self.__result > 0

        return self.__subsystem_mask

    def digraph(self):
        pass

    def __repr__(self):
        return "MinimalWitness(system=%s, states=%s)" % (self.__system, self.subsystem_mask.sum())