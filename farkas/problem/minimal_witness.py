import numpy as np
from graphviz import Digraph
from ..model import DTMC, MDP
from ..utils import color_from_hash

class MinimalWitness:
    def __init__(self, system, state_action_weights):
        assert state_action_weights.shape[0] == system.P.shape[0], (
            "result shape must be the amount of state-action pairs (%d!=%d)." % (state_action_weights.shape[0], system.P.shape[0]))
        assert ((0 <= state_action_weights) + (state_action_weights <= 1)).all(), "result has faulty values."
        self.__system = system
        self.__state_action_weights = state_action_weights
        self.__subsystem_mask = None

    @property
    def subsystem_mask(self):
        if self.__subsystem_mask is None:
            self.__subsystem_mask = np.zeros(self.__system.P.shape[1])
            for index in range(self.__system.P.shape[0]):
                if self.__state_action_weights[index] > 0:
                    (st,_) = self.__system.index_by_state_action.inv[index]
                    self.__subsystem_mask[st] = True

        return self.__subsystem_mask

    def digraph(self):
        graph = Digraph()
        model = None

        def state_map(stateidx, labels):
            in_subsystem = (stateidx < len(self.subsystem_mask) and self.subsystem_mask[stateidx])
            color = color_from_hash( tuple([sorted(labels), in_subsystem]) )
            
            return { "style" : "filled",  
                     "color" : color,  
                     "label" : "State %d\n%s" % (stateidx,",".join(labels)) } 

        def action_map(sourceidx, action, sourcelabels):
            return { "node" : { "color" : "black",   
                                "label" : str(action),  
                                "style" : "solid",   
                                "shape" : "rectangle" },   
                     "edge" : { "color" : "black",  
                                "dir" : "none" } }

        if self.__system.P.shape[0] == self.__system.P.shape[1]:
            model = DTMC.from_reachability_form(self.__system)
            graph = model.digraph(state_map=state_map)
        else:
            model = MDP.from_reachability_form(self.__system)
            graph = model.digraph(state_map=state_map, action_map=action_map)

        return graph

    def __repr__(self):
        return "MinimalWitness(system=%s, states=%s)" % (self.__system, self.subsystem_mask.sum())