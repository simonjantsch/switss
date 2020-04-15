import numpy as np
from graphviz import Digraph
from ..model import DTMC, MDP
from ..utils import color_from_hash

class Subsystem:
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

    @property
    def is_quadratic(self):
        return self.__system.P.shape[0] == self.__system.P.shape[1]

    def digraph(self):
        graph = Digraph()
        model = None

        def state_map(stateidx, labels):
            in_subsystem = (stateidx < len(self.subsystem_mask) and self.subsystem_mask[stateidx])
            label = "State %d\n%s" % (stateidx,",".join(labels))
            color = color_from_hash( tuple([sorted(labels), in_subsystem]) )
            if self.is_quadratic and in_subsystem:
                weight = self.__state_action_weights[stateidx]
                # coloring works, but is disabled for now.
                # color = "gray%d" % int(100-weight*100)
                label = "{}\nweight={:.5f}".format(label, weight)
                
            return { "style" : "filled",  
                     "color" : color,  
                     "label" : label } 

        def action_map(sourceidx, action, sourcelabels):
            in_subsystem = (sourceidx < len(self.subsystem_mask) and self.subsystem_mask[sourceidx])
            color, label = "black", str(action)
            if in_subsystem:
                index = self.__system.index_by_state_action[(sourceidx, action)]
                weight = self.__state_action_weights[index]
                # coloring works, but is disabled for now.
                # color = "gray%d" % int(weight*100)
                label = "{}\nweight={:.5f}".format(label, weight)
                
            return { "node" : { "color" : color,   
                                "label" : label,
                                "style" : "solid",   
                                "shape" : "rectangle" },   
                     "edge" : { "color" : color,  
                                "dir" : "none" } }

        if self.is_quadratic:
            model = DTMC.from_reachability_form(self.__system)
            graph = model.digraph(state_map=state_map)
        else:
            model = MDP.from_reachability_form(self.__system)
            graph = model.digraph(state_map=state_map, action_map=action_map)

        return graph

    def __repr__(self):
        return "Subsystem(system=%s, states=%s)" % (self.__system, int(self.subsystem_mask.sum()))