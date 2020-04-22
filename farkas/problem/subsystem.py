import numpy as np
from graphviz import Digraph
from ..model import DTMC, MDP
from ..utils import color_from_hash

class Subsystem:
    def __init__(self, reachability_form, state_action_weights):
        assert state_action_weights.shape[0] == reachability_form.P.shape[0], (
            "result shape must be the amount of state-action pairs (%d!=%d)." % (state_action_weights.shape[0], reachability_form.P.shape[0]))
        assert ((0 <= state_action_weights) + (state_action_weights <= 1)).all(), "result has faulty values."
        self.__reachability_form = reachability_form
        self.__state_action_weights = state_action_weights
        self.__subsystem_mask = None

    @property
    def subsystem_mask(self):
        if self.__subsystem_mask is None:
            self.__subsystem_mask = np.zeros(self.__reachability_form.P.shape[1])
            for index in range(self.__reachability_form.P.shape[0]):
                if self.__state_action_weights[index] > 0:
                    (st,_) = self.__reachability_form.index_by_state_action.inv[index]
                    self.__subsystem_mask[st] = True

        return self.__subsystem_mask

    @property 
    def reachability_form(self):
        return self.__reachability_form

    def digraph(self):
        graph = Digraph()

        def state_map(stateidx, labels):
            is_fail = stateidx == len(self.subsystem_mask)+1
            is_target = stateidx == len(self.subsystem_mask)
            in_subsystem = not is_fail and not is_target and self.subsystem_mask[stateidx]
            label = "State %d\n%s" % (stateidx,",".join(labels))

            color = None
            if in_subsystem:
                color = "deepskyblue2"
            elif is_fail:
                color = "red"
            elif is_target:
                color = "green"
            else:
                color = "azure3"

            # only label state with weight if it is a markov chain
            # otherwise the actions are labeled
            if isinstance(self.reachability_form.system, DTMC) and in_subsystem:
                weight = self.__state_action_weights[stateidx]
                # coloring works, but is disabled for now.
                # color = "gray%d" % int(100-weight*100)
                label = "{}\nweight={:.5f}".format(label, weight)
                
            return { "style" : "filled",  
                     "color" : color,  
                     "label" : label } 

        def action_map(sourceidx, action, sourcelabels):
            fail_or_target = sourceidx in [len(self.subsystem_mask), len(self.subsystem_mask)+1]
            in_subsystem = not fail_or_target and self.subsystem_mask[sourceidx]
            color, label = "black", str(action)
            if in_subsystem:
                index = self.reachability_form.index_by_state_action[(sourceidx, action)]
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

        graph = self.reachability_form.system.digraph(state_map=state_map, action_map=action_map)
        return graph

    def __repr__(self):
        return "Subsystem(reachability_form=%s, states=%s)" % (self.reachability_form, int(self.subsystem_mask.sum()))