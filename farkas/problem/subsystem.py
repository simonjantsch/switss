from bidict import bidict
import numpy as np
from graphviz import Digraph
from scipy.sparse import dok_matrix

from ..model import DTMC, MDP, ReachabilityForm
from ..utils import color_from_hash, InvertibleDict

class Subsystem:
    def __init__(self, reachability_form, state_action_weights):
        assert state_action_weights.shape[0] == reachability_form.P.shape[0], (
            "result shape must be the amount of state-action pairs (%d!=%d)." % (state_action_weights.shape[0], reachability_form.P.shape[0]))
        assert ((0 <= state_action_weights) + (state_action_weights <= 1)).all(), "result has faulty values."
        self.__supersys_reachability_form = reachability_form
        self.__state_action_weights = state_action_weights
        self.__subsystem_mask = None
        self.__model = None
        self.__reachability_form = None

    @property
    def subsystem_mask(self):
        if self.__subsystem_mask is None:
            C,N = self.__supersys_reachability_form.P.shape
            self.__subsystem_mask = np.zeros(N)
            for index in range(C):
                if self.__state_action_weights[index] > 0:
                    (st,_) = self.__supersys_reachability_form.\
                        index_by_state_action.inv[index]
                    self.__subsystem_mask[st] = True

        return self.__subsystem_mask

    @property
    def reachability_form(self):
        if self.__reachability_form != None:
            return self.__reachability_form
        initial_label = self.supersys_reachability_form.initial_label
        target_label = self.supersys_reachability_form.target_label

        self.__reachability_form = ReachabilityForm(
            self.model,initial_label,target_label)

    @property
    def supersys_reachability_form(self):
        return self.__supersys_reachability_form

    @property
    def model(self):
        if self.__model != None:
            return self.__model

        state_vector = self.subsystem_mask
        reach_form = self.supersys_reachability_form
        P = reach_form.P
        C,N = P.shape

        new_to_old_states = bidict()
        new_index_by_state_action = bidict()
        new_label_to_states = InvertibleDict({},is_default=True)
        new_N = 0
        new_C = 0

        old_label_by_states = reach_form.system.labels_by_state

        # Map the new states to new indices and compute a map to the old states
        for i in range(0,N):
            if state_vector[i] == True:
                new_to_old_states[new_N] = i
                for l in old_label_by_states[i]:
                    new_label_to_states.add(l,new_N)
                new_N += 1

        # Compute the new number of choices (= rows)
        for rowidx in range(0,C):
            (source,action) = reach_form.index_by_state_action.inv[rowidx]
            if state_vector[source] == True:
                new_source = new_to_old_states.inv[source]
                new_index_by_state_action[(new_source,action)] = new_C
                new_C += 1

        target_state, fail_state = new_N, new_N+1
        new_index_by_state_action[(target_state,0)] = new_C
        new_index_by_state_action[(fail_state,0)] = new_C+1

        new_P = dok_matrix((new_C+2,new_N+2))
        new_P[new_C, target_state] = 1
        new_P[new_C+1, fail_state] = 1

        target_label = self.supersys_reachability_form.target_label
        new_label_to_states.add("target",target_state)
        new_label_to_states.add("fail",fail_state)

        not_to_fail = np.zeros(new_C)

        # Populate the new transition matrix
        for (rowidx,target), prob in P.items():
            (source,action) = reach_form.index_by_state_action.inv[rowidx]
            if state_vector[source] == True and state_vector[target] == True:
                new_source = new_to_old_states.inv[source]
                new_target = new_to_old_states.inv[target]
                new_row_idx = new_index_by_state_action[(new_source,action)]
                new_P[new_row_idx,new_target] = prob
                not_to_fail[new_row_idx] += prob

        # populate probabilities to goal and fail
        for rowidx, p_target in enumerate(reach_form.to_target):
            (source,action) = reach_form.index_by_state_action.inv[rowidx]
            if state_vector[source] == True:
                new_source = new_to_old_states.inv[source]
                new_row_idx = new_index_by_state_action[(new_source,action)]
                new_P[new_row_idx, target_state] = p_target
                new_P[new_row_idx, fail_state] = 1 - (p_target + not_to_fail[new_row_idx])

        new_initial = new_to_old_states.inv[reach_form.initial]

        if new_C == new_N:
            model = DTMC(new_P,new_label_to_states)
        else:
            model = MDP(new_P,new_index_by_state_action,{},new_label_to_states)

        return model

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

        graph = self.supersys_reachability_form.system.digraph(
            state_map=state_map, action_map=action_map)
        return graph

    def __repr__(self):
        return "Subsystem(supersys_reachability_form=%s, states=%s)" % (
            self.supersys_reachability_form, int(self.subsystem_mask.sum()))
