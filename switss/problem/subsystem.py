from bidict import bidict
import numpy as np
from graphviz import Digraph
from scipy.sparse import dok_matrix
from collections import defaultdict

from ..model import DTMC, MDP, ReachabilityForm
from ..utils import color_from_hash, InvertibleDict

class Subsystem:
    def __init__(self, supersystem, certificate, certform, ignore_consistency_checks=False):
        assert isinstance(supersystem, ReachabilityForm)
        assert certform in ["min","max"]
        D,C,N = certificate.shape[0], supersystem.system.P.shape[0], supersystem.system.P.shape[1]
        # can be read as "certform == max ==> D == C"
        assert certform != "max" or D == C-2, "certificate shape must be the amount of state-action pairs - 2 (%d!=%d)." % (D, C-2)
        assert certform != "min" or D == N-2, "certificate shape must be the amount of states - 2 (%d!=%d)." % (D, N-2) 
        # assert ((0 <= certificate) + (certificate <= 1)).all(), "result has faulty values."
        
        self.__supersys = supersystem
        self.__certificate = certificate 
        self.__certform = certform
        self.__subsystem_mask = None
        self.__subsys = None
        self.__ignore_consistency_checks = ignore_consistency_checks

    @property
    def certform(self):
        return self.__certform
    
    @property
    def certificate(self):
        return self.__certificate

    @property
    def subsystem_mask(self):
        if self.__subsystem_mask is None:
            C,N = self.__supersys.system.P.shape
            self.__subsystem_mask = np.zeros(N-2)
            if self.__certform == "max":
                for index in range(C-2):
                    if self.certificate[index] > 0:
                        (st,_) = self.__supersys.system.index_by_state_action.inv[index]
                        self.__subsystem_mask[st] = True
            else:
                self.__subsystem_mask = self.certificate > 0

        return self.__subsystem_mask

    @property
    def supersys(self):
        return self.__supersys

    @property
    def subsys(self):
        if self.__subsys != None:
            return self.__subsys

        state_vector = self.subsystem_mask
        reach_form = self.supersys
        P = reach_form.system.P
        C,N = P.shape

        new_to_old_states = bidict()
        new_index_by_state_action = bidict()
        new_label_to_states = defaultdict(set)
        new_label_to_actions = defaultdict(set) 
        new_N = 0
        new_C = 0

        old_label_by_states = reach_form.system.labels_by_state

        # Map the new states to new indices and compute a map to the old states
        for i in range(N-2):
            if state_vector[i] == True:
                new_to_old_states[new_N] = i
                for l in old_label_by_states[i]:
                    new_label_to_states[l].add(new_N)
                new_N += 1

        # Compute the new number of choices (= rows)
        for rowidx in range(C-2):
            (source,action) = reach_form.system.index_by_state_action.inv[rowidx]
            if state_vector[source] == True:
                actionlabels = reach_form.system.labels_by_action[(source,action)]
                new_source = new_to_old_states.inv[source]
                new_index_by_state_action[(new_source,action)] = new_C
                new_label_to_actions[(new_source,action)] = actionlabels
                new_C += 1

        fail_state, target_state = new_N + 1, new_N
        new_index_by_state_action[(fail_state,0)] = new_C+1
        new_index_by_state_action[(target_state,0)] = new_C
        
        new_label_to_states[reach_form.target_label] = {target_state}
        new_label_to_states[reach_form.fail_label] = {fail_state}

        new_P = dok_matrix((new_C+2,new_N+2))
        new_P[new_C+1, fail_state] = 1
        new_P[new_C, target_state] = 1

        not_to_fail = np.zeros(new_C)

        # Populate the new transition matrix
        for (rowidx,target), prob in P.items():
            (source,action) = reach_form.system.index_by_state_action.inv[rowidx]
            if target >= N-2 or source >= N-2:
                # P also contains target & fail state - but state_vector only has N-2 entries
                continue

            if state_vector[source] == True and state_vector[target] == True:
                new_source = new_to_old_states.inv[source]
                new_target = new_to_old_states.inv[target]
                new_row_idx = new_index_by_state_action[(new_source,action)]
                new_P[new_row_idx,new_target] = prob
                not_to_fail[new_row_idx] += prob

        # populate probabilities to fail
        # maps every target state with probability 1 to itself.
        for rowidx, p_target in enumerate(reach_form.to_target):
            (source,action) = reach_form.system.index_by_state_action.inv[rowidx]
            if state_vector[source] == True:
                new_source = new_to_old_states.inv[source]
                new_row_idx = new_index_by_state_action[(new_source,action)]
                new_P[new_row_idx, target_state] = p_target 
                new_P[new_row_idx, fail_state] = 1 - (p_target + not_to_fail[new_row_idx])

        # model type is same as supersystems model type.
        # if model type is DTMC, additional parameters are ignored.
        modeltype = type(self.supersys.system)
        model = modeltype(P=new_P, 
            index_by_state_action=new_index_by_state_action, 
            label_to_actions=dict(new_label_to_actions), 
            label_to_states=dict(new_label_to_states))

        model = ReachabilityForm(
            model, 
            self.supersys.initial_label, 
            self.supersys.target_label, 
            self.supersys.fail_label,
            ignore_consistency_checks=self.__ignore_consistency_checks)

        self.__subsys = model
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

            # only label state with weight if it is a markov chain or certform is "min"
            # otherwise the actions are labeled
            if (self.certform == "min" or isinstance(self.subsys.system, DTMC)) and in_subsystem:
                cert = self.certificate[stateidx]
                # coloring works, but is disabled for now.
                # color = "gray%d" % int(100-weight*100)
                label = "{}\nc[{}]={:.5f}".format(label, stateidx, cert)
                
            return { "style" : "filled",  
                     "color" : color,  
                     "label" : label } 

        def action_map(sourceidx, action, sourcelabels):
            fail_or_target = sourceidx in [len(self.subsystem_mask), len(self.subsystem_mask)+1]
            in_subsystem = not fail_or_target and self.subsystem_mask[sourceidx]
            color, label = "black", str(action)
            if self.certform == "max" and in_subsystem:
                index = self.supersys.index_by_state_action[(sourceidx, action)]
                cert = self.certificate[index]
                # coloring works, but is disabled for now.
                # color = "gray%d" % int(weight*100)
                label = "{}\nc[{}]={:.5f}".format(label, index, cert)
                
            return { "node" : { "color" : color,   
                                "label" : label,
                                "style" : "solid",   
                                "shape" : "rectangle" },   
                     "edge" : { "color" : color,  
                                "dir" : "none" } }

        graph = self.supersys.system.digraph(state_map=state_map, action_map=action_map)
        return graph

    def __repr__(self):
        return "Subsystem(supersys=%s, subsys=%s)" % (self.supersys, self.subsys)
