from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np
from scipy.sparse import dok_matrix
from collections import defaultdict
from graphviz import Digraph
from bidict import bidict
import os.path
import tempfile

from . import ReachabilityForm
from ..prism import parse_label_file, prism_to_tra
from ..utils import InvertibleDict, array_to_dok_matrix


class AbstractMDP(ABC):
    def __init__(self, P, index_by_state_action, label_to_states):
        # transform P into dok_matrix if neccessary
        self.P = P if isinstance(P, dok_matrix) else array_to_dok_matrix(P)  
        self.C, self.N = self.P.shape
        # transform mapping into bidict if neccessary (applying bidict to bidict doesn't change anything)
        self.index_by_state_action = bidict(index_by_state_action)
        self.__label_to_states_invertible = InvertibleDict(label_to_states, is_default=True, default=set)
        self.__check_correctness()

    def __check_correctness(self):
        # make sure all rows of P sum to one
        for idx,s in enumerate(self.P.sum(axis=1)):  
            assert s == 1, "Sum of row %d of P is %d but should be 1." % (idx, s)
        # make sure that all values x are 0<=x<=1
        for (i,j), p in self.P.items():
            assert p >= 0 and p <= 1, "P[%d,%d]=%f, violating 0<=%f<=1." % (i,j,p,p)

    @property
    def states_by_label(self):
        """Returns a mapping from labels to states.
        
        :return: The mapping.
        :rtype: Dict[str, Set[int]]
        """        
        return self.__label_to_states_invertible

    @property
    def labels_by_state(self):
        """Returns a mapping from states to labels.
        
        :return: The mapping.
        :rtype: Dict[int, Set[str]]
        """        
        return self.__label_to_states_invertible.inv

    def reachable_set(self, from_set, mode):
        """Computes the set of states that are reachable from the 'from_set'.
        
        :param from_set: The set of states the search should start from.
        :type from_set: Set[int]
        :param mode: Either 'forward' or 'backward'. Defines the direction of search.
        :type mode: str
        :return: The set of states that are reachable.
        :rtype: Set[int]
        """        
        assert mode in ["forward", "backward"], "Mode must be either 'forward' or 'backward' but is %s." % mode
        reachable = from_set.copy()
        neighbour_iter = { "forward" : self._successors, "backward" : self._predecessors }[mode]
        while True:
            old_count = len(reachable)
            newtmp = set({})
            for fromidx in reachable:
                succ = set(map(lambda sap: sap[0], neighbour_iter(fromidx)))
                newtmp.update(succ)
            reachable.update(newtmp)
            if old_count == len(reachable):
                break
        return reachable

    def reachability_form(self, initial_label, targets_label, debug=False):
        """Computes the reachability form of this model, an initial state (index) and a set of target states (indices).
        Will do a forwards (states which are reachable from the initial state) and backwards (states which are able to 
        reach the target states) reachability test on the states of the given MDP. Removes states which fail the forwards 
        reachability test. Creates a unique goal state to which all target states move with probability one.
        All fail states (states which fail the backwards reachability test) are mapped to a single fail state.
        
        :param initial_label: The label of the initial state, which must yield exactly one initial state.
        :type initial_label: str
        :param targets_label: The label of the target states, which must yield at least one target state.
        :type targets_label: str
        :return: The reduced model in reachability form and a mapping from states of this model to states of the reachability form. 
        :rtype: Tuple[model.ReachabilityForm, utils.InvertibleDict]
        """   
        assert len(self.states_by_label[targets_label]) > 0, "There needs to be at least one target state."
        target_states = self.states_by_label[targets_label]
        initial_state_count = len(self.states_by_label[initial_label])
        assert initial_state_count == 1, "There were %d initial states given. Must be 1." % initial_state_count
        initial = list(self.states_by_label[initial_label])[0]

        backward_reachable = self.reachable_set(target_states, "backward")
        forward_reachable = self.reachable_set(set([initial]), "forward")
        # states which are reachable from the initial state AND are able to reach target states
        reachable = backward_reachable.intersection(forward_reachable)
        
        if debug:
            print("tested backward & forward reachability test: %s" % reachable)

        def reachable_from_non_target_states(ts):
            # if ts is a target state, then
            # ts reachable from non-target states <=> there is a predecessor of ts which is not a target state
            predecessors = map(lambda sap: sap[0], self._predecessors(ts))
            return ts not in target_states or len(set(predecessors).difference(target_states)) != 0
        # remove target states that are only reachable from other target states
        reachable = set(filter(reachable_from_non_target_states, reachable))
        
        if debug:
            print("removed target states that are only reachable from other target states: %s" % reachable)
        
        # fix some kind of order
        reachable = list(reachable)

        # reduce states + new target and new fail state 
        new_state_count = len(reachable) + 2
        target_idx, fail_idx = new_state_count - 2, new_state_count - 1
        
        if debug:
            print("new states: %s, target index: %s, fail index: %s" % (new_state_count, target_idx, fail_idx))
        
        # create a mapping from this to reachability form
        to_reachability = {}
        # [0...len(reachable)-1] reachable (mapped to respective states)
        # [len(reachable)...M-1] not in reachable but target states (mapped to target)
        # [M-1...N-1] neither reachable nor target states (mapped to fail)
        # overall N entries in "to_reachability"
        for stateidx in range(self.N):
            newidx = None
            if stateidx in reachable:
                # state is reachable
                newidx = reachable.index(stateidx) # result is something in [0,...len(reachable)-1]
            elif targets_label in self.labels_by_state[stateidx]:
                # state is not in reachable but a target state
                # => map to target state
                newidx = target_idx
            else:
                # state is not reachable and not a target state
                # => map to fail state
                newidx = fail_idx
            to_reachability[stateidx] = newidx
        
        if debug:
            print("new state mapping: %s" % to_reachability)
        
        # make dictionary invertible -> this is useful since it allows mapping back from the reachability form to full form
        to_reachability = InvertibleDict(to_reachability, is_default=True, default=set)

        # compute reduced transition matrix (without non-reachable states)
        # compute probability of reaching the target state in one step 
        new_N = len(reachable)
        new_C = len(set([(x,y) for (x,y) in self.index_by_state_action.keys() if x in reachable]))
        new_P = dok_matrix((new_C, new_N))
        new_index_by_state_action = bidict()
        to_target = np.zeros(new_C)

        i = 0
        for (sapidx, destidx), p in self.P.items():
            sourceidx, action = self._state_action_pair_by_idx(sapidx)
            new_sourceidx, new_destidx = to_reachability[sourceidx], to_reachability[destidx]

            if new_sourceidx not in [target_idx, fail_idx]:

                if (new_sourceidx, action) not in new_index_by_state_action:
                    new_index_by_state_action[new_sourceidx, action] = i
                    i += 1
                index = new_index_by_state_action[(new_sourceidx,action)]

                if targets_label in self.labels_by_state[sourceidx]:
                    # if old source is a target state, then it is remapped to the new target state with p=1
                    to_target[index] = 1
                elif new_destidx not in [target_idx, fail_idx]:
                    # new transition matrix
                    new_P[index, new_destidx] = p

        if debug:
            print("new transition matrix: \n%s" % new_P)
            print("to_target: %s" % to_target)

        return ReachabilityForm(new_P, initial, to_target, new_index_by_state_action), to_reachability


    def _predecessors(self, fromidx):
        """Yields an iterator that computes state-action-probability-pairs (s,a,p) such that
        applying action a to state s yields the given state with probability p.
        
        :param fromidx: The given state.
        :type fromidx: int
        :yield: A state-action-pair (s,a,p)
        :rtype: Tuple[int, int, float]
        """        
        for (idx,_), p in self.P[:,fromidx].items():
            if p > 0:
                tpl = self._state_action_pair_by_idx(idx)
                yield tpl[0], tpl[1], p

    def _successors(self, fromidx):
        """Yields an iterator that computes state-action-probability-pairs (d,a,p) where applying action a to
        the given state yields state d with probability p.
        
        :param fromidx: The given state.
        :type fromidx: int
        :yield: A state-action-probability-pair (d,a,p)
        :rtype: Tuple[int,int,float]
        """        
        saps = filter(lambda key: key[0] == fromidx, self.index_by_state_action.keys())
        for _, action in saps:
            idx = self._idx_by_state_action_pair(fromidx, action)
            for (_,dest), p in self.P[idx,:].items():
                if p > 0:
                    yield dest, action, p

    def _idx_by_state_action_pair(self, state, action):
        """Computes the index of a state-action-pair in the transition matrix.
        
        :param state: The state.
        :type state: int
        :param action: The action.
        :type action: int
        :return: The index of the state-action-pair.
        :rtype: int
        """        
        return self.index_by_state_action[(state, action)]
    
    def _state_action_pair_by_idx(self, idx):
        """Computes the state and the action of a row-index of the transition matrix.
        
        :param idx: The row-index.
        :type idx: int
        :return: The corresponding state and action.
        :rtype: Tuple[int,int]
        """        
        return self.index_by_state_action.inv[idx]

    @classmethod
    def from_file(cls, label_file_path, tra_file_path):
        """Computes an instance of this model from a given .lab and .tra file.
        
        :param label_file_path: Path of .lab-file.
        :type label_file_path: str
        :param tra_file_path: Path of .tra-file.
        :type tra_file_path: str
        :return: Instance of this model.
        :rtype: [This Model]
        """        
        # identify all states
        states_by_label, _, _ = parse_label_file(label_file_path)
        # then load the transition matrix
        res = cls._load_transition_matrix(tra_file_path)
        return cls(*res, states_by_label)

    @classmethod
    def from_reachability_form(cls, reachability_form):
        """Computes an instance of this model from a given reachability form.
        
        :param reachability_form: The reachability form.
        :type reachability_form: model.ReachabilityForm
        :return: Instance of this model.
        :rtype: [This Model]
        """        
        C,N = reachability_form.P.shape
        P_compl = dok_matrix((C+2, N+2))
        target_state, fail_state = N, N+1

        index_by_state_action_compl = reachability_form.index_by_state_action.copy()
        index_by_state_action_compl[(target_state,0)] = C
        index_by_state_action_compl[(fail_state,0)] = C+1
        label_to_states = { "init" : {reachability_form.initial}, "target" : {target_state}, "fail" : {fail_state}}

        not_to_fail = np.zeros(N)
        for (idx, dest), p in reachability_form.P.items():
            sourceidx, action = index_by_state_action_compl.inv[idx]
            if p > 0:
                not_to_fail[sourceidx] += p
                P_compl[idx, dest] = p

        for idx, p_target in enumerate(reachability_form.to_target):
            sourceidx, action = index_by_state_action_compl.inv[idx]
            if p_target > 0:
                P_compl[idx, target_state] = p_target
            p_fail = 1 - (p_target + not_to_fail[sourceidx])
            if p_fail > 0:
                P_compl[idx, fail_state] = p_fail

        P_compl[C, target_state] = 1
        P_compl[C+1, fail_state] = 1

        return cls( P=P_compl, 
                    index_by_state_action=index_by_state_action_compl, 
                    label_to_states=label_to_states)
        
    @classmethod
    def from_prism_model(cls, model_file_path, prism_constants = {}, extra_labels = {}):
        """Computes an instance of this model from a PRISM model.
        
        :param model_file_path: File path of model without file type (e.g. tra or lab).
        :type model_file_path: str
        :param prism_constants: A dictionary of constants to be assigned in the model, defaults to {}.
        :type prism_constants: Dict[str,int], optional
        :param extra_labels: A dictionary that defines additional labels (than the ones defined in the prism module) to 
            be added to the .lab file. The keys are label names and the values are PRISM expressions over the module variables, 
            defaults to {}.
        :type extra_labels: Dict[str,str], optional
        :return: Instance of this model.
        :rtype: [This Model]
        """ 
        with tempfile.TemporaryDirectory() as tempdirname:
            temp_model_file = os.path.join(tempdirname, "model")
            temp_tra_file = temp_model_file + ".tra"
            temp_lab_file = temp_model_file + ".lab"
            if prism_to_tra(model_file_path,temp_model_file,prism_constants,extra_labels):
                return cls.from_file(temp_lab_file,temp_tra_file)
            else:
                assert False, "Prism call to create model failed."
        
    @abstractmethod
    def save(self, filepath):
        """Saves the .tra and .lab-file according to the given filepath.
        
        :param filepath: the file path
        :type filepath: str
        :return: path of .tra and .lab-file
        :rtype: Tuple[str,str]
        """     
        pass

    @abstractmethod
    def digraph(self, state_map = None, trans_map = None, action_map = None):
        pass

    @abstractclassmethod
    def _load_transition_matrix(cls, filepath):
        pass

    def __repr__(self):
        return "%s(C=%s, N=%s, labels={%s})" % (
                        type(self).__name__, 
                        self.C, 
                        self.N, 
                        ", ".join(["%s (%d)" % (k, len(v)) for k,v in self.states_by_label.items()]))