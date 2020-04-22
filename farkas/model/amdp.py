from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np
from scipy.sparse import dok_matrix
from collections import defaultdict
from graphviz import Digraph
from bidict import bidict
import os.path
import tempfile

from ..prism import parse_label_file, prism_to_tra
from ..utils import InvertibleDict, cast_dok_matrix


class AbstractMDP(ABC):
    def __init__(self, P, index_by_state_action, label_to_actions, label_to_states):
        # transform P into dok_matrix if neccessary
        self.P = cast_dok_matrix(P)  
        self.C, self.N = self.P.shape
        # transform mapping into bidict if neccessary (applying bidict to bidict doesn't change anything)
        self.index_by_state_action = bidict(index_by_state_action)
        self.__label_to_actions_invertible = InvertibleDict(label_to_actions, is_default=True)
        self.__label_to_states_invertible = InvertibleDict(label_to_states, is_default=True)
        self.__check_correctness()

    def __check_correctness(self):
        # make sure all rows of P sum to one
        for idx,s in enumerate(self.P.sum(axis=1)):  
            assert np.round(s,9) == 1, "Sum of row %d of P is %f but should be 1." % (idx, s)
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

    @property
    def actions_by_label(self):
        return self.__label_to_actions_invertible

    @property
    def label_by_action(self):
        return self.__label_to_actions_invertible.inv

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
        reachable = set()
        active = from_set.copy()
        neighbour_iter = { "forward" : self.successors, "backward" : self.predecessors }[mode]
        while True:
            fromidx = active.pop()
            reachable.add(fromidx)
            succ = set(map(lambda sap: sap[0], neighbour_iter(fromidx)))
            active.update(succ.difference(reachable))
            if len(active) == 0:
                break
        return reachable

    def predecessors(self, fromidx):
        """Yields an iterator that computes state-action-probability-pairs (s,a,p) such that
        applying action a to state s yields the given state with probability p.
        
        :param fromidx: The given state.
        :type fromidx: int
        :yield: A state-action-pair (s,a,p)
        :rtype: Tuple[int, int, float]
        """        
        for (idx,_), p in self.P[:,fromidx].items():
            if p > 0:
                tpl = self.index_by_state_action.inv[idx]
                yield tpl[0], tpl[1], p

    def successors(self, fromidx):
        """Yields an iterator that computes state-action-probability-pairs (d,a,p) where applying action a to
        the given state yields state d with probability p.
        
        :param fromidx: The given state.
        :type fromidx: int
        :yield: A state-action-probability-pair (d,a,p)
        :rtype: Tuple[int,int,float]
        """        
        saps = filter(lambda key: key[0] == fromidx, self.index_by_state_action.keys())
        for _, action in saps:
            idx = self.index_by_state_action[(fromidx, action)]
            for (_,dest), p in self.P[idx,:].items():
                if p > 0:
                    yield dest, action, p

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
