from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np
from scipy.sparse import dok_matrix
from collections import defaultdict
from graphviz import Digraph
from bidict import bidict
import os.path
import tempfile

from switss.utils.graph import Graph
from ..prism import parse_label_file, prism_to_tra
from ..utils import InvertibleDict, cast_dok_matrix


class AbstractMDP(ABC):
    """Abstract superclass for Markov Decision Processes (MDPs) and Discrete Time Markov Chains (DTMCs)
    that supports labeling for states and actions, getting successors/predecessors, computing 
    reachability sets, rendering of MDPs/DTMCs as graphviz digraphs, loading from .pm-files and 
    loading/storing from/to .lab,.tra files. 
    """    
    def __init__(self, P, index_by_state_action, label_to_actions={}, label_to_states={}):
        """Instantiates an AbstractMDP from a transition matrix, a bidirectional
        mapping from state-action pairs to corresponding transition matrix entries and labelings for states and actions.

        :param P: :math:`C_{S_{\\text{all}}} \\times N_{S_{\\text{all}}}` transition matrix. 
        :type P: Either 2d-list, numpy.matrix, numpy.array or scipy.sparse.spmatrix
        :param index_by_state_action: A bijection of state-action pairs :math:`(s,a) \in \mathcal{M}_{S_{\\text{all}}}` 
            to indices :math:`i=0,\dots,C_{S_{\\text{all}}}-1` and vice versa.
        :type index_by_state_action: Dict[Tuple[int,int],int]
        :param label_to_actions: Mapping from labels to sets of state-action pairs.
        :type label_to_actions: Dict[str,Set[Tuple[int,int]]]
        :param label_to_states: Mapping from labels to sets of states.
        :type label_to_states: Dict[str,Set[int]]
        """        
        # transform P into dok_matrix if neccessary
        self.P = cast_dok_matrix(P)
        # for fast column & row slicing
        self.__P_csc = self.P.tocsc()
        self.__P_csr = self.P.tocsr()
        self.C, self.N = self.P.shape
        # transform mapping into bidict if neccessary (applying bidict to bidict doesn't change anything)
        self.index_by_state_action = bidict(index_by_state_action)
        if isinstance(label_to_actions,InvertibleDict):
            self.__label_to_actions_invertible = label_to_actions
        else:
            self.__label_to_actions_invertible = InvertibleDict(label_to_actions, is_default=True)
        if isinstance(label_to_states,InvertibleDict):
            self.__label_to_states_invertible = label_to_states
        else:
            self.__label_to_states_invertible = InvertibleDict(label_to_states, is_default=True)
        self.__check_correctness()
        self.__available_actions = None
        self.__graph = Graph(self.P, self.index_by_state_action)

    def __check_correctness(self):
        """Validates correctness of a this model by checking

        * :math:`\sum_j P_{(i,j)} = 1 \quad \\forall i \in \{ 1,2,\dots,C \}`
        
        * :math:`0 \leq P_{(i,j)} \leq 1 \quad \\forall (i,j) \in \{1,\dots,C\} \\times \{1,\dots,N\}`
        
        """
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
        """Returns a mapping from labels to actions (given as state-action-pairs).

        :return: The mapping.
        :rtype: Dict[str, Set[Tuple[int,int]]
        """        
        return self.__label_to_actions_invertible

    @property
    def labels_by_action(self):
        """Returns a mapping from actions (given as state-action-pairs) to labels.

        :return: The mapping.
        :rtype: Dict[Tuple[int,int], Set[str]]
        """
        return self.__label_to_actions_invertible.inv

    @property
    def actions_by_state(self):
        """Returns a mapping from states to sets of (available) actions.

        :return: The mapping.
        :rtype: Dict[int, Set[int]]
        """        
        if self.__available_actions is None:
            self.__available_actions = InvertibleDict({})
            for idx in range(self.C):
                state,action = self.index_by_state_action.inv[idx]
                self.__available_actions.add(state, action)
        return self.__available_actions

    def reachable_mask(self, from_set, mode, blocklist=set()):
        """Computes an :math:`N_{S_{\\text{all}}}`-dimensional vector which has a True-entry (False otherwise) for
        every state index that is reachable from 'from_set' in the given search mode (forward or backward).
        
        :param from_set: The set of states the search should start from.
        :type from_set: Set[int]
        :param mode: Either 'forward' or 'backward'. Defines the direction of search.
        :type mode: str
        :param blocklist: Set of states that should block any further search.
        :type blocklist: Set[int]
        :return: Resulting vector.
        :rtype: np.ndarray[bool]
        """
        return self.__graph.reachable(from_set, mode, blocklist)

    def predecessors(self, fromidx):
        """Yields an iterator that computes state-action-pairs (s,a) such that
        applying action a to state s yields the given state with some probability p > 0.
        
        :param fromidx: The given state.
        :type fromidx: int
        :yield: A state-action-pair (s,a)
        :rtype: Iterator[Tuple[int, int]]
        """       
        return self.__graph.predecessors(fromidx)

    def successors(self, fromidx):
        """Yields an iterator that computes state-action-pairs (d,a) where applying action a to
        the given state yields state d with some probability p > 0.
        
        :param fromidx: The given state.
        :type fromidx: int
        :yield: A state-action-pair (d,a)
        :rtype: Iterator[Tuple[int,int]]
        """        
        return self.__graph.successors(fromidx)

    @classmethod
    def from_file(cls, label_file_path, tra_file_path):
        """Computes an instance of this model from a given .lab and .tra file.
        
        :param label_file_path: Path of .lab-file.
        :type label_file_path: str
        :param tra_file_path: Path of .tra-file.
        :type tra_file_path: str
        :return: Instance of given class.
        :rtype: cls
        """        
        # identify all states
        t = time()
        states_by_label, _, _ = parse_label_file(label_file_path)
        # then load the transition matrix
        t = time()
        res = cls._load_transition_matrix(tra_file_path)
        return cls(**res, label_to_states=states_by_label)
        
    @classmethod
    def from_prism_model(cls, model_file_path, prism_constants = {}, extra_labels = {}):
        """Computes an instance of this model from a PRISM model.
        
        :param model_file_path: File path of .pm-model.
        :type model_file_path: str
        :param prism_constants: A dictionary of constants to be assigned in the model, defaults to {}.
        :type prism_constants: Dict[str,int], optional
        :param extra_labels: A dictionary that defines additional labels (other than the ones defined in the prism module) to 
            be added to the .lab file. The keys are label names and the values are PRISM expressions over the module variables, 
            defaults to {}.
        :type extra_labels: Dict[str,str], optional
        :return: Instance of the class this function is called from.
        :rtype: [This Class]
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
    def digraph(self, **kwargs):
        """Renders this instance as a graphviz.Digraph object.
        """        
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
