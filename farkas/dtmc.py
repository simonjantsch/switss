from scipy.sparse import dok_matrix
from graphviz import Digraph
from prism import parse_label_file, prism_to_tra
import re
import numpy as np
import tempfile
import os.path
import graphviz_utils

class DTMC:

    def __init__(self, P, states_by_label = None):
        """Initializes a new DTMC.
        
        :param P:  A (NxN) matrix that contains the transition probabilities
        :type P: scipy.sparse.dok_matrix
        :param states_by_label: A function that assigns labels (string) to sets of states, defaults to None
        :type states_by_label: Dict[str, Set[int]], optional
        """

        self.P = P
        self.N = P.shape[0]
        self.__states_by_label = states_by_label
        self.__labels_by_state = None

    @staticmethod
    def from_prism_model(model_file_path, prism_constants = {}, extra_labels = {}):
        """Initializes a new DTMC from a PRISM model
        
        :param model_file_path: file path of model without file type (e.g. tra or lab)
        :type model_file_path: str
        :param prism_constants: A dictionary of constants to be assigned in the model, defaults to {}
        :type prism_constants: Dict[str,int], optional
        :param extra_labels: A dictionary that defines additional labels (than the ones defined in the prism module) to be added to the .lab file. The keys are label names and the values are PRISM expressions over the module variables, defaults to {}
        :type extra_labels: Dict[str,str], optional
        :return: the DTMC
        :rtype: dtmc.DTMC
        """       
        with tempfile.TemporaryDirectory() as tempdirname:
            temp_model_file = os.path.join(tempdirname, "model")
            temp_tra_file = temp_model_file + ".tra"
            temp_lab_file = temp_model_file + ".lab"
            if prism_to_tra(model_file_path,temp_model_file,prism_constants,extra_labels):
                return DTMC.from_file(temp_lab_file,temp_tra_file)
            else:
                assert False, "Prism call to create model failed."

    @staticmethod
    def from_file(label_file_path, tra_file_path):
        """Creates a DTMC from a model specified by a set of labels and transitions.
        
        :param label_file_path: the file describing state labels (type is .lab)
        :type label_file_path: str
        :param tra_file_path: the file describing state transitions (type is .tra)
        :type tra_file_path: str
        :return: the resulting DTMC
        :rtype: dtmc.DTMC
        """              
        # identify all states
        states_by_label, _, _ = parse_label_file(label_file_path)
        # then load the transition matrix
        P = DTMC.__load_transition_matrix(tra_file_path)
        return DTMC(P, states_by_label)    

    def states_by_label(self):
        """A dictionary that yields a set of states for each label.
        
        :return: the dictionary
        :rtype: Dict[str, Set[int]]
        """        
        return self.__states_by_label

    def labels_by_state(self):
        """inverse of dtmc.states_by_label, i.e. gives a dictionary that specifies a set of labels for each state.
        
        :return: the dictionary
        :rtype: Dict[int, Set[str]]
        """                   
        if self.__labels_by_state is None:
            self.__labels_by_state = dict(enumerate([set({}) for i in range(self.N)]))
            for label, states in self.__states_by_label.items():
                for state in states:
                    self.__labels_by_state[state].add(label)
        return self.__labels_by_state

    @staticmethod
    def __load_transition_matrix(filepath):
        """Loads a transition matrix from a .tra-file.
        
        :param filepath: A .tra-file which contains the transition probabilities.
        :type filepath: str
        :return: a (NxN) transition matrix
        :rtype: scipy.sparse.dok_matrix
        """

        P = dok_matrix((1,1))
        N = 0

        with open(filepath) as tra_file:
            for line in tra_file:
                line_split = line.split()
                # check for first lines, which has format "#states #transitions"
                if len(line_split) == 2:
                    N = int(line_split[0])
                    P.resize((N,N))
                # all other lines have format "from to prob"
                else:
                    source = int(line_split[0])
                    dest = int(line_split[1])
                    prob = float(line_split[2])
                    P[source,dest] = prob
        return P

    def reduce(self, initial_label, targets_label):
        """Creates a reduced DTMC from a given transition matrix, an initial state (index) and a set of target states (indices).
        Will do a forwards (states which are reachable from the initial state) and
        backwards (states which are able to reach the target states) reachability test on the states of the given DTMC.
        Removes states which fail the forwards reachability test. Will also remap all target states to a single new
        target state and all fail states (states which fail the backwards reachability test) to one single fail state.
        
        :param initial_label: the label of the initial state, which must yield exactly one initial state
        :type initial_label: str
        :param targets_label: the label of the target states, which must yield at least one target state
        :type targets_label: str
        :return: the reduced DTMC
        :rtype: dtmc.ReducedDTMC
        """
        assert len(self.states_by_label()[targets_label]) > 0, "There needs to be at least one target state."
        target_states = self.states_by_label()[targets_label]
        # order doesn't matter since there should only be one initial state
        initial_state_count = len(self.states_by_label()[initial_label])
        assert initial_state_count == 1, "There were %d initial states given. Must be 1." % initial_state_count
        initial = list(self.states_by_label()[initial_label])[0]

        # computes states which are able to reach the target states
        reaching_target = DTMC.reachable(self.P, target_states, mode="backward")
        # if a state is not able to reach the target states, consider it as a fail-state
        P, to_fail = DTMC.__compute_fail_states(self.P, reaching_target)
        # compute all states which are reachable from the initial state
        reachable = DTMC.reachable(P, set([initial]), mode="forward")

        # there are target states which are only reachable through other target states.
        reachable, targets = DTMC.__remove_unneccessary_target_states(P, reachable, target_states)

        # remove states which are not reachable and recalculate initial state index
        P_red, to_target_red, full_to_red, red_to_full = DTMC.__restrict_to_reachable(
            P, initial, reachable, targets, to_fail)

        red_to_full[P_red.shape[0]] = "T"
        red_to_full[P_red.shape[0]+1] = "F"

        return ReducedDTMC(P_red, to_target_red, full_to_red[initial]), full_to_red, red_to_full

    def save(self, filepath):
        """Saves the .tra and .lab-file according to the given filepath
        
        :param filepath: the file path
        :type filepath: str
        :return: path of .tra and .lab-file
        :rtype: Tuple[str,str]
        """        
        tra_path = filepath + ".tra"
        lab_path = filepath + ".lab"

        with open(tra_path, "w") as tra_file:
            tra_file.write("%d %d\n" % (self.N, self.P.nnz))
            for (source,dest), p in self.P.items():
                if p > 0:
                    tra_file.write("%d %d %f\n" % (source, dest, p))

        with open(lab_path, "w") as lab_file:
            unique_labels_list = list(self.states_by_label().keys())
            header = ["%d=\"%s\"" % (i, label) for i,label in enumerate(unique_labels_list)]
            lab_file.write("%s\n" % (" ".join(header)))
            for idx, labels in self.labels_by_state().items():
                if len(labels) == 0:
                    continue
                labels_str = " ".join(map(str, map(unique_labels_list.index, labels)))
                lab_file.write("%d: %s\n" % (idx, labels_str))

        return tra_path, lab_path

    def graphviz_digraph(self, state_map = None, trans_map = None):
        """creates a graphviz.Digraph object from this DTMC.
        
        :param state_map: a function that is able to enable/disable states and computes a color and a label, defaults to None
        :type state_map: (stateidx, labels) -> {"color" : str, "label" : str, "enable" : bool} , optional
        :param trans_map: a function that is able to enable/disable state transitions and computes a color and a label, defaults to None
        :type trans_map: (sourceidx, destidx, sourcelabels, destlabels, p) -> {"color" : str, "label" : str, "enable" : bool}, optional
        :return: the Digraph
        :rtype: graphviz.Digraph
        """        

        def standard_state_map(stateidx, labels):
            return { "color" : graphviz_utils.color_from_hash(tuple(sorted(labels))),
                     "label" : "State %d\n%s" % (stateidx,",".join(labels)),
                     "enable" : True }

        def standard_trans_map(sourceidx, destidx, sourcelabels, destlabels, p):
            return { "color" : "black", "label" : str(round(p,10)), "enable" : True }

        state_map = standard_state_map if state_map is None else state_map
        trans_map = standard_trans_map if trans_map is None else trans_map

        dg = Digraph(node_attr = {"style" : "filled"})

        # connect nodes between each other
        existing_nodes = set({})

        for (source, dest), p in self.P.items():

            # transition from source to dest w/ probability p
            if p > 0:
                for node in [source, dest]:
                    if node not in existing_nodes:
                        # print(self.labels[node])
                        state_setting = state_map(node, self.labels_by_state()[node])
                        if state_setting["enable"]:
                            dg.node(str(node),
                                    label=state_setting["label"],
                                    color=state_setting["color"])
                        existing_nodes.add(node)

                params = (source, dest, self.labels_by_state()[source], self.labels_by_state()[dest], p)
                trans_setting = trans_map(*params)
                if trans_setting["enable"]:
                    dg.edge(str(source), str(dest),
                            label=trans_setting["label"],
                            color=trans_setting["color"])

        return dg

    @staticmethod
    def __compute_fail_states(P, reaching_target):
        """Computes a vector (to_fail) which contains the probability of reaching the fail state in one step for each state.
        
        :param P: A (NxN) transition matrix
        :type P: scipy.sparse.dok_matrix
        :param reaching_target: set of states that are not able to reach the target state. All states which are not able to reach
            the target state are considered as fail states.
        :type reaching_target: Set[int]
        :return: the resulting transition matrix if all states which are not able to reach the target states are removed
            and the "to_fail" vector. 
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray]
        """        

        N = P.shape[0]
        to_fail = np.zeros(N)
        P_tmp = dok_matrix((N,N))

        for (source,dest) in P.keys():
            if source in reaching_target:
                if dest in reaching_target:
                    P_tmp[source,dest] = P[source,dest]
                else:
                    to_fail[source] += P[source,dest]

        return P_tmp, to_fail


    @staticmethod
    def reachable(P, initial, mode):
        """returns the set of all states that are forward or backwards reachable from a set of states
        
        :param P: the transition matrix
        :type P: scipy.sparse.dok_matrix
        :param initial: initial set of states for the reachability check
        :type initial: Set[int]
        :param mode: either "forward" or "backward"; the reachability mode
        :type mode: str
        :return: the set of states that are reachable
        :rtype: Set[int]
        """
        assert mode in ["forward", "backward"], "mode must be either 'forward' or 'backward' but is '%s'." % mode

        reachable_states = initial.copy()
        while True:
            current_size = len(reachable_states)
            for (source,dest), p in P.items():
                if p > 0:
                    fr,to = (source,dest) if mode == "forward" else (dest,source)
                    if fr in reachable_states:
                        reachable_states.add(to)
            if len(reachable_states) == current_size:
                break
        return reachable_states

    @staticmethod
    def __remove_unneccessary_target_states(P, reachable, target_states):
        """Removes all target states from "reachable" and "target_states" if they are only reachable through other target states.
        They are unneccessary because they will be unreachable from the initial state after
        every target state is remapped to a single new target state. If not removed, the size of P would increase.
        
        :param P: A (NxN) transition matrix.
        :type P: scipy.sparse.dok_matrix
        :param reachable: set of reachable states
        :type reachable: Set[int]
        :param target_states: set of all target states
        :type target_states: Set[int]
        :return: a new set containing only states which are reachable and will not be removed,
            and a new set containing only target states which will not be removed.
        :rtype: Tuple[Set[int],Set[int]]
        """
        _reachable = set({})

        # only keep target states that are reachable from at least one state which is not a target state
        for state in reachable:
            if state not in target_states:
                _reachable.add(state)
            else:
                for (source, _), p in P[:, state].items():
                    # there is a transition from a state which is not a target state to this target state
                    if source not in target_states:
                        _reachable.add(state)
                        break
        _target_states = _reachable.intersection(target_states)

        return _reachable, _target_states

    @staticmethod
    def __restrict_to_reachable(P_old, initial_old, reachable, target_states, to_fail_full):
        """Creates a new transition matrix (P) from another transition matrix (P_old) by
        removing all unreachable states. All target states will be remapped to a single
        new target state with probability 1; all the other states will lead to the target
        state with probability 0.
        
        :param P_old: old transition matrix
        :type P_old: scipy.sparse.dok_matrix
        :param initial_old: index of initial state in the old transition matrix
        :type initial_old: int
        :param reachable: set of states which are forwards and backwards reachable
        :type reachable: Set[int]
        :param target_states: set of target states
        :type target_states: Set[int]
        :param to_fail_full: transition probability of reaching a fail state in one step
        :type to_fail_full: np.ndarray
        :return: a new transition matrix, a vector containing transition probabilities from states to target states in one step,
        a dictionary that maps from P-states to P_old-states, a dictionary that maps from P_old-states to P-states
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray, Dict[int,int], Dict[int,int]]
        """
        reachable_but_not_target = reachable.difference(target_states)

        N = len(reachable)

        to_target = np.zeros(N)
        to_fail_reduced = np.zeros(N)

        P = dok_matrix((N,N))
        full_to_reduced, reduced_to_full = {}, {}

        for iterator, old in enumerate(reachable):
            full_to_reduced[old] = iterator
            to_fail_reduced[iterator] = to_fail_full[old]
            reduced_to_full[iterator] = old

        for state in target_states:
            to_target[full_to_reduced[state]] = 1

        for (source,dest), p in P_old.items():
            if (source in reachable_but_not_target) and (dest in reachable):
                P[full_to_reduced[source], full_to_reduced[dest]] = p

        return P, to_target, full_to_reduced, reduced_to_full


class ReducedDTMC:

    def __init__(self, P, to_target, initial):     
        """Creates a DTMC from a given transition matrix (not containing reachability probabilities from or to the target state),
        a vector containing the probabilties of reaching the target state in one step (to_target= :math:`b`) and a given initial state index.
        This is equivalent to giving the DTMC in the form :math:`x = P x + b`. 
        
        :param P: (NxN) transition matrix
        :type P: scipy.sparse.dok_matrix
        :param to_target: probability of reaching the target state in one step
        :type to_target: np.ndarray
        :param initial: index of the initial state
        :type initial: int
        """
        self.P = P
        self.to_target = to_target
        self.initial = initial
        self.N = P.shape[0]

    def as_dtmc(self):
        """creates a DTMC from this ReducedDTMC.
        
        :return: the DTMC
        :rtype: dtmc.DTMC
        """        
        P_compl, target_state, fail_state = self.__transition_matrix()
        labels = { "init" : set({self.initial}), "target" : set({target_state}), "fail" : set({fail_state}) }
        return DTMC(P_compl, labels)

    def __transition_matrix(self):
        """Computes the transition matrix (including fail and target state)
        
        :return: the transition matrix, index of the target state, index of the fail state
        :rtype: Tuple[scipy.sparse.dok_matrix, int, int]
        """
        # creates a transition matrix which includes the target and fail state
        P_compl = dok_matrix((self.N+2, self.N+2))
        target_state = self.N
        fail_state = self.N+1

        not_to_fail = np.zeros(self.N)
        for (source, dest), p in self.P.items():
            if p > 0:
                not_to_fail[source] += p
                P_compl[source, dest] = p

        for idx, p_target in enumerate(self.to_target):
            if p_target > 0:
                P_compl[idx, target_state] = p_target
            p_fail = 1 - (p_target + not_to_fail[idx])
            if p_fail > 0:
                P_compl[idx, fail_state] = p_fail

        P_compl[fail_state, fail_state] = 1
        P_compl[target_state, target_state] = 1

        return P_compl, target_state, fail_state
