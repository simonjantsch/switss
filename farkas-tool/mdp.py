from __future__ import annotations
from bidict import bidict
from graphviz import Digraph
from scipy.sparse import dok_matrix
from typing import Set, Dict, Tuple, Callable
import numpy as np
import re
import prism as prism

class MDP:
    def __init__(self,
                 P : dok_matrix,
                 index_by_state_action: bidict,
                 states_by_label : Dict[str, Set[int]] = None):
        '''
        Initialises an MDP.

        Parameters
        ----------
        P : dok_matrik
            A (C x N) matrix where N is the number of states of the MDP, and C is the number of active state-action pairs.
            A row gives the probabilistic distribution over successor states for some state-action pair.

        index_by_state_actions : bidict
            A bidirectional dictionary that maps a state-action pair to its index in the matrix P.

        states_by_label : Dict[str, Set[int]]
            A dictionary that defines labels over the MDP.
            Each label is mapped to a subset of the states.

        '''
        (C,N) = P.shape
        assert len(index_by_state_action.values()) == C, "The number of state-action pairs must correspond to the number of rows in the matrix."
        assert len(set([x for x,_ in index_by_state_action.keys()])) == N, "Each state needs participate in some state-action pair."

        self.P = P
        self.__index_by_state_action = index_by_state_action
        self.N = N
        self.C = C
        self.__states_by_label = states_by_label
        self.__labels_by_state = None


    @staticmethod
    def __get_state(index_by_state_action : bidict, index : int) -> int:
        return index_by_state_action.inverse[index][0]

    @staticmethod
    def __get_action(index_by_state_action : bidict, index : int) -> int:
        return index_by_state_action.inverse[index][1]

    @staticmethod
    def __get_state_action(index_by_state_action : bidict, index : int) -> int:
        return index_by_state_action.inverse[index]

    @staticmethod
    def from_file(label_file_path : str, tra_file_path : str) -> MDP:
        # identify all states
        states_by_label, labels_by_state, labelid_to_label = prism.parse_label_file(label_file_path)
        # then load the transition matrix
        index_by_state_action,P = MDP.__load_transition_matrix(tra_file_path)
        return MDP(P, index_by_state_action,states_by_label)

    def states_by_label(self):
        return self.__states_by_label

    def labels_by_state(self):
        if self.__labels_by_state is None:
            self.__labels_by_state = dict(enumerate([set({}) for i in range(self.N)]))
            for label, states in self.__states_by_label.items():
                for state in states:
                    self.__labels_by_state[state].add(label)
        return self.__labels_by_state

    @staticmethod
    def __load_transition_matrix(tra_file_path : str) -> Tuple[bidict,dok_matrix]:
        '''
        Loads a transition matrix from a .tra-file.

        Parameters
        ----------
        filepath : str
            A .tra-file which contains the transition probabilities.

        Returns
        -------
        bidict :
            A bidirectional dictionary that maps state-action pairs into an index set {0,..,C}.

        dok_matrix
            transition matrix in the form (index, destination)
        '''

        P = dok_matrix((1,1))
        index_by_state_action = bidict()

        with open(tra_file_path,"r") as tra_file:
            # the first line should have format "#states #choices #transitions"
            # the number of choices is the number of active state-action pairs
            first_line_split = tra_file.readline().split()
            N = int(first_line_split[0])
            C = int(first_line_split[1])
            P.resize((C,N))

            max_index = 0
            for line in tra_file.readlines():
                # all other lines have format "source action dest prob"
                line_split = line.split()
                source = int(line_split[0])
                action = int(line_split[1])
                dest = int(line_split[2])
                prob = float(line_split[3])

                if (source,action) in index_by_state_action:
                    index = index_by_state_action[(source,action)]
                else:
                    index = max_index
                    index_by_state_action[(source,action)] = max_index
                    max_index += 1
                P[index,dest] = prob

        return index_by_state_action,P

    def reduce(self, initial_label : str, targets_label : str):
        '''
        Creates a reduced MDP from a given transition matrix, an initial state (index) and a set of target states (indices).
        Will do a forwards (states which are reachable from the initial state) and
        backwards (states which are able to reach the target states) reachability test on the states of the given MDP.
        Removes states which fail the forwards reachability test. Will also remap all target states to a single new
        target state and all fail states (states which fail the backwards reachability test) to one single fail state.

        Parameters
        ----------
        P : dok_matrix
            The transition matrix in the form (index, destination).
        initial : int
            Index of the initial state.
        targets : Set[int]
            Set of indices of the target states.

        '''
        assert len(self.states_by_label()[targets_label]) > 0, "There needs to be at least one target state."
        target_states = self.states_by_label()[targets_label]
        # order doesn't matter since there should only be one initial state
        initial_state_count = len(self.states_by_label()[initial_label])
        assert initial_state_count == 1, "There were %d initial states given. Must be 1." % initial_state_count
        initial = list(self.states_by_label()[initial_label])[0]

        # computes states which are able to reach the target states
        reaching_target = MDP.reachable(self.P, self.__index_by_state_action, target_states, mode="backward")
        # if a state is not able to reach the target states, consider it as a fail-state
        P, to_fail = MDP.__compute_fail_states(self.P,
                                               self.__index_by_state_action,
                                               self.C,
                                               self.N,
                                               reaching_target)
        # compute all states which are reachable from the initial state
        reachable = MDP.reachable(P, self.__index_by_state_action, set([initial]), mode="forward")

        # there are target states which are only reachable through other target states.
        reachable, targets = MDP.__remove_unneccessary_target_states(P, self.__index_by_state_action, reachable, target_states)

        # remove states which are not reachable and recalculate initial state index
        P_red, index_to_state_action_red, to_target_red, full_to_red, red_to_full = MDP.__restrict_to_reachable(
            P,
            self.__index_by_state_action,
            initial,
            reachable,
            targets,
            to_fail)

        red_to_full[P_red.shape[1]] = "T"
        red_to_full[P_red.shape[1]+1] = "F"

        return ReducedMDP(P_red, index_to_state_action_red, to_target_red, full_to_red[initial]), full_to_red, red_to_full

    def save(self, filepath : str) -> Tuple[str, str]:
        '''
        Saves the .tra and .lab-file according to the given filepath.

        Parameters
        ----------
        filepath : str
            The file path.

        Returns
        -------
        Tuple[str, str]
            (path of .tra-file, path of .lab-file).

        '''
        tra_path = filepath + ".tra"
        lab_path = filepath + ".lab"

        with open(tra_path, "w") as tra_file:
            tra_file.write("%d %d %d\n" % (self.N, self.C, self.P.nnz))
            for (index,dest), p in self.P.items():
                if p > 0:
                    source,action = MDP.__get_state_action(self.__index_by_state_action,index)
                    tra_file.write("%d %d %d %f\n" % (source, action, dest, p))

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

    @staticmethod
    def __compute_fail_states(P : dok_matrix,
                              index_by_state_action : bidict,
                              C : int,
                              N : int,
                              reaching_target : Set[int]) -> Tuple[dok_matrix, Set[int], np.ndarray]:
        '''
        Computes a vector (to_fail) which contains the probability of reaching the fail state in one step for each state.

        P : dok_matrix
            Transition matrix in the form (index, destination).
        index_by_state_action : bidict
            Mapping from state-action pairs to an index set {0,..,C}
        C : int
            Number of rows of the transition matrix.
        N : int
            Number of colums of the transition matrix.
        reaching_target : Set[int]
            These are the states which are not able to reach the target state.
            All states which are not in "reaching_target" are considered as fail states.

        Returns
        -------
        P_tmp : dok_matrix
            The resulting transition matrix if all states which are not able to reach the target state are removed.
        to_fail : np.ndarray
            i-th index contains the probability of reaching a state which is not in
            "reaching_target" in one step from the i-th state.

        '''

        to_fail = np.zeros(N)
        P_tmp = dok_matrix((C,N))

        for (index,dest) in P.keys():
            source,action = MDP.__get_state_action(index_by_state_action,index)
            if source in reaching_target:
                if dest in reaching_target:
                    P_tmp[index,dest] = P[index,dest]
                else:
                    to_fail[index] += P[index,dest]

        return P_tmp, to_fail

    @staticmethod
    def reachable(P : dok_matrix, index_by_state_action : bidict, initial : Set[int], mode : str) -> Set[int]:
        '''

        Parameters
        ----------
        P : dok_matrix
            Transition matrix in the form (source, destination).
        index_by_state_action : bidict
            Mapping from state-action pairs to an index set {0,..,C}
        initial : Set[int]
            Initial set of states.
        mode : str
            Must be either "forward" or "backward".

        Returns
        -------
        Set[int]
            A set of all forward or backward-reachable states.

        '''
        assert mode in ["forward", "backward"], "mode must be either 'forward' or 'backward' but is '%s'." % mode

        reachable_states = initial.copy()
        while True:
            current_size = len(reachable_states)
            for (index,dest), p in P.items():
                if p > 0:
                    source = MDP.__get_state(index_by_state_action,index)
                    fr,to = (source,dest) if mode == "forward" else (dest,source)
                    if fr in reachable_states:
                        reachable_states.add(to)
            if len(reachable_states) == current_size:
                break
        return reachable_states

    @staticmethod
    def __remove_unneccessary_target_states(P : dok_matrix,
                                            index_by_state_action : bidict,
                                            reachable : Set[int],
                                            target_states : Set[int]) -> Tuple[Set[int], Set[int]]:
        '''
        Removes all target states from "reachable" and "target_states" if they are only reachable through other target states.
        They are unneccessary because they will be unreachable from the initial state after
        every target state is remapped to a single new target state. If not removed, the size of P would increase.

        Parameters
        ----------
        P : dok_matrix
            Transition matrix in the form (index, destination).
        index_by_state_action : bidict
            Mapping from state-action pairs to an index set {0,..,C}
        reachable : Set[int]
            Set of reachable states.
        target_states : Set[int]
            Set of all target states.

        Returns
        -------
        Tuple[Set[int], Set[int]]
            (new reachable, new target states).

        '''
        _reachable = set({})

        # only keep target states that are reachable from at least one state which is not a target state
        for state in reachable:
            if state not in target_states:
                _reachable.add(state)
            else:
                for (index, _), p in P[:, state].items():
                    # there is a transition from a state which is not a target state to this target state
                    if MDP.__get_state(index_by_state_action,index) not in target_states:
                        _reachable.add(state)
                        break
        _target_states = _reachable.intersection(target_states)

        return _reachable, _target_states

    @staticmethod
    def __restrict_to_reachable(P_old : dok_matrix,
                                index_by_state_action_old : bidict,
                                initial_old : int,
                                reachable : Set[int],
                                target_states : Set[int],
                                to_fail_full : np.ndarray) -> Tuple[dok_matrix, np.ndarray, Dict[int, int], Dict[int, int]]:
        '''
        Creates a new transition matrix (P) from another transition matrix (P_old) by
        removing all unreachable states. All target states will be remapped to a single
        new target state with probability 1; all the other states will lead to the target
        state with probability 0.

        Parameters
        ----------
        P_old : dok_matrix
            Transition matrix before removal of unreachable states.
        index_by_state_action_old : bidict
            Mapping from state-action pairs to an index set {0,..,C}
        initial_old : int
            Index of the initial state.
        reachable : Set[int]
            Set of states which are forwards and backwards reachable.
        target_states : Set[int]
            Set of target states.
        to_fail_full : np.ndarray
            Transition probability of reaching a fail state in one step.

        Returns
        -------
        P : dok_matrix
            Transition matrix after removal of unreachable states.
        index_by_state_action : bidict
            Mapping from state-action pairs to an index set {0,..,C}
        to_target : np.ndarray
            Probability of reaching the target state in one step.
        full_to_reduced : Dict[int, int]
            State index mapping from full form to reduced form.
        reduced_to_full : Dict[int, int]
            State index mapping from reduced form to full form.

        '''

        N = len(reachable)
        C = len(set([(x,y) for (x,y) in index_by_state_action_old.keys() if x in reachable]))

        to_target = np.zeros(C)
        to_fail_reduced = np.zeros(N)

        P = dok_matrix((C,N))
        full_to_reduced, reduced_to_full = {}, {}
        index_by_state_action = bidict()

        for iterator, old in enumerate(reachable):
            full_to_reduced[old] = iterator
            to_fail_reduced[iterator] = to_fail_full[old]
            reduced_to_full[iterator] = old

        i = 0
        for (index,dest), p in P_old.items():
            old_state = MDP.__get_state(index_by_state_action_old,index)
            action = MDP.__get_action(index_by_state_action_old,index)
            if (old_state in reachable):
                new_state = full_to_reduced[old_state]
                if (new_state,action) in index_by_state_action:
                    index = index_by_state_action[(new_state,action)]
                else:
                    index = i
                    index_by_state_action[new_state,action] = index
                    i += 1
                if (old_state in target_states):
                    to_target[index] = 1
                elif (dest in reachable):
                    P[new_state, full_to_reduced[dest]] = p

        return P, index_by_state_action, to_target, full_to_reduced, reduced_to_full


class ReducedMDP:

    def __init__(self, P : dok_matrix, index_by_state_action : bidict, to_target : np.ndarray, initial : int):
        '''
        Creates a MDP from a given transition matrix (not containing reachability probabilities from or to the target state),
        a vector containing the probabilities of reaching the target state in one step (to_target=b) and a given initial state index.

        P and b are exactly as when computing min/max reachability probabilities for MDP using the LPs

            $$ max c x. \;\; (I-P) x \leq b $$
            $$ min c x. \;\; (I-P) x \geq b $$

        Parameters
        ----------
        P : dok_matrix
            Transition matrix in the form (index, destination).
        index_by_state_action : bidict
            Mapping from state-action pairs to an index set {0,..,C}
        to_target : np.ndarray
            Probability of reaching the target state in one step.
        initial : int
            Index of the initial state.

        '''
        self.P = P
        self.to_target = to_target
        self.index_by_state_action = index_by_state_action
        self.initial = initial
        self.C = P.shape[0]
        self.N = P.shape[1]

    def as_mdp(self) -> DTMC:
        P_compl, index_by_state_action_compl, target_state, fail_state = self.__transition_matrix()
        labels = { "init" : set({self.initial}), "target" : set({target_state}), "fail" : set({fail_state}) }
        return MDP(P_compl, labels)

    def __transition_matrix(self) -> Tuple[dok_matrix, int, int]:
        '''
        Computes the transition matrix (including fail and target state).

        Returns
        -------
        The transition matrix and index_to_state_actions bidict of the corresponding complete MDP with a single target and fail state.

        '''
        # creates a transition matrix which includes the target and fail state
        P_compl = dok_matrix((self.C+2, self.N+2))
        target_state = self.N
        fail_state = self.N+1
        index_by_state_action_compl = self.index_by_state_action.copy()
        index_by_state_action_compl[(target_state,0)] = C
        index_by_state_action_compl[(fail_state,0)] = C+1

        not_to_fail = np.zeros(self.N)
        for (index, dest), p in self.P.items():
            if p > 0:
                not_to_fail[index] += p
                P_compl[index, dest] = p

        for idx, p_target in enumerate(self.to_target):
            if p_target > 0:
                P_compl[idx, target_state] = p_target
            p_fail = 1 - (p_target + not_to_fail[idx])
            if p_fail > 0:
                P_compl[idx, fail_state] = p_fail

        P_compl[index_by_state_action_compl[(fail_state,0)], fail_state] = 1
        P_compl[index_by_state_action_compl[(target_state,0)], target_state] = 1

        return P_compl, index_by_state_action_compl, target_state, fail_state
