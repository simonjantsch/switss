from __future__ import annotations
import re
from typing import Set, Dict, Tuple, Callable
from scipy.sparse import dok_matrix
import numpy as np
from graphviz import Digraph
 

class ReducedDTMC:
    
    def __init__(self, P : dok_matrix, to_target : np.ndarray, initial : int):
        '''
        Creates a DTMC from a given transition matrix (not containing reachability probabilities from or to the target state), 
        a vector containing the probabilities of reaching the target state in one step (to_target=b) and a given initial state index.

        This is equivalent to giving the DTMC in the form 
        
            $$ x = P x + b $$ 
        
        where solving the linear equation system for x would yield the probability of eventually reaching the
        target state for each state x1,...xN.

        Parameters
        ----------
        P : dok_matrix
            Transition matrix in the form (source, destination).
        to_target : np.ndarray
            Probability of reaching the target state in one step.
        initial : int
            Index of the initial state.

        '''
        self.P = P
        self.to_target = to_target
        self.initial = initial
        self.N = P.shape[0]
    
    @staticmethod
    def from_dtmc(P : dok_matrix, targets : Set[int], initial : int):
        '''
        Creates a reduced DTMC from a given transition matrix, an initial state (index) and a set of target states (indices).
        Will do a forwards (states which are reachable from the initial state) and 
        backwards (states which are able to reach the target states) reachability test on the states of the given DTMC. 
        Removes states which fail the forwards reachability test. Will also remap all target states to a single new
        target state and all fail states (states which fail the backwards reachability test) to one single fail state.

        Parameters
        ----------
        P : dok_matrix
            The transition matrix in the form (source, destination).
        initial : int
            Index of the initial state.
        targets : Set[int]
            Set of indices of the target states.

        '''
        
        # computes states which are able to reach the target states
        reaching_target = ReducedDTMC.__reachable(P, targets, mode="backward")
        # if a state is not able to reach the target states, consider it as a fail-state
        P, to_fail = ReducedDTMC.__compute_fail_states(P, P.shape[0], reaching_target)
        # compute all states which are reachable from the initial state
        reachable = ReducedDTMC.__reachable(P, set([initial]), mode="forward")
        
        # there are target states which are only reachable through other target states.
        reachable, targets = ReducedDTMC.__remove_unneccessary_target_states(P, reachable, targets)
        
        # remove states which are not reachable and recalculate initial state index
        P_red, to_target_red, full_to_red, red_to_full = ReducedDTMC.__restrict_to_reachable(
            P, initial, reachable, targets, to_fail)
        
        red_to_full[P_red.shape[0]] = "T"
        red_to_full[P_red.shape[0]+1] = "F"
        
        return ReducedDTMC(P_red, to_target_red, full_to_red[initial]), red_to_full
        
    @staticmethod
    def __compute_fail_states(P : dok_matrix, N : int, reaching_target : Set[int]) -> Tuple[dok_matrix, Set[int], np.ndarray]:    
        '''
        Computes a vector (to_fail) which contains the probability of reaching the fail state in one step for each state.
        
        P : dok_matrix
            Transition matrix in the form (source, destination).
        N : int
            Size of the transition matrix.
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
        P_tmp = dok_matrix((N,N))
        
        for (source,dest) in P.keys():
            if source in reaching_target:
                if dest in reaching_target:
                    P_tmp[source,dest] = P[source,dest]
                else:
                    to_fail[source] += P[source,dest]
        
        return P_tmp, to_fail
    
    
    @staticmethod
    def __reachable(P : dok_matrix, initial : Set[int], mode : str) -> Set[int]:
        '''
    
        Parameters
        ----------
        P : dok_matrix
            Transition matrix in the form (source, destination).
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
            for (source,dest), p in P.items():
                if p > 0:
                    fr,to = (source,dest) if mode == "forward" else (dest,source)
                    if fr in reachable_states:
                        reachable_states.add(to)
            if len(reachable_states) == current_size:
                break
        return reachable_states
    
    @staticmethod
    def __remove_unneccessary_target_states(P : dok_matrix, reachable : Set[int], target_states : Set[int]) -> Tuple[Set[int], Set[int]]:
        '''
        Removes all target states from "reachable" and "target_states" if they are only reachable through other target states.
        They are unneccessary because they will be unreachable from the initial state after 
        every target state is remapped to a single new target state. If not removed, the size of P would increase.
        
        Parameters
        ----------
        P : dok_matrix
            Transition matrix in the form (source, destination).
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
                for (source, _), p in P[:, state].items():
                    # there is a transition from a state which is not a target state to this target state
                    if source not in target_states:
                        _reachable.add(state)
                        break
        _target_states = _reachable.intersection(target_states)
        
        return _reachable, _target_states
        
    @staticmethod
    def __restrict_to_reachable(P_old : dok_matrix, 
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
        to_target : np.ndarray
            Probability of reaching the target state in one step.
        full_to_reduced : Dict[int, int]
            State index mapping from full form to reduced form.
        reduced_to_full : Dict[int, int]
            State index mapping from reduced form to full form.

        '''                    
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
    
    @staticmethod
    def __load_transition_matrix(filepath : str) -> dok_matrix:
        '''
        Loads a transition matrix from a .tra-file.
        
        Parameters
        ----------
        filepath : str
            A .tra-file which contains the transition probabilities.
            
        Returns
        -------
        dok_matrix
            transition matrix in the form (source, destination)
    
        '''
        
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
    
    
    
    @staticmethod
    def __load_states(filepath : str) -> Dict[str, Set[int]]:
        '''
        Loads a set of states (by label) from a .lab file.
    
        Parameters
        ----------
        filepath : str
            A .lab-file which contains the state labels.
    
        Returns
        -------
        Dict[str, Set[int]]
            A mapping from labels to state indices.
    
        '''
        
        labelid_to_label = {}
        state_dict = {}
        mark_regexp = re.compile(r"([0-9]+)=\"(.*?)\"")
        line_regexp = re.compile(r"([0-9]+):([\W,0-9]+)")
        with open(filepath) as label_file:
            lines = label_file.readlines()
            regexp_res = re.finditer(mark_regexp, lines[0])
            for _, match in enumerate(regexp_res, start=1):
                labelid, label = int(match.group(1)), match.group(2)
                labelid_to_label[labelid] = label
                state_dict[label] = set({})
            
            for line in lines[1:]:
                regexp_res = line_regexp.search(line)
                state_labelids = map(int, regexp_res.group(2).split())
                for labelid in state_labelids:
                    label = labelid_to_label[labelid]
                    state_dict[label].add(int(regexp_res.group(1)))
        
        return state_dict
    

    @staticmethod
    def from_file(label_file_path : str, tra_file_path : str, initial : str, target : str) -> ReducedDTMC:
        '''
        Loads a DTMC from a given .lab and a given .tra-description. 
    
        Parameters
        ----------
        label_file_path : str
            File path of the .lab-description of the DTMC.
        tra_file_path : str
            File path of the .tra-description of the DTMC.
        initial : str
            Label of the initial state (must be exactly one).
        target : str
            Label of the target states (must be more than one).

        Returns
        -------
        DTMC
            The respective DTMC.

        '''
        
        # identify all states
        all_states = ReducedDTMC.__load_states(label_file_path)
        
        # identify all target states
        target_states = all_states[target]
        assert len(target_states) > 0, "0 target states identified."
        
        # identify the initial state
        initial_states = all_states[initial]
        assert len(initial_states) == 1, "found %d initial state(s). but exactly 1 is required." % len(initial_states)
        init = initial_states.pop()
        
        # then load the transition matrix
        P = ReducedDTMC.__load_transition_matrix(tra_file_path)
        
        return ReducedDTMC.from_dtmc(P, target_states, init)
    
    def transition_matrix(self) -> Tuple[dok_matrix, int, int]:
        '''
        Computes the transition matrix (including fail and target state).

        Returns
        -------
        Tuple[dok_matrix, int, int]
            (transition matrix, index of target state, index of fail state).

        '''
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
    
    def graphviz_digraph(self,  
        state_color_map : Callable[[int, bool, bool, bool], str] = lambda stateidx, is_initial, is_target, is_fail: "green" if is_target else "blue" if is_initial else "orange" if is_fail else "lightblue",
        state_label_map : Callable[[int, bool, bool, bool], str] = lambda stateidx, is_initial, is_target, is_fail: "Target [%s]" % stateidx if is_target else "Fail [%s]" % stateidx if is_fail else "Initial [%s]" % stateidx if is_initial else "State [%s]" % stateidx,
        trans_color_map : Callable[[int, int, float], str] = lambda sourceidx, destidx, p: "black",
        trans_label_map : Callable[[int, int, float], str] = lambda sourceidx, destidx, p: str(round(p,15)),
        target_state_is_separate : bool = True,
        include_fail_state: bool = True,
        ) -> Digraph:
        '''

        Parameters
        ----------
        state_color_map : Callable[[int, bool, bool, bool], str], optional
            * Maps each tuple (stateidx, is_initial, is_target, is_fail) to a color (in the graph).
            * is_initial is True iff the state is the initial state.
            * is_target is True iff the state is a (or the) target state.
            * is_fail is True iff the state is the fail state.
        state_label_map : Callable[[int, bool, bool, bool], str], optional
            * Maps each tuple (stateidx, is_initial, is_target, is_fail) to a label (in the graph).
            * is_initial is True iff the state is the initial state.
            * is_target is True iff the state is a (or the) target state.
            * is_fail is True iff the state is the fail state.
        trans_color_map : Callable[[int, int, float], str], optional
            * Maps each transition (sourceidx, destinationidx, p) to a color.
        trans_label_map : Callable[[int, int, float], str], optional
            * Maps each transition (sourceidx, destinationidx, p) to a label.
        target_state_is_separate : bool, optional
            * Creates a single target state if set to True. 
            * Otherwise, do not include a single target state but set "is_target" to True in the state_label_map and state_color_map if the probability of reaching the target state is 1.
            * Can be useful if many states are connected to the target state. 
            * The default is True.
        include_fail_state : bool, optional
            * Removes the fail state from the digraph if set to False. 
            * Can be useful if many states are connected to the fail state. 
            * The default is True.

        Returns
        -------
        Digraph
            Creates a graphviz-Digraph representation of the DTMC. 
            For an overview on how to use graphviz and especially Digraphs,
            see https://graphviz.readthedocs.io/en/stable/manual.html

        '''
        
        dg = Digraph(node_attr = {"style" : "filled"})
        
        # connect nodes between each other
        existing_nodes = set({})
        P_compl, target_state, fail_state = self.transition_matrix()
        
        for (source, dest), p in P_compl.items():
            
            # transition from source to dest w/ probability p
            if p > 0:
                for node in [source, dest]:
                    if node not in existing_nodes:  
                        is_initial = self.initial == node
                        is_fail = fail_state == node
                        is_single_target = target_state == node 
                        is_target = is_single_target or (P_compl[node, target_state] == 1 and not target_state_is_separate)
                        params = (node, is_initial, is_target, is_fail) 
                        # can be read as "node is the single target state => target state is separate"
                        # and "node is the fail state => include fail state"
                        if (not is_single_target or target_state_is_separate) and (not is_fail or include_fail_state):
                            dg.node(str(node), label=state_label_map(*params), color=state_color_map(*params))   
                        # add a self-edge if neccessary 
                        if is_target and not is_single_target and not target_state_is_separate:
                            dg.edge(str(node), str(node), label=trans_label_map(node, node, 1), color=trans_color_map(node, node, 1))                                                                                                                  
                            
                        existing_nodes.add(node)  
                
                dest_is_fail = fail_state == dest 
                dest_is_single_target = target_state == dest
                # can be read as "destination is the single target state => target state is separate"
                # and "destination is the fail state => include fail state"
                if (not dest_is_single_target or target_state_is_separate) and (not dest_is_fail or include_fail_state):        
                    dg.edge(str(source), str(dest), label=trans_label_map(source, dest, p), color=trans_color_map(source, dest, p))
                
        return dg
            
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
        
        P_compl, target_state, fail_state = self.transition_matrix()
        
        with open(tra_path, "w") as tra_file:
            tra_file.write("%d %d\n" % (P_compl.shape[0], P_compl.nnz))
            for (source,dest), p in P_compl.items():
                if dest == target_state:
                    tra_file.write("%d %d %f\n" % (source, source, 1))
                elif p > 0:
                    tra_file.write("%d %d %f\n" % (source, dest, p))
        with open(lab_path, "w") as lab_file:
            lab_file.write("0=\"init\" 1=\"target\" 2=\"fail\"\n")
            lab_file.write("%d: 0\n" % self.initial)
            lab_file.write("%d: 2\n" % fail_state)
            for (source, _), p in P_compl[:,target_state].items():
                if source != target_state:
                    lab_file.write("%d: 1\n" % source)
        
        return tra_path, lab_path
