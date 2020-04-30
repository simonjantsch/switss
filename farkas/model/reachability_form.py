from farkas.utils import cast_dok_matrix
from . import AbstractMDP
from ..utils import InvertibleDict
from ..solver.milp import LP

from collections import defaultdict
from bidict import bidict
import numpy as np
from scipy.sparse import dok_matrix,hstack,vstack

class ReachabilityForm:
    """ A reachability form is an MDP with a dedicated initial and target state. 
    It is represented by its transition matrix, an initial state and a vector that holds 
    the probability to move to the target state in one step for each state. 
    The target state is not included in the transition matrix. """
    
    def __init__(self, system, initial_label, target_label, debug=False):
        assert isinstance(system, AbstractMDP)

        assert len(system.states_by_label[target_label]) > 0, "There needs to be at least one target state."
        target_states = system.states_by_label[target_label]
        initial_state_count = len(system.states_by_label[initial_label])
        assert initial_state_count == 1, "There were %d initial states given. Must be 1." % initial_state_count
        initial = list(system.states_by_label[initial_label])[0]
        
        if debug:
            print("calculating reachable mask (backward)...")
        backward_reachable = system.reachable_mask(target_states, "backward")
        if debug:
            print("calculating reachable mask (forward)...")
        forward_reachable = system.reachable_mask(set([initial]), "forward")
        # states which are reachable from the initial state AND are able to reach target states
        reachable_mask = backward_reachable & forward_reachable
        # TODO: use reachable_mask instead of reachable everywhere
        # this is much better for performance since lookup of states is always O(1)
        reachable = { idx for idx,x in enumerate(reachable_mask) if x }

        if debug:
            print("tested backward & forward reachability test")

        def reachable_from_non_target_states(ts):
            # if ts is a target state, then
            # ts reachable from non-target states <=> there is a predecessor of ts which is not a target state
            predecessors = map(lambda sap: sap[0], system.predecessors(ts))
            return ts not in target_states or len(set(predecessors).difference(target_states)) != 0

        # remove target states that are only reachable from other target states
        reachable = set(filter(reachable_from_non_target_states, reachable)) 
        # for idx in range(len(reachable_mask)):
        #     reachable_mask[idx] = reachable_from_non_target_states(idx)
        
        if debug:
            print("removed target states that are only reachable from other target states")

        reachable = list(reachable)

        # ctr = 0
        # reachable_mapping = {}
        # for stateidx,state_reachable in enumerate(reachable_mask):
        #     if state_reachable:
        #         reachable_mapping[stateidx] = ctr
        #         ctr += 1 

        # reduce states + new target and new fail state 
        new_state_count = len(reachable) + 2
        target_idx, fail_idx = new_state_count - 2, new_state_count - 1
        
        if debug:
            print("new states: %s, target index: %s, fail index: %s" % (new_state_count, target_idx, fail_idx))
        
        # create a mapping from system to reachability form
        to_reachability, to_reachability_sap = {}, bidict()
        # [0...len(reachable)-1] reachable (mapped to respective states)
        # [len(reachable)...M-1] not in reachable but target states (mapped to target)
        # [M-1...N-1] neither reachable nor target states (mapped to fail)
        # overall N entries in "to_reachability"
        for sapidx in range(system.C):
            stateidx, actionidx = system.index_by_state_action.inv[sapidx]
            newidx = None
            if stateidx in reachable:
                # state is reachable
                newidx = reachable.index(stateidx)
                # newidx = reachable_mapping[stateidx] # result is something in [0,...len(reachable)-1]
                to_reachability_sap[(stateidx,actionidx)] = (newidx,actionidx)
            elif target_label in system.labels_by_state[stateidx]:
                # state is not in reachable but a target state
                # => map to target state
                newidx = target_idx
            else:
                # state is not reachable and not a target state
                # => map to fail state
                newidx = fail_idx
            to_reachability[stateidx] = newidx

        if debug:
            print("computed state-action mapping")

        # compute reduced transition matrix (without non-reachable states)
        # compute probability of reaching the target state in one step 
        new_N = len(reachable)
        new_C = len(set([(s,a) for (s,a) in system.index_by_state_action.keys() if s in reachable]))
        new_P = dok_matrix((new_C, new_N))
        if debug:
            print("shape of new_P %s" % (new_P.shape,))
        new_index_by_state_action = bidict()
        to_target = np.zeros(new_C)

        i = 0
        for (sapidx, destidx), p in system.P.items():
            sourceidx, action = system.index_by_state_action.inv[sapidx]
            new_sourceidx = to_reachability[sourceidx]
            new_destidx = to_reachability[destidx]
            
            if new_sourceidx not in [target_idx, fail_idx]:

                if (new_sourceidx, action) not in new_index_by_state_action:
                    new_index_by_state_action[new_sourceidx, action] = i
                    i += 1
                index = new_index_by_state_action[(new_sourceidx,action)]

                if target_label in system.labels_by_state[sourceidx]:
                    # if old source is a target state, then it is remapped to the new target state with p=1
                    to_target[index] = 1
                elif new_destidx not in [target_idx, fail_idx]:
                    # new transition matrix
                    new_P[index, new_destidx] = p

        if debug:
            print("computed transition matrix & to_target")

        self.P = new_P
        self.initial = to_reachability[initial]
        self.initial_label = initial_label
        self.target_label = target_label
        self.to_target = to_target
        self.index_by_state_action = new_index_by_state_action
        self.__system = self.__initialize_system(to_reachability_sap, system, target_label)

    @property
    def system(self):
        return self.__system

    def __initialize_system(self, mapping, configuration, target_label):
        C,N = self.P.shape
        P_compl = dok_matrix((C+2, N+2))
        target_state, fail_state = N, N+1

        index_by_state_action_compl = self.index_by_state_action.copy()
        index_by_state_action_compl[(target_state,0)] = C
        index_by_state_action_compl[(fail_state,0)] = C+1
        
        # copy labels from configuration (i.e. a system)
        # mapping defines which state-action pairs in the system map to which state-action pairs in the r.f.
        label_to_actions = defaultdict(set)
        label_to_states = defaultdict(set)
        for idx in range(C):
            stateidx, actionidx = self.index_by_state_action.inv[idx]
            sys_stateidx, sys_actionidx = mapping.inv[(stateidx,actionidx)]
            labels = configuration.labels_by_state[sys_stateidx]
            for l in labels:
                label_to_states[".%s"%l].add(stateidx)
            actionlabels = configuration.labels_by_action[(sys_stateidx,sys_actionidx)]
            for l in actionlabels:
                label_to_actions[".%s"%l].add((sys_stateidx,sys_actionidx))
        label_to_states["fail"].add(fail_state)
        label_to_states[target_label].add(target_state)

        not_to_fail = np.zeros(N)
        for (idx, dest), p in self.P.items():
            sourceidx, _ = index_by_state_action_compl.inv[idx]
            if p > 0:
                not_to_fail[sourceidx] += p
                P_compl[idx, dest] = p

        for idx, p_target in enumerate(self.to_target):
            sourceidx, _ = index_by_state_action_compl.inv[idx]
            if p_target > 0:
                P_compl[idx, target_state] = p_target
            p_fail = 1 - (p_target + not_to_fail[sourceidx])
            if p_fail > 0:
                P_compl[idx, fail_state] = p_fail

        P_compl[C, target_state] = 1
        P_compl[C+1, fail_state] = 1

        return type(configuration)( 
                    P=P_compl, 
                    index_by_state_action=index_by_state_action_compl, 
                    label_to_states=label_to_states,
                    label_to_actions=label_to_actions)

    def __repr__(self):
        return "ReachabilityForm(C=%s, N=%s, initial=%s, system=%s)" % (self.P.shape[0], self.P.shape[1], self.initial, self.system)

    def fark_z_constraints(self, threshold):
        """ Returns matrix and rhs of the Farkas z-constraints:

        .. math::

            (I-P) \, \mathbf{z} \leq \mathbf{b} \land \mathbf{z}(\\texttt{init}) \leq \lambda

        where :math:`\mathbf{b}`` is the vector "to_target", :math:`\lambda` is the threshold and :math:`I` is the matrix that for every row (state,action) has a 1 in the column "state".

        :param threshold: The threshold :math:`\lambda` for which the Farkas z-constraints should be constructed
        :type threshold: Float
        :return: :math:`C+1 \\times N`-matrix :math:`M`, and :math:`N`-vector :math:`rhs` such that :math:`M \mathbf{z} \leq rhs` yields the Farkas z-constraints
        """
        C,N = self.P.shape
        I = self._reach_form_id_matrix()

        rhs = self.to_target.copy()
        rhs.resize(C+1)
        rhs[C] = -threshold

        delta = np.zeros(N)
        delta[self.initial] = 1

        fark_z_matr = vstack(((I-self.P),-delta))
        return fark_z_matr, rhs

    def fark_y_constraints(self, threshold):
        """ Returns the constraints of the Farkas y-constraints:

        .. math::

            \mathbf{y} \, (I-P) \leq \delta_{\\texttt{init}} \land \mathbf{b} \, \mathbf{y} \leq \lambda

        where :math:`\mathbf{b}`` is the vector "to_target", :math:`\lambda` is the threshold and :math:`I` is the matrix that for every row (state,action) has a 1 in the column "state". The vector :math:`\delta_{\\texttt{init}}` is 1 for the initial state, and otherwise 0.

        :param threshold: The threshold :math:`\lambda` for which the Farkas y-constraints should be constructed
        :type threshold: Float
        :return: :math:`N+1 \\times C`-matrix :math:`M`, and :math:`C`-vector :math:`rhs` such that :math:`M \mathbf{y} \leq rhs` yields the Farkas y-constraints
        """
        C,N = self.P.shape
        I = self._reach_form_id_matrix()

        b = cast_dok_matrix(self.to_target)

        rhs = np.zeros(N+1)
        rhs[self.initial] = 1
        rhs[N] = -threshold

        fark_y_matr = hstack(((I-self.P),-b)).T
        return fark_y_matr, rhs

    def _reach_form_id_matrix(self):
        """Computes the matrix :math:`I` for a given reachability form that for every row (st,act) has an entry 1 at the column corresponding to st."""
        C,N = self.P.shape
        I = dok_matrix((C,N))

        for i in range(0,C):
            (state, _) = self.index_by_state_action.inv[i]
            I[i,state] = 1

        return I

    def max_z_state(self,solver="cbc"):
        C,N = self.P.shape
        matr, rhs = self.fark_z_constraints(0)
        opt = np.zeros(N)
        opt[self.initial] = 1
        max_z_lp = LP.from_coefficients(matr,rhs,opt,sense="<=",objective="max")
        result = max_z_lp.solve(solver=solver)
        return result.result_vector

    def max_z_state_action(self,solver="cbc"):
        max_z_vec = self.max_z_state(solver=solver)
        return (self.P.dot(max_z_vec) + self.to_target)

    def max_y_state_action(self,solver="cbc"):
        N,C = self.P.shape

        matr, rhs = self.fark_y_constraints(0)
        max_y_lp = LP.from_coefficients(matr,rhs,self.to_target,sense="<=",objective="max")

        result = max_y_lp.solve(solver=solver)
        return result.result_vector

    def max_y_state(self,solver="cbc"):
        C,N = self.P.shape
        max_y_vec = self.max_y_state_action(solver=solver)
        max_y_states = np.zeros(N)
        max_y_states[self.initial] = 1
        for sap_idx in range(C):
            (st,act) = self.index_by_state_action.inv[sap_idx]
            max_y_states[st] = max_y_states[st] + max_y_vec[sap_idx]
        return max_y_states

    def pr_min(self,solver="cbc"):
        return self.max_z_state()

    def pr_max(self,solver="cbc"):
        N,C = self.P.shape

        matr, rhs = self.fark_z_constraints(0)
        opt = np.zeros(N)
        opt[self.initial] = 1
        pr_max_z_lp = LP.from_coefficients(matr,rhs,opt,sense=">=",objective="min")
        result = pr_max_z_lp.solve(solver=solver)
        return result.result_vector

#     def induced_subsystem(self, state_vector):
#         """ Given a reachability form with :math:`N` states and a vector in :math:`\{0,1\}^N` 
#         computes the induced subsystem of that vector.
# 
#         :param reach_form: A reachability form
#         :type reach_form: model.ReachabilityForm
#         :param state_vector: a vector indicating which states to keep
#         :type state_vector: :math:`N`-vector over :math:`\{0,1\}`
#         :return: The induced subsystem as a rechability form, and a bidirectional mapping from states in the subsystem to states in the original system.
#         :rtype: Tuple[model.ReachabilityForm, bidict]
#         """
#         C,N = self.P.shape
# 
#         assert state_vector.size == N
#         assert state_vector[self.initial] == 1
# 
#         new_to_old_states = bidict()
#         new_index_by_state_action = bidict()
#         new_N = 0
#         new_C = 0
# 
#         # Map the new states to new indices and compute a map to the old states
#         for i in range(0,N):
#             assert state_vector[i] in [0,1]
#             if state_vector[i] == 1:
#                 new_to_old_states[new_N] = i
#                 new_N += 1
# 
#         # Compute the new number of choices (= rows)
#         for rowidx in range(0,C):
#             (source,action) = self.index_by_state_action.inv[rowidx]
#             if state_vector[source] == 1:
#                 new_source = new_to_old_states.inv[source]
#                 new_index_by_state_action[(new_source,action)] = new_C
#                 new_C += 1
# 
#         new_P = dok_matrix((new_C,new_N))
# 
#         # Populate the new transition matrix
#         for (rowidx,target) in self.P.keys():
#             (source,action) = self.index_by_state_action.inv[rowidx]
#             if state_vector[source] == 1 and state_vector[target] == 1:
#                 new_source = new_to_old_states.inv[source]
#                 new_target = new_to_old_states.inv[target]
#                 new_row_idx = new_index_by_state_action[(new_source,action)]
#                 new_P[new_row_idx,new_target] = self.P[rowidx,target]
# 
# 
#         new_to_target = np.zeros(new_N)
# 
#         for new_state in range(0,new_N):
#             old_to_target = self.to_target
#             new_to_target[new_state] = old_to_target[new_to_old_states[new_state]]
# 
#         new_initial = new_to_old_states.inv[self.initial]
# 
#         subsys_self = ReachabilityForm(new_P,new_initial,new_to_target,new_index_by_state_action)
# 
#         return subsys_self,new_to_old_states
