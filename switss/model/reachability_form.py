from . import AbstractMDP,MDP
from ..utils import InvertibleDict, cast_dok_matrix
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
    def __init__(self, system, initial_label, target_label="rf_target", fail_label="rf_fail", ignore_consistency_checks=False):
        if not ignore_consistency_checks:
            ReachabilityForm.assert_consistency(system, initial_label, target_label, fail_label)
        
        self.__P = system.P[:system.C-2, :system.N-2]
        self.__system = system
        self.initial = next(iter(system.states_by_label[initial_label]))
        self.target_label = target_label
        self.fail_label = fail_label
        self.initial_label = initial_label
        self.__index_by_state_action = system.index_by_state_action.copy()
        del self.__index_by_state_action.inv[system.C-2]
        del self.__index_by_state_action.inv[system.C-1]
        
        self.A = self._reach_form_id_matrix() - self.__P
        self.to_target = system.P.getcol(system.N-2).todense()[:system.C-2]
        
    @staticmethod
    def assert_consistency(system, initial_label, target_label="rf_target", fail_label="rf_fail"):
        assert isinstance(system, AbstractMDP)
        assert len({initial_label, target_label, fail_label}) == 3, "Initial label, target label and fail label must be pairwise distinct"

        # check that there is exactly 1 target,fail and initial state resp.
        states = []
        for statelabel, name in [(initial_label, "initial"), (target_label, "target"), (fail_label, "fail")]:
            labeledstates = system.states_by_label[statelabel].copy()
            count = len(labeledstates)  
            assert count == 1, \
                "There must be exactly 1 %s state. There are %s states in system %s with label %s" % (name, count, system, statelabel)
            states.append(labeledstates.pop())
        init,target,fail = states
        
        # check that fail and target state only map to themselves
        for state,name,rowidx,colidx in [(target,"target",system.C-2,system.N-2),(fail,"fail",system.C-1,system.N-1)]:
            successors = list(system.successors(state))
            assert len(successors) == 1 and successors[0][0] == state, "State %s must only one action and successor; itself" % name
            saindex = system.index_by_state_action[successors[0]]
            assert saindex == rowidx, "State-action of %s must be at index %s but is at %s" % (name, rowidx, saindex)
            assert state == colidx, "State %s must be at index %s but is at %s" % (name, colidx, state)

        # fail_mask has a 1 only at the fail state and zeros otherwise
        fail_mask = np.zeros(system.N,dtype=np.bool)
        fail_mask[fail] = True 

        # check that every state is reachable
        # the fail state doesn't need to be reachable
        fwd_mask = system.reachable_mask({init},"forward")
        assert (fwd_mask | fail_mask).all(), "Not every state is reachable from %s in system %s" % (initial_label, system)

        # check that every state except fail reaches goal
        # if bwd_mask[fail] == 1 then bwd_mask[fail] ^ fail_mask[fail] == 0
        bwd_mask = system.reachable_mask({target},"backward")
        assert (bwd_mask ^ fail_mask).all(), "Not every state reaches %s in system %s" % (target_label, system)

    @staticmethod
    def reduce(system, initial_label, target_label, new_target_label="rf_target", new_fail_label="rf_fail", debug=False):
        assert isinstance(system, AbstractMDP)
        assert new_target_label not in system.states_by_label.keys(), "Label '%s' for target state already exists in system %s" % (new_target_label, system)
        assert new_fail_label not in system.states_by_label.keys(), "Label '%s' for fail state already exists in system %s" % (new_fail_label, system)
        assert len(system.states_by_label[target_label]) > 0, "There needs to be at least one target state."
        target_states = system.states_by_label[target_label]
        initial_state_count = len(system.states_by_label[initial_label])
        assert initial_state_count == 1, "There are %d states with label '%s'. Must be 1." % (initial_state_count, initial_label)
        initial = list(system.states_by_label[initial_label])[0]
        
        if debug:
            print("calculating reachable mask (backward)...")
        backward_reachable = system.reachable_mask(target_states, "backward")
        if debug:
            print("calculating reachable mask (forward)...")
        forward_reachable = system.reachable_mask(set([initial]), "forward", blacklist=target_states)
        # states which are reachable from the initial state AND are able to reach target states
        reachable_mask = backward_reachable & forward_reachable
        # TODO: use reachable_mask instead of reachable everywhere
        # this is much better for performance since lookup of states is always O(1)
        reachable = { idx for idx,x in enumerate(reachable_mask) if x }

        if debug:
            print("tested backward & forward reachability test")
        
        # if debug:
        #     print("removed target states that are only reachable from other target states")

        reachable = list(reachable)

        # reduce states + new target and new fail state 
        new_state_count = len(reachable) + 2
        target_idx, fail_idx = new_state_count - 2, new_state_count - 1
        
        if debug:
            print("new states: %s, target index: %s, fail index: %s" % (new_state_count, target_idx, fail_idx))
        
        # create a mapping from system to reachability form
        to_rf_cols, to_rf_rows = {}, bidict()

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
                to_rf_rows[(stateidx,actionidx)] = (newidx,actionidx)
            elif target_label in system.labels_by_state[stateidx]:
                # state is not in reachable but a target state
                # => map to target state
                newidx = target_idx
            else:
                # state is not reachable and not a target state
                # => map to fail state
                newidx = fail_idx
            to_rf_cols[stateidx] = newidx

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
            new_sourceidx = to_rf_cols[sourceidx]
            new_destidx = to_rf_cols[destidx]
            
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
        
        rf_system = ReachabilityForm.__initialize_system(
            new_P, 
            new_index_by_state_action,
            to_target, 
            to_rf_rows, 
            system, 
            new_target_label, 
            new_fail_label)

        rf = ReachabilityForm(
            rf_system, 
            initial_label, 
            target_label=new_target_label, 
            fail_label=new_fail_label, 
            ignore_consistency_checks=True)

        return rf, to_rf_cols, to_rf_rows

    @property
    def system(self):
        return self.__system

    @staticmethod
    def __initialize_system(P, index_by_state_action, to_target, mapping, configuration, target_label, fail_label):
        C,N = P.shape
        P_compl = dok_matrix((C+2, N+2))
        target_state, fail_state = N, N+1

        index_by_state_action_compl = index_by_state_action.copy()
        index_by_state_action_compl[(target_state,0)] = C
        index_by_state_action_compl[(fail_state,0)] = C+1
        
        # copy labels from configuration (i.e. a system)
        # mapping defines which state-action pairs in the system map to which state-action pairs in the r.f.
        label_to_actions = defaultdict(set)
        label_to_states = defaultdict(set)
        for idx in range(C):
            stateidx, actionidx = index_by_state_action.inv[idx]
            sys_stateidx, sys_actionidx = mapping.inv[(stateidx,actionidx)]
            labels = configuration.labels_by_state[sys_stateidx]
            for l in labels:
                label_to_states[l].add(stateidx)
            actionlabels = configuration.labels_by_action[(sys_stateidx,sys_actionidx)]
            for l in actionlabels:
                label_to_actions[l].add((sys_stateidx,sys_actionidx))
        label_to_states[fail_label].add(fail_state)
        label_to_states[target_label].add(target_state)

        not_to_fail = np.zeros(C)
        for (idx, dest), p in P.items():
            if p > 0:
                not_to_fail[idx] += p
                P_compl[idx, dest] = p

        for idx, p_target in enumerate(to_target):
            if p_target > 0:
                P_compl[idx, target_state] = p_target
            p_fail = 1 - (p_target + not_to_fail[idx])
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
        return "ReachabilityForm(initial=%s, target=%s, fail=%s, system=%s)" % (self.initial_label, self.target_label, self.fail_label, self.system)

    def fark_z_constraints(self, threshold):
        """ Returns matrix and rhs of the Farkas z-constraints:

        .. math::

            (I-P) \, \mathbf{z} \leq \mathbf{b} \land \mathbf{z}(\\texttt{init}) \leq \lambda

        where :math:`\mathbf{b}`` is the vector "to_target", :math:`\lambda` is the threshold and :math:`I` is the matrix that for every row (state,action) has a 1 in the column "state".

        :param threshold: The threshold :math:`\lambda` for which the Farkas z-constraints should be constructed
        :type threshold: Float
        :return: :math:`C+1 \\times N`-matrix :math:`M`, and :math:`N`-vector :math:`rhs` such that :math:`M \mathbf{z} \leq rhs` yields the Farkas z-constraints
        """
        C,N = self.__P.shape

        rhs = self.to_target.A1.copy()
        rhs.resize(C+1)
        rhs[C] = -threshold

        delta = np.zeros(N)
        delta[self.initial] = 1

        fark_z_matr = vstack((self.A,-delta))
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
        C,N = self.__P.shape

        b = cast_dok_matrix(self.to_target)

        rhs = np.zeros(N+1)
        rhs[self.initial] = 1
        rhs[N] = -threshold

        fark_y_matr = hstack((self.A,-b)).T
        return fark_y_matr, rhs

    def _reach_form_id_matrix(self):
        """Computes the matrix :math:`I` for a given reachability form that for every row (st,act) has an entry 1 at the column corresponding to st."""
        C,N = self.__P.shape
        I = dok_matrix((C,N))

        for i in range(0,C):
            (state, _) = self.__index_by_state_action.inv[i]
            I[i,state] = 1

        return I

    def max_z_state(self,solver="cbc"):
        C,N = self.__P.shape
        matr, rhs = self.fark_z_constraints(0)
        opt = np.ones(N)
        max_z_lp = LP.from_coefficients(
            matr,rhs,opt,sense="<=",objective="max")

        for st_idx in range(N):
            max_z_lp.add_constraint([(st_idx,1)],">=",0)
            max_z_lp.add_constraint([(st_idx,1)],"<=",1)

        result = max_z_lp.solve(solver=solver)
        return result.result_vector

    def max_z_state_action(self,solver="cbc"):
        max_z_vec = self.max_z_state(solver=solver)
        return self.__P.dot(max_z_vec) + self.to_target.A1

    def max_y_state_action(self,solver="cbc"):
        N,C = self.__P.shape

        matr, rhs = self.fark_y_constraints(0)
        max_y_lp = LP.from_coefficients(
            matr,rhs,self.to_target,sense="<=",objective="max")

        for sap_idx in range(C):
            max_y_lp.add_constraint([(sap_idx,1)],">=",0)

        result = max_y_lp.solve(solver=solver)
        return result.result_vector

    def max_y_state(self,solver="cbc"):
        C,N = self.__P.shape
        max_y_vec = self.max_y_state_action(solver=solver)
        max_y_states = np.zeros(N)
        max_y_states[self.initial] = 1
        for sap_idx in range(C):
            (st,act) = self.__index_by_state_action.inv[sap_idx]
            max_y_states[st] = max_y_states[st] + max_y_vec[sap_idx]
        return max_y_states

    def pr_min(self,solver="cbc"):
        return self.max_z_state()

    def pr_max(self,solver="cbc"):
        C,N = self.__P.shape

        matr, rhs = self.fark_z_constraints(0)
        opt = np.ones(N)
        pr_max_z_lp = LP.from_coefficients(
            matr,rhs,opt,sense=">=",objective="min")

        for st_idx in range(N):
            pr_max_z_lp.add_constraint([(st_idx,1)],">=",0)
            pr_max_z_lp.add_constraint([(st_idx,1)],"<=",1)

        result = pr_max_z_lp.solve(solver=solver)
        return result.result_vector

    def _check_mec_freeness(self):

        # indices of old fail and target state
        target_state, target_action = self.system.N-2, self.system.C-2
        fail_state, fail_action = self.system.N-1, self.system.C-1

        if len(set(self.system.predecessors(fail_state))) == 1:
            # if that happens, then fail state has no predecessors but itself.
            # in that case, the fail state has no impact on the other states.
            assert (self.pr_min() == 1).all()
            return

        import copy
        new_label_to_states = copy.deepcopy(self.system.states_by_label)
        new_index_by_state_action = copy.deepcopy(self.system.index_by_state_action)

        # create a new transition matrix with 2 new entries for a new target and fail state
        P = dok_matrix((self.system.C+2,self.system.N+2))
        P[:self.system.C,:self.system.N] = self.system.P

        # indices of new target and new fail state according to RF
        new_target_state, new_target_action = self.system.N, self.system.C
        new_fail_state, new_fail_action = self.system.N+1, self.system.C+1
        # index state-action pairs
        new_index_by_state_action[(new_target_state,0)] = new_target_action
        new_index_by_state_action[(new_fail_state,0)] = new_fail_action
        
        # map new target and new fail state only to themselves
        P[new_target_action,new_target_state] = 1
        P[new_fail_action,new_fail_state] = 1

        # remap old fail & target state to new target state
        P[target_action,target_state] = 0
        P[target_action,new_target_state] = 1
        P[fail_action,fail_state] = 0
        P[fail_action,new_target_state] = 1
        
        # remove fail and target label from old target and fail state
        new_label_to_states.remove(self.target_label, target_state)
        new_label_to_states.remove(self.fail_label, fail_state)
        # add fail and target label to new target and new fail state
        new_label_to_states.add(self.target_label, new_target_state)
        new_label_to_states.add(self.fail_label, new_fail_state)

        # new system should already be in RF, so calling .reduce is not necessary
        target_or_fail_system = type(self.system)(
            P=P, 
            index_by_state_action=new_index_by_state_action,
            label_to_actions={},
            label_to_states=new_label_to_states)
        
        target_or_fail_rf = ReachabilityForm(
            target_or_fail_system,
            self.initial_label,
            self.target_label,
            self.fail_label)

        assert (target_or_fail_rf.pr_min() == 1).all(), target_or_fail_rf.pr_min()

