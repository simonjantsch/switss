from . import AbstractMDP,MDP,ReachabilityForm

import copy as copy
import numpy as np
from scipy.sparse import dok_matrix,hstack,vstack
from ..utils import InvertibleDict, cast_dok_matrix, DTMCVisualizationConfig, VisualizationConfig
from ..solver.milp import LP


class RewardReachabilityForm:

    def __init__(self, reachability_form, reward_vector):

        self.__reach_form = reachability_form
        self.reward_vector = reward_vector

        self.__system = reachability_form.system
        self.__P = reachability_form.system.P[:self.__system.C-2, :self.__system.N-2]
        self.__A = reachability_form.A
        self.__index_by_state_action = reachability_form.system.index_by_state_action.copy()
        del self.__index_by_state_action.inv[self.__system.C-2]
        del self.__index_by_state_action.inv[self.__system.C-1]

        self.target_label = reachability_form.target_label
        self.initial_label = reachability_form.initial_label
        self.initial = reachability_form.initial

    @property
    def system(self):
        """The underlying system (instance of model.AbstractMDP)"""
        return self.__system

    @property
    def A(self):
        return self.__A

    @property
    def is_ec_free(self):
        return True

    def in_proper_ec(self, state):
        return False

    @classmethod
    def fromsystem(cls, system, initial_label, target_label="rrf_target",ignore_consistency_checks=False):
        if not ignore_consistency_checks:
            RewardReachabilityForm.assert_system_consistency(system, initial_label, target_label)

        # add a dummy fail state, which is required for reachability form
        N = system.P.N
        C = system.P.C

        system.P.resize(C + 1, N + 1)
        system.P[C + 1, N + 1] = 1

        system.add_label(N, "rf_fail")

        # initialize underlying reachability form
        return cls(ReachabilityForm(system, initial_label, target_label,"rf_fail",ignore_consistency_checks), system.reward_vector[:-2])


    @staticmethod
    def reduce(system, initial_label, target_label, new_target_label="rrf_target", debug=False, ignore_consistency_checks=False):

        reach_form, to_rf_cols, to_rf_rows = ReachabilityForm.reduce(system, initial_label, target_label)
        RewardReachabilityForm.assert_reachform_consistency(reach_form)

        return RewardReachabilityForm(reach_form, reach_form.system.reward_vector[:-2]), to_rf_cols, to_rf_rows

    @staticmethod
    def assert_reachform_consistency(reachform):
        """Checks whether a reachability form fulfills the required properties to be the underlying RF of a reward reachability form.

        :param reachform: The Reachability form
        :type system: model.ReachabilityForm
        """
        assert isinstance(reachform, ReachabilityForm)
        assert reachform.system.reward_vector is not None

        fail_mask = np.zeros(reachform.system.N,dtype=bool)
        fail_mask[reachform.fail_state_idx] = True

        bwd_mask = reachform.system.reachable_mask({reachform.fail_state_idx},"backward")
        assert (bwd_mask == fail_mask).all(), "The fail state is reachable in a RF when trying to define a reward reachability form"
        
        target_mask = np.zeros(reachform.system.N,dtype=bool)
        target_mask[reachform.target_state_idx] = True

        mecs,proper_mecs,nr_of_mecs = reachform.system.maximal_end_components()

        print("mecs:" + str(mecs))
        print("proper_mecs:" + str(proper_mecs))
        print("nr_of_mecs:" + str(nr_of_mecs))

        assert (sum(proper_mecs) == 2), "there is some proper end component apart from the target and fail state"

    @staticmethod
    def assert_system_consistency(system, initial_label, target_label="rf_target"):
        """Checks whether a system fulfills the reward reachability form properties.

        :param system: The system
        :type system: model.AbstractMDP
        :param initial_label: Label of initial state
        :type initial_label: str
        :param target_label: Label of target state, defaults to "rf_target"
        :type target_label: str, optional
        """
        assert isinstance(system, AbstractMDP)
        assert system.reward_vector is not None
        assert len({initial_label, target_label}) == 2, "Initial label and target label must be distinct"

        # check that there is exactly 1 target and initial state resp.
        states = []
        for statelabel, name in [(initial_label, "initial"), (target_label, "target")]:
            labeledstates = system.states_by_label[statelabel].copy()
            count = len(labeledstates)  
            assert count == 1, \
                "There must be exactly 1 %s state. There are %s states in system %s with label %s" % (name, count, system, statelabel)
            states.append(labeledstates.pop())
        init,target = states
        
        # check that target state only maps to itself
        for state,name,rowidx,colidx in [(target,"target",system.C-1,system.N-1)]:
            successors = list(system.successors(state))
            assert len(successors) == 1 and successors[0][0] == state, "State %s must only one action and successor; itself" % name
            succst,succact,p = successors[0]
            saindex = system.index_by_state_action[(succst,succact)]
            assert saindex == rowidx, "State-action of %s must be at index %s but is at %s" % (name, rowidx, saindex)
            assert state == colidx, "State %s must be at index %s but is at %s" % (name, colidx, state)

        # check that every state is reachable
        fwd_mask = system.reachable_mask({init},"forward")
        assert (fwd_mask).all(), "Not every state is reachable from %s in system %s" % (initial_label, system)

        # check that every state reaches goal
        bwd_mask = system.reachable_mask({target},"backward")
        assert (bwd_mask).all(), "Not every state reaches %s in system %s" % (target_label, system)

        # check that there is no proper end component apart from the target state
        mec_vec, proper_mec_vec, nr_of_mecs = system.maximal_end_components()
        assert (sum(proper_mec_vec) == 1), "there is some proper end component apart from the target state"
        assert (proper_mec_vec[mec_vec[target]] == 1), "target state does not induce a proper end component"

    def fark_constraints(self, threshold, mode):
        """returns the right constraint set dependent on the given mode.

        :param threshold: the threshold
        :type threshold: float
        :param mode: either 'min' or 'max'
        :type mode: str
        :param rewards: if true, returns Farkas constraints for reward thresholds. Defaults to False
        :type rewards: Optional bool
        :return: either :math:`(C+1) \\times N`-matrix :math:`M_z`, and vector of length :math:`C+1` :math:`rhs_z` or :math:`(N+1) \\times C`-matrix :math:`M_y`, and :math:`N+1`-vector :math:`rhs_y`.
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """ 
        assert mode in ["min", "max"]

        if mode == "min":
            return self.fark_z_constraints(threshold)
        else:
            return self.fark_y_constraints(threshold)

    def fark_z_constraints(self, threshold):
        """
        Returns a matrix :math:`M_z` and a vector :math:`rhs_z` such that for a :math:`N` vector :math:`\mathbf{z}`

        .. math::

            M_z\, \mathbf{z} \leq rhs_z \quad \\text{  iff  } \quad 
            \\mathbf{A} \, \mathbf{z} \leq \mathbf{b} \land \mathbf{z}(\\texttt{init}) \geq \lambda
            \quad \\text{  iff  } \quad
            \mathbf{z} \in \mathcal{P}^{\\text{min}}(\lambda)
                
        :param threshold: The threshold :math:`\lambda` for which the Farkas z-constraints should be constructed
        :type threshold: Float
        :return: :math:`(C+1) \\times N`-matrix :math:`M_z`, and vector of length :math:`C+1` :math:`rhs_z`
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """
        C,N = self.__P.shape

        rhs = self.reward_vector.copy()
        rhs.resize(C+1)
        rhs[C] = -threshold

        delta = np.zeros(N)
        delta[self.initial] = 1

        fark_z_matr = vstack((self.A,-delta))
        return fark_z_matr, rhs

    def fark_y_constraints(self, threshold):
        """ 
        Returns a matrix :math:`M_y` and a vector :math:`rhs_y` such that for a :math:`C` vector :math:`\mathbf{y}`

        .. math::

            M_y\, \mathbf{y} \leq rhs_y \quad \\text{  iff  } \quad
            \mathbf{y} \, \mathbf{A} \leq \delta_{\\texttt{init}} \land \mathbf{b} \, \mathbf{y} \geq \lambda
            \quad \\text{  iff  } \quad
            \mathbf{y} \in \mathcal{P}^{\\text{max}}(\lambda)

        where :math:`\lambda` is the threshold, :math:'\mathbf{A}' is the system matrix and :math:`\mathbf{b}` is the reward vector. The vector :math:`\delta_{\\texttt{init}}` is 1 for the initial state, and otherwise 0.

        :param threshold: The threshold :math:`\lambda` for which the Farkas y-constraints should be constructed
        :type threshold: Float
        :return: :math:`(N+1) \\times C`-matrix :math:`M_y`, and :math:`N+1`-vector :math:`rhs_y` 
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """
        C,N = self.__P.shape

        b = cast_dok_matrix(self.reward_vector)

        rhs = np.zeros(N+1)
        rhs[self.initial] = 1
        rhs[N] = -threshold

        fark_y_matr = hstack((self.A,-b)).T
        return fark_y_matr, rhs

    def max_z_state(self,solver="cbc"):
        """
        Returns a solution to the LP        

        .. math::

            \max \, \sum_{s} \mathbf{x}(s) \quad \\text{ subject to } \quad \mathbf{x} \in \mathcal{P}^{\\text{min}}(0)
            
        The solution vector corresponds to the minimal reward, i.e. 
        :math:`\mathbf{x}^*(s) = \mathbf{ExRew}^{\\text{min}}_s(\diamond \\text{goal})` for all :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        
        C,N = self.__P.shape
        matr, rhs = self.fark_z_constraints(0)
        opt = np.ones(N)
        max_z_lp = LP.from_coefficients(
            matr,rhs,opt,sense="<=",objective="max")

        for st_idx in range(N):
            max_z_lp.add_constraint([(st_idx,1)],">=",0)

        result = max_z_lp.solve(solver=solver)
        return result.result_vector

    def max_z_state_action(self,solver="cbc"):
        """
        Let :math:`\mathbf{x}` be a solution vector to `max_z_state`. This function then returns a 
        :math:`C` vector :math:`\mathbf{v}` such that

        .. math::

            \mathbf{v}((s,a)) = rew(s) + \sum_{d \in S } \mathbf{P}((s,a),d) \mathbf{x}(d)

        for all :math:`(s,a) \in \mathcal{M}`.

        :param solver: [description], defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        

        max_z_vec = self.max_z_state(solver=solver)
        return self.__P.dot(max_z_vec) + self.reward_vector

    def max_y_state_action(self,solver="cbc"):
        """
        Returns a solution to the LP        

        .. math::

            \max \, \mathbf{rew} \, \mathbf{x} \quad \\text{ subject to } \quad \mathbf{x} \in \mathcal{P}^{\\text{max}}(0)
            
        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """
        C,N = self.__P.shape

        matr, rhs = self.fark_y_constraints(0)
        max_y_lp = LP.from_coefficients(
            matr,rhs,self.reward_vector,sense="<=",objective="max")

        for sap_idx in range(C):
            max_y_lp.add_constraint([(sap_idx,1)],">=",0)

        result = max_y_lp.solve(solver=solver)
        return result.result_vector

    def max_y_state(self,solver="cbc"):
        """
        Let :math:`\mathbf{x}` be a solution vector to `max_y_state_action`. This function then returns a 
        :math:`N` vector :math:`\mathbf{v}` such that

        .. math::

            \mathbf{v}(s) = \sum_{a \in \\text{Act}(s)} \mathbf{x}((s,a))

        for all :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        
        C,N = self.__P.shape
        max_y_vec = self.max_y_state_action(solver=solver)
        max_y_states = np.zeros(N)
        max_y_states[self.initial] = 1
        for sap_idx in range(C):
            (st,act) = self.__index_by_state_action.inv[sap_idx]
            max_y_states[st] = max_y_states[st] + max_y_vec[sap_idx]
        return max_y_states

    def rew_min(self,solver="cbc"):
        """Computes an :math:`N` vector :math:`\mathbf{x}` such that 
        :math:`\mathbf{x}(s) = \mathbf{ExpRew}^{\\text{min}}_s(\diamond \\text{goal})` for :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        
        return self.max_z_state(solver=solver)

    def rew_max(self,solver="cbc"):
        """Computes an :math:`N` vector :math:`\mathbf{x}` such that :math:`\mathbf{x}(s) = 
        \mathbf{ExpRew}^{\\text{max}}_s(\diamond \\text{goal})` for :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """
        C,N = self.__P.shape

        opt = np.ones(N)
        pr_max_z_lp = LP.from_coefficients(
            self.A, self.reward_vector , opt, sense=">=", objective="min")

        for st_idx in range(N):
            pr_max_z_lp.add_constraint([(st_idx,1)],">=",0)

        result = pr_max_z_lp.solve(solver=solver)
        return result.result_vector
