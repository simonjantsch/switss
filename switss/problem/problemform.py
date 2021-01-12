from abc import ABC, abstractmethod, abstractproperty
from bidict import bidict
from collections import deque
import numpy as np

from ..solver import MILP,LP
from ..utils import InvertibleDict

class ProblemFormulation:
    """A ProblemFormulation is an abstract base class for
    problems that are aimed at finding minimal witnesses
    for DTMCs or MDPs.
    """    
    def __init__(self):
        pass

    def solve(self, 
              reachability_form, 
              threshold, 
              mode, 
              labels=None, 
              timeout=None, 
              fixed_values=dict()):
        """Searches for small subsystems for a given reachability form
        such that the probability of reaching the target state is above
        a given threshold: 

        .. math::

            \mathbf{Pr}_{\mathbf{x}}^{*}(\diamond \\text{goal}) \geq \lambda

        where :math:`\lambda` is the given threshold and :math:`* \in \{\\text{min},\\text{max}\}`.

        `.solve` returns the final result if multiple solutions are found.

        :param reachability_form: The system that should be minimized.
        :type reachability_form: model.ReachabilityForm
        :param threshold: The given threshold.
        :type threshold: float
        :param mode: The polytope that should be selected for optimization, either "min" or "max"
        :type mode: str
        :param labels: A list of labels. 
        :type labels: List[str]
        :param fixed_values: A dictionary mapping states to fixed values.
        :type fixed_values: Dict[int, int]
        :return: The resulting subsystem.
        :rtype: problem.Subsystem
        """
        return deque(self.solveiter(    reachability_form, 
                                        threshold, 
                                        mode,
                                        labels=labels,
                                        timeout=timeout,
                                        fixed_values=fixed_values), maxlen=1).pop()

    def solveiter(self, 
                  reachability_form, 
                  threshold, 
                  mode, 
                  labels=None, 
                  timeout=None, 
                  fixed_values=dict()):
        """Searches for small subsystems for a given reachability form
        such that the probability of reaching the target state is above
        a given threshold: 

        .. math::

            \mathbf{Pr}_{\mathbf{x}}^{*}(\diamond \\text{goal}) \geq \lambda

        where :math:`\lambda` is the given threshold and :math:`* \in \{\\text{min},\\text{max}\}`.

        `.solveiter` returns an iterator over all systems that are found. 
        
        :param reachability_form: The system that should be minimized.
        :type reachability_form: model.ReachabilityForm
        :param threshold: The given threshold.
        :type threshold: float
        :param mode: The polytope that should be selected for optimization, either "min" or "max"
        :type mode: str
        :param labels: A list of labels. 
        :type labels: List[str]
        :param fixed_values: A dictionary mapping states or state-action-pairs 
            or labels to fixed values.
        :type fixed_values: Dict[int, int] or Dict[str, int]
        :return: The resulting subsystem.
        :rtype: problem.Subsystem
        """
        assert (threshold >= 0) and (threshold <= 1)
        assert mode in ["min","max"]

        if labels is not None:
            for l in labels:
                available = reachability_form.system.states_by_label.keys()
                assert l in available, "'%s' is not an existing label. \
                                       Available are %s" % (l,available)
            groups = ProblemFormulation._groups_from_labels(reachability_form, labels, mode)

        else:
            C,N = reachability_form.system.P.shape
            maxvars = N-2 if mode == "min" else C-2
            groups = InvertibleDict({ i : { i } for i in range(maxvars)})

        return self._solveiter(reachability_form, 
                               threshold,
                               mode, 
                               groups, 
                               timeout=timeout, 
                               fixed_values=fixed_values)

    @abstractmethod
    def _solveiter(self, 
                   reachability, 
                   threshold, 
                   mode, 
                   groups, 
                   timeout=None, 
                   fixed_values=dict()):
        pass

    def __repr__(self):
        params = ["%s=%s" % (k,v) for k,v in self.details.items() if k != "type"]
        return "%s(%s)" % (self.details["type"], ",".join(params))

    @abstractproperty
    def details(self):
        """A dictionary that contains information about this instance. Content
        is dependent on respective class and instance.
        """        
        pass

    @staticmethod
    def _compute_upper_bound(matr,rhs,lower_bound=0,solver="cbc"):
        C,N = matr.shape

        upper_obj = np.ones(N)
        upper_bound_LP = LP.from_coefficients(
            matr,rhs,upper_obj,objective="max")

        for idx in range(N):
            upper_bound_LP.add_constraint([(idx,1)],">=",0)

        lp_result = upper_bound_LP.solve(solver=solver)

        assert lp_result.status != "unbounded"
        return lp_result.status,lp_result.value

    @staticmethod
    def _groups_program(matr,
                            rhs,
                            groups,
                            upper_bound = None,
                            indicator_type="binary",
                            fixed_values=dict()):

        assert indicator_type in ["binary","real"]
        C,N = matr.shape

        # compute upper bound
        if upper_bound is None:
            stat, upper_bound = ProblemFormulation._compute_upper_bound(matr,rhs)
            if stat != "optimal":
                return None, None

        # construct basic program
        if indicator_type == "binary":
            program = MILP.from_coefficients(
                matr, rhs, np.zeros(N), 
                ["real"]*N, sense="<=", objective="min")
        else:
            program = LP.from_coefficients(
                matr, rhs, np.zeros(N),
                sense="<=", objective="min")

        # add indicator variables (one for each group)
        indicator_to_group = {}
        for group, var_indices in groups.items():
            indicator_var = program.add_variables(indicator_type)
            indicator_to_group[indicator_var] = var_indices
            for var_idx in var_indices: # add constraints var[i] <= K*indicator[i] 
                program.add_constraint([(var_idx,1), (indicator_var, -upper_bound)], "<=", 0)
            if group in fixed_values: # restrict indicator to fixed value
                fixed_value = fixed_values[group]
                program.add_constraint([(indicator_var, 1)], "=", fixed_value)
            elif indicator_type != "binary": # restrict indicator to range [0,1] if not binary
                program.add_constraint([(indicator_var, 1)], "<=", 1)
                program.add_constraint([(indicator_var, 1)], ">=", 0)
        indicator_to_group = InvertibleDict(indicator_to_group)

        return program, indicator_to_group

    @staticmethod
    def _project_from_binary_indicators(result_vector,
                                        projected_length,
                                        indicator_to_group):
        result_projected = np.zeros(projected_length)
        handled_vars = dict()
        
        for (indicator, group) in indicator_to_group.items():
            if result_vector[indicator] == 1:
                for var_idx in group:
                    result_projected[var_idx] = result_vector[var_idx]
                    handled_vars[var_idx] = True
            else:
                for var_idx in group:
                    result_projected[var_idx] = 0
                    handled_vars[var_idx] = True

        for n in range(projected_length):
            if n not in handled_vars.keys():
                result_projected[n] = result_vector[n]
            elif not handled_vars[n]:
                result_projected[n] = result_vector[n]

        return result_projected

    @staticmethod
    def _groups_from_labels(reach_form, labels, mode): 
        """Creates an invertible mapping from labels to states (min) or state-actions pair indices (max), dependent on mode. 

        :param reach_form: The reachability form that contains states with labels respectively. 
        :type reach_form: model.ReachabilityForm
        :param labels: List of labels that should be considered.
        :type labels: list[str]
        :param mode: either "min" or "max".
        :type mode: str
        :return: Mapping from labels to states or state-action pair indices.
        :rtype: utils.InvertibleDict[str, Set[int]]
        """              
        assert mode in ["min","max"]
        C,N = reach_form.system.P.shape

        # maps lables to sets of states
        sys_st_by_label = reach_form.system.states_by_label
        min_labels = InvertibleDict({ l : sys_st_by_label[l] for l in labels})

        if mode == "min":
            # for state-based optimization min_labels is sufficient
            return min_labels
        else:
            # for state-action-based optimization, we need to consider all state-action pairs
            groups = InvertibleDict({})
            # iterate over all state-labels-pairs
            for state, state_labels in min_labels.inv.items():
                # compute actions for this state
                actions = reach_form.system.actions_by_state[state]
                for act in actions:
                    sap_idx = reach_form.system.index_by_state_action[(state, act)]
                    # add corresponding state-action-pair index to this label
                    for label in state_labels:
                        groups.add(label, sap_idx)
            return groups

