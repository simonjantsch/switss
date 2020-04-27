from . import ProblemFormulation, ProblemResult, Subsystem
from farkas.solver import SolverResult
from farkas.utils import InvertibleDict

from bidict import bidict
import numpy as np


class MILPExact(ProblemFormulation):
    """MILPExact implements the computation of minimal witnessing subsystems
    using mixed integer linear programs (MILP) over the corresponding
    Farkas-polytopes, based on the following MILP:

    .. math::

       \min \sum_G \sigma(G) \;\; \\text{ subj. to } \;\; \mathbf{x} \in \\mathcal{F}(\\lambda)  \;\; \\text{ and }  \;\; \mathbf{x}(i) \leq K \cdot \sigma(g(i))

    where :math:`\sigma` is a vector of binary variables, :math:`g` is a
    mapping from the variables into a set of groups,
    :math:`\mathcal{F}(\lambda)` is the Farkas (y- or z-)polytope for
    threshold :math:`\lambda` and :math:`K` is a upper bound on the variables
    :math:`\mathbf{x}` in :math:`\mathcal{F}(\lambda)`.

    It follows that in any solution :math:`\sigma(G)` is one iff one of the variables :math:`\mathbf{x}(i)` such that :math:`g(i) = G` is strictly positive.
    """
    def __init__(self, mode, solver="cbc"):
        super().__init__()
        assert mode in ["min","max"]

        self.solver = solver
        self.mode = mode

    def __repr__(self):
        return "MILPExact(mode=%s, solver=%s)" % (
            self.mode, self.solver)

    def solve(self, reach_form, threshold, labels=None):
        """Runs MILPExact using the Farkas (y- or z-) polytope
        depending on the value in mode."""
        assert (threshold >= 0) and (threshold <= 1)
        if labels != None:
            for l in labels:
                assert ("." + l) in reach_form.system.states_by_label.keys()

        if self.mode == "min":
            return self.solve_min(reach_form, threshold, labels)
        else:
            return self.solve_max(reach_form, threshold, labels)

    def solve_min(self, reach_form, threshold, labels=None):
        """Runs MILPExact using the Farkas z-polytope."""

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_z_constraints(threshold)

        if labels == None:
            var_groups = InvertibleDict({i : set([i]) for i in range(N) })
        else:
            var_groups = ProblemFormulation._var_groups_from_labels(
                reach_form,labels,"min")

        milp_result = min_nonzero_groups(fark_matr,
                                          fark_rhs,
                                          var_groups,
                                          upper_bound=1,
                                          solver=self.solver)

        # this creates a new C-dimensional vector which carries values for state-action pairs.
        # every state-action pair is assigned the weight the state has.
        # this "blowing-up" will make it easier for visualizing subsystems.
        state_action_weights = np.zeros(C)
        for idx in range(C):
            state,_ = reach_form.index_by_state_action.inv[idx]
            state_action_weights[idx] = milp_result.result_vector[state]

        witness = Subsystem(reach_form, state_action_weights)

        return ProblemResult(
            milp_result.status,witness,milp_result.value)

    def solve_max(self, reach_form, threshold, labels=None):
        """Runs MILPExact using the Farkas y-polytope."""

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_y_constraints(threshold)

        if labels != None:
            var_groups = ProblemFormulation._var_groups_from_labels(
                reach_form, labels, mode="max")
        else:
            var_groups = InvertibleDict({})
            for sap_idx in range(C):
                (st,act) = reach_form.index_by_state_action.inv[sap_idx]
                #TODO change when InvertibleDict interface changes
                var_groups[st] = sap_idx

        milp_result = min_nonzero_groups(fark_matr,
                                        fark_rhs,
                                        var_groups,
                                        upper_bound=None,
                                        solver=self.solver)

        witness = Subsystem(reach_form, milp_result.result_vector)

        return ProblemResult(
            milp_result.status,witness,milp_result.value)


def min_nonzero_groups(matrix,
                      rhs,
                      var_groups,
                      upper_bound = None,
                      solver = "cbc"):
    C,N = matrix.shape

    min_nonzero_milp, indicator_var_to_vargroup_idx = ProblemFormulation._var_groups_program(
        matrix, rhs, var_groups, upper_bound, indicator_type="binary")

    milp_result = min_nonzero_milp.solve(solver)

    result_projected = ProblemFormulation._project_from_binary_indicators(
        milp_result.result_vector,
        N,
        var_groups,
        indicator_var_to_vargroup_idx)

    return SolverResult(milp_result.status,
                        result_projected,
                        milp_result.value)
