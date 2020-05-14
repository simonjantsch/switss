from . import ProblemFormulation, ProblemResult, Subsystem, AllOnesInitializer
from switss.solver import SolverResult
from switss.utils import InvertibleDict

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

    @property
    def details(self):
        return {
            "type" : "MILPExact",
            "mode" : self.mode,
            "solver" : self.solver
        }

    def _solveiter(self, reach_form, threshold, labels, timeout=None):
        """Runs MILPExact using the Farkas (y- or z-) polytope
        depending on the value in mode."""
        if self.mode == "min":
            return self.solve_min(reach_form, threshold, labels, timeout=timeout)
        else:
            return self.solve_max(reach_form, threshold, labels, timeout=timeout)

    def solve_min(self, reach_form, threshold, labels, timeout=None):
        """Runs MILPExact using the Farkas z-polytope."""

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_z_constraints(threshold)

        if labels == None:
            var_groups = InvertibleDict({i : set([i]) for i in range(N) })
        else:
            var_groups = ProblemFormulation._var_groups_from_labels(
                reach_form,labels,"min")

        milp_result = MILPExact.__min_nonzero_groups(fark_matr,
                                                     fark_rhs,
                                                     var_groups,
                                                     upper_bound=1,
                                                     solver=self.solver,
                                                     timeout=timeout)

        if milp_result.status != "optimal":
            yield ProblemResult(milp_result.status,None,None,None)

        else:
            witness = Subsystem(reach_form, milp_result.result_vector, "min")
            yield ProblemResult(
                "success",witness,milp_result.value,milp_result.result_vector)

    def solve_max(self, reach_form, threshold, labels, timeout=None):
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
                var_groups.add(st, sap_idx)

        milp_result = MILPExact.__min_nonzero_groups(fark_matr,
                                                     fark_rhs,
                                                     var_groups,
                                                     upper_bound=None,
                                                     solver=self.solver,
                                                     timeout=timeout)

        if milp_result.status != "optimal":
            yield ProblemResult(milp_result.status,None,None,None)
        else:
            witness = Subsystem(reach_form, milp_result.result_vector, "max")

            yield ProblemResult(
                "success",witness,milp_result.value,milp_result.result_vector)

    @staticmethod
    def __min_nonzero_groups(matrix,
                             rhs,
                             var_groups,
                             upper_bound = None,
                             solver = "cbc",
                             timeout=None):
        C,N = matrix.shape

        min_nonzero_milp, indicator_var_to_vargroup_idx = ProblemFormulation._var_groups_program(
            matrix, rhs, var_groups, upper_bound, indicator_type="binary")

        if min_nonzero_milp == None:
            return SolverResult("infeasible",None,None)

        objective = AllOnesInitializer(indicator_var_to_vargroup_idx).initialize()
        min_nonzero_milp.set_objective_function(objective)
        milp_result = min_nonzero_milp.solve(solver,timeout=timeout)

        result_projected = ProblemFormulation._project_from_binary_indicators(
            milp_result.result_vector,
            N,
            var_groups,
            indicator_var_to_vargroup_idx)

        return SolverResult(milp_result.status,
                            result_projected,
                            milp_result.value)
