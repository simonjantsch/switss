from . import ProblemFormulation, ProblemResult, Subsystem, AllOnesInitializer
from switss.solver import SolverResult
from switss.utils import InvertibleDict

from bidict import bidict
import numpy as np

class MILPExact(ProblemFormulation):
    """
    MILPExact implements the computation of minimal witnessing subsystems using mixed integer linear programs (MILPs)
    over the corresponding Farkas-polytopes :math:`\mathcal{F}(\lambda) \in \{ \mathcal{P}^{\\text{max}}(\lambda),
    \mathcal{P}^{\\text{min}}(\lambda) \}`. Supported are both the minimization of systems while ignoring the state labels;    
    
    .. math::

       \min \sum_i \sigma(i) \quad \\text{ subject to } \quad \mathbf{x} \in \\mathcal{F}(\\lambda)  \quad
       \\text{ and }  \quad \mathbf{x}(i) \leq K \cdot \sigma(i),\; \sigma(i) \in \{0,1\},

    \- where :math:`K` is a suitable upper bound (see [FJB19]_ for more information) \-
    and label-based system minimization. In the second case, let :math:`L` be a set of labels and 
    :math:`\Lambda : S \mapsto 2^\mathcal{L}` a mapping from states to sets of labels. The MILP is then given as

    .. math::

        \min \sum_{l \in L} \sigma(l) \quad \\text{ subject to } \quad \mathbf{x} \in \mathcal{P}^{\\text{max}}(\\lambda) \quad 
        \\text{ and } \quad \mathbf{x}((s,a)) \leq K \cdot \sigma(l),\; \sigma(l) \in \{0,1\}, \\\\
        \\text{ for all }\quad (s,a) \in \mathcal{M},\; s \\not \in \{goal,fail\}\; l \in \Lambda(s)   

    for the y-form and as

    .. math::

        \min \sum_{l \in L} \sigma(l) \quad \\text{ subject to } \quad \mathbf{x} \in \mathcal{P}^{\\text{min}}(\\lambda) \quad 
        \\text{ and } \quad \mathbf{x}(s) \leq \sigma(l),\; \sigma(l) \in \{0,1\}, \\\\
        \\text{ for all }\quad s \in S \\backslash \{goal,fail\},\; l, \in \Lambda(s)   

    for the z-form. In both cases, :math:`\sigma` is a :math:`|L|`-dimensional vector.
    """
    def __init__(self, mode, solver="cbc"):
        """Instantiates a MILPExact instance from a given mode ("min" or "max") and a solver.

        :param mode: The mode, either "min" or "max"
        :type mode: str
        :param solver: Solver the should be used, defaults to "cbc"
        :type solver: str, optional
        """
        super().__init__()
        assert mode in ["min","max"]

        self.solver = solver
        self.mode = mode

    @property
    def details(self):
        """Returns a dictionary with method details. Keys are "type", "mode" and "solver"."""
        return {
            "type" : "MILPExact",
            "mode" : self.mode,
            "solver" : self.solver
        }

    def _solveiter(self, reach_form, threshold, labels, timeout=None):
        if self.mode == "min":
            return self._solve_min(reach_form, threshold, labels, timeout=timeout)
        else:
            return self._solve_max(reach_form, threshold, labels, timeout=timeout)

    def _solve_min(self, reach_form, threshold, labels, timeout=None):
        """Runs MILPExact using the Farkas z-polytope."""

        C,N = reach_form.system.P.shape

        fark_matr,fark_rhs = reach_form.fark_z_constraints(threshold)

        if labels == None:
            var_groups = InvertibleDict({i : set([i]) for i in range(N-2) })
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

    def _solve_max(self, reach_form, threshold, labels, timeout=None):
        """Runs MILPExact using the Farkas y-polytope."""

        C,N = reach_form.system.P.shape

        fark_matr,fark_rhs = reach_form.fark_y_constraints(threshold)

        if labels != None:
            var_groups = ProblemFormulation._var_groups_from_labels(
                reach_form, labels, mode="max")
        else:
            var_groups = InvertibleDict({})
            for sap_idx in range(C-2):
                (st,act) = reach_form.system.index_by_state_action.inv[sap_idx]
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
