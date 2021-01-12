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

       \min \sum_i \sigma(i) \; \\text{s.t.} \; \mathbf{x} \in \\mathcal{F}(\\lambda)  \;
       \\text{and}  \; \mathbf{x}(i) \leq K \cdot \sigma(i),\; \sigma(i) \in \{0,1\},

    \- where :math:`K` is a suitable upper bound (see [FJB19]_ for more information) \-
    and label-based system minimization. In the second case, let :math:`L` be a set of labels and 
    :math:`\Lambda : S \mapsto 2^\mathcal{L}` a mapping from states to sets of labels. The MILP is then given as

    .. math::

        \min \sum_{l \in L} \sigma(l) \; \\text{s.t.} \; \mathbf{x} \in \mathcal{P}^{\\text{max}}(\\lambda) \; 
        \\text{and} \; \mathbf{x}((s,a)) \leq K \cdot \sigma(l),\; \sigma(l) \in \{0,1\}, \\\\
        \\text{for all}\; (s,a) \in \mathcal{M},\; l \in \Lambda(s)   

    for the y-form and as

    .. math::

        \min \sum_{l \in L} \sigma(l) \; \\text{s.t.} \; \mathbf{x} \in \mathcal{P}^{\\text{min}}(\\lambda) \; 
        \\text{and} \; \mathbf{x}(s) \leq \sigma(l),\; \sigma(l) \in \{0,1\}, \\\\
        \\text{for all}\; s \in S,\; l \in \Lambda(s)

    for the z-form. In both cases, :math:`\sigma` is a :math:`|L|`-dimensional vector.
    """
    def __init__(self, solver="cbc"):
        """Instantiates a MILPExact instance from a given mode ("min" or "max") and a solver.

        :param mode: The mode, either "min" or "max"
        :type mode: str
        :param solver: Solver the should be used, defaults to "cbc"
        :type solver: str, optional
        """
        super().__init__()
        self.solver = solver

    @property
    def details(self):
        """Returns a dictionary with method details. Keys are "type", "mode" and "solver"."""
        return {
            "type" : "MILPExact",
            "solver" : self.solver
        }

    def _solveiter(self, reach_form, threshold, mode, groups, timeout=None, fixed_values=dict()):
        if mode == "min":
            return self._solve_min(reach_form, 
                                   threshold, 
                                   mode,
                                   groups, 
                                   timeout=timeout, 
                                   fixed_values=fixed_values)
        else:
            return self._solve_max(reach_form, 
                                   threshold,
                                   mode, 
                                   groups, 
                                   timeout=timeout, 
                                   fixed_values=fixed_values)

    def _solve_min(self, reach_form, threshold, mode, groups, timeout=None, fixed_values=dict()):
        """Runs MILPExact using the Farkas z-polytope."""
        fark_matr,fark_rhs = reach_form.fark_z_constraints(threshold)
        milp_result = MILPExact.__min_nonzero_groups(fark_matr,
                                                     fark_rhs,
                                                     groups,
                                                     upper_bound=1,
                                                     solver=self.solver,
                                                     timeout=timeout,
                                                     fixed_values=fixed_values)
        if milp_result.status != "optimal":
            yield ProblemResult(milp_result.status, None, None, None)
        else:
            witness = Subsystem(reach_form, milp_result.result_vector, "min")
            yield ProblemResult("success",
                                witness,
                                milp_result.value,
                                milp_result.result_vector)


    def _solve_max(self, reach_form, threshold, mode, groups, timeout=None, fixed_values=dict()):
        """Runs MILPExact using the Farkas y-polytope."""
        fark_matr,fark_rhs = reach_form.fark_y_constraints(threshold)
        milp_result = MILPExact.__min_nonzero_groups(fark_matr,
                                                     fark_rhs,
                                                     groups,
                                                     upper_bound=None,
                                                     solver=self.solver,
                                                     timeout=timeout,
                                                     fixed_values=fixed_values)
        if milp_result.status != "optimal":
            yield ProblemResult(milp_result.status, None, None, None)
        else:
            witness = Subsystem(reach_form, milp_result.result_vector, "max")
            yield ProblemResult("success",
                                witness,
                                milp_result.value,
                                milp_result.result_vector)

    @staticmethod
    def __min_nonzero_groups(matrix,
                             rhs,
                             groups,
                             upper_bound = None,
                             solver = "cbc",
                             timeout=None,
                             fixed_values=dict()):
        C,N = matrix.shape
        min_nonzero_milp, indicator_var_to_vargroup_idx = ProblemFormulation._groups_program(
            matrix, 
            rhs, 
            groups, 
            upper_bound, 
            indicator_type="binary", 
            fixed_values=fixed_values)

        if min_nonzero_milp is None:
            return SolverResult("infeasible", None, None)

        objective = AllOnesInitializer(indicator_var_to_vargroup_idx).initialize()
        min_nonzero_milp.set_objective_function(objective)
        milp_result = min_nonzero_milp.solve(solver,timeout=timeout)

        result_projected = ProblemFormulation._project_from_binary_indicators(
            milp_result.result_vector,
            N,
            indicator_var_to_vargroup_idx)

        return SolverResult(milp_result.status,
                            result_projected,
                            milp_result.value)
