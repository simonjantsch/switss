from . import ProblemFormulation, ProblemResult, Subsystem, AllOnesInitializer, construct_MILP, certificate_size
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

    def _solveiter(self, reach_form, threshold, mode, labels, timeout=None):
        model, _ = construct_MILP(reach_form, 
                                  threshold, 
                                  mode=mode, 
                                  labels=labels, 
                                  relaxed=False, 
                                  upper_bound_solver="cbc",
                                  modeltype_str="gurobi" if self.solver=="gurobi" else "pulp")

        if model is None:
            yield ProblemResult("infeasible", None, None, None)
        else:
            result = model.solve(solver=self.solver, timeout=timeout)
            if result.status != "optimal":
                yield ProblemResult(result.status, None, None, None)
            else:
                certsize = certificate_size(reach_form, mode)
                certificate = result.result_vector[:certsize]
                witness = Subsystem(reach_form, certificate, mode)
                yield ProblemResult("success", witness, result.value, certificate)
