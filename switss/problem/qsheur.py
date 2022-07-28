from . import ProblemFormulation, ProblemResult, Subsystem
from . import AllOnesInitializer, InverseResultUpdater, construct_MILP, certificate_size
from switss.utils import InvertibleDict
from switss.solver import LP
import numpy as np

class QSHeur(ProblemFormulation):
    """The class QSHeur implements a set of iterative heuristics for
    computing small witnessing subsystems. Its goal is to find points in 
    the corresponding Farkas-polytope with a small number of positive entries.
    It works by solving a sequence of LPs similar to the one in MILPExact.
    The :math:`i`-th LP is given as
    
    .. math::

       \min \mathbf{o}_i \cdot \mathbf{x} \; \\text{s.t.} \; \mathbf{x} \in \\mathcal{F}(\\lambda),
    
    where :math:`\\mathcal{F}(\\lambda)` is the Farkas (y- or z-)polytope. Also, 
    :math:`\mathbf{o}_0` is a vector of initial weights. Weights :math:`\mathbf{o}_{i}` for
    :math:`i>0` are computed recursively from the past result: :math:`\mathbf{o}_{i} =
    \operatorname{upd}(QS_{\mathbf{x}}(i-1))` where :math:`QS_{\mathbf{x}}(i-1)` denotes the solution
    to the :math:`(i-1)`-th LP and :math:`\operatorname{upd}` is some update function. 

    QSHeur also supports label-based minimizaton. In that case, let :math:`L` be a set of labels and 
    :math:`\Lambda : S \mapsto 2^\mathcal{L}` a mapping from states to sets of labels. The :math:`i`-th LP 
    is then given as

    .. math::

        \min \mathbf{o}_i \cdot \sigma \; \\text{s.t.} \; \mathbf{x} \in \mathcal{P}^{\\text{max}}(\\lambda) \;
        \\text{and} \; \mathbf{x}((s,a)) \leq K \cdot \sigma(l) \;
        \\text{for all}\; (s,a) \in \mathcal{M},\; l \in \Lambda(s)    

    for the y-form and as

    .. math::

        \min \mathbf{o}_i \cdot \sigma \; \\text{s.t.} \; \mathbf{x} \in \mathcal{P}^{\\text{min}}(\\lambda) \; 
        \\text{and} \; \mathbf{x}(s) \leq \sigma(l),\;
        \\text{for all}\; s \in S,\; l \in \Lambda(s) 

    for the z-form.
    """
    def __init__(self,
                 iterations = 3,
                 initializertype = AllOnesInitializer,
                 updatertype = InverseResultUpdater,
                 solver="cbc"):
        """Instantiates a QSHeur from a given mode, a number of iterations and a initializer as well as 
        a updater.

        :param iterations: Number of repeated LP instances, defaults to 3
        :type iterations: int, optional
        :param initializertype: The used initialization-method, defaults to AllOnesInitializer
        :type initializertype: problem.Initializer, optional
        :param updatertype: The used update-method, defaults to InverseResultUpdater
        :type updatertype: problem.Updater, optional
        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        """        
        super().__init__()

        self.iterations = iterations
        self.solver = solver
        self.updatertype = updatertype
        self.initializertype = initializertype

    @property
    def details(self):
        """Returns a dictionary with method details. Keys are "type", "mode", "solver", "iterations", "initializertype"
        and "updatertype"."""
        return {
            "type" : "QSHeur",
            "solver" : self.solver,
            "iterations" : self.iterations,
            "initializertype" : self.initializertype.__name__,
            "updatertype" : self.updatertype.__name__
        }

    def _solveiter(self, reach_form, threshold, mode, labels, timeout=None):
        """Runs the QSheuristic using the Farkas (y- or z-) polytope
        depending on the value in mode."""
        if self.solver == "gurobi":
            modeltype_str = "gurobi"
        else:
            modeltype_str = "pulp"

        model, indicators = construct_MILP(reach_form, 
                                           threshold, 
                                           mode=mode, 
                                           labels=labels, 
                                           relaxed=True, 
                                           upper_bound_solver="cbc",
                                           modeltype_str=modeltype_str)
        if model is None:
            yield ProblemResult("infeasible", None, None, None)
            return

        certsize = certificate_size(reach_form, mode)
        initializer = self.initializertype(reachability_form=reach_form, mode=mode, indicator_to_group=indicators)
        updater = self.updatertype(reachability_form=reach_form, mode=mode, indicator_to_group=indicators)
        current_objective = initializer.initialize()

        for i in range(self.iterations):
            model.set_objective_function(current_objective)
            result = model.solve(self.solver, timeout=timeout)

            if result.status == "optimal":
                certificate = result.result_vector[:certsize]
                witness = Subsystem(reach_form, certificate, mode)
                indicator_weights = result.result_vector[certsize:]
                no_nonzero_groups = len([i for i in indicator_weights if i > 0])
                yield ProblemResult("success", witness, no_nonzero_groups, certificate)

                current_objective = updater.update(result.result_vector)
                new_constraints = updater.constraints(result.result_vector)
                for constraint in new_constraints:
                    model.add_constraint(*constraint)
            else:
                # failed to optimize LP
                yield ProblemResult(result.status, None, None, None)
                break
