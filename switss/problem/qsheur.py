from . import ProblemFormulation, ProblemResult, Subsystem
from . import AllOnesInitializer, InverseResultUpdater
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
                 mode,
                 iterations = 3,
                 initializertype = AllOnesInitializer,
                 updatertype = InverseResultUpdater,
                 solver="cbc"):
        """Instantiates a QSHeur from a given mode, a number of iterations and a initializer as well as 
        a updater.

        :param mode: Either "min" or "max"
        :type mode: str
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
        assert mode in ["min","max"]

        self.mode = mode
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
            "mode" : self.mode,
            "solver" : self.solver,
            "iterations" : self.iterations,
            "initializertype" : self.initializertype.__name__,
            "updatertype" : self.updatertype.__name__
        }

    def _solveiter(self, reach_form, threshold,labels,timeout=None):
        """Runs the QSheuristic using the Farkas (y- or z-) polytope
        depending on the value in mode."""
        if self.mode == "min":
            return self._solve_min(reach_form, threshold, labels,timeout=timeout)
        else:
            return self._solve_max(reach_form, threshold, labels,timeout=timeout)

    def _solve_min(self, reach_form, threshold, labels, timeout=None):
        """Runs the QSheuristic using the Farkas z-polytope of a given
        reachability form for a given threshold."""
        C,N = reach_form.system.P.shape

        fark_matr,fark_rhs = reach_form.fark_z_constraints(threshold)

        if labels is None:
            var_groups = InvertibleDict({ i : set([i]) for i in range(N-2)})
        else:
            var_groups = ProblemFormulation._var_groups_from_labels(reach_form, labels, "min")

        heur_lp, indicator_to_group = ProblemFormulation._var_groups_program(
            fark_matr,fark_rhs,var_groups,upper_bound=1,indicator_type="real")

        if heur_lp == None:
            yield ProblemResult("infeasible",None,None,None)
            return

        intitializer = self.initializertype(
            reachability_form=reach_form, mode=self.mode, indicator_to_group=indicator_to_group)
        updater = self.updatertype(reachability_form=reach_form, mode=self.mode, indicator_to_group=indicator_to_group)
        current_objective = intitializer.initialize()

        # iteratively solves the corresponding LP, and computes the next
        # objective function from the result of the previous round
        # according to the given update function
        for i in range(self.iterations):

            heur_lp.set_objective_function(current_objective)

            heur_result = heur_lp.solve(self.solver,timeout=timeout)

            if heur_result.status == "optimal":
                certificate = heur_result.result_vector[:N-2]
                witness = Subsystem(reach_form, certificate, "min")

                indicator_weights = heur_result.result_vector[N-2:]
                no_nonzero_groups = len([i for i in indicator_weights if i > 0])
                yield ProblemResult("success", witness, no_nonzero_groups, certificate)

                current_objective = updater.update(heur_result.result_vector)
            else:
                # failed to optimize LP
                yield ProblemResult(heur_result.status, None,None,None)
                break

    def _solve_max(self, reach_form, threshold, labels, timeout=None):
        """Runs the QSheuristic using the Farkas y-polytope of a given reachability form for a given threshold."""
        C,N = reach_form.system.P.shape

        fark_matr,fark_rhs = reach_form.fark_y_constraints(threshold)

        if labels is None:
            var_groups = InvertibleDict({ i : set([i]) for i in range(C-2)})
        else:
            var_groups = ProblemFormulation._var_groups_from_labels(reach_form,labels,"max")

        heur_lp, indicator_to_group = ProblemFormulation._var_groups_program(
            fark_matr,fark_rhs,var_groups,upper_bound=None,indicator_type="real")

        if heur_lp == None:
            yield ProblemResult("infeasible",None,None,None)
            return

        intitializer = self.initializertype(reachability_form=reach_form, mode=self.mode, indicator_to_group=indicator_to_group)
        updater = self.updatertype(reachability_form=reach_form, mode=self.mode, indicator_to_group=indicator_to_group)
        current_objective = intitializer.initialize()

        # iteratively solves the corresponding LP, and computes the
        # next objective function
        # from the result of the previous round according to the given
        # update function
        for i in range(0,self.iterations):
            heur_lp.set_objective_function(current_objective)

            heur_result = heur_lp.solve(self.solver,timeout=timeout)

            if heur_result.status == "optimal":
                # for the max-form, the resulting vector will be
                # (C-2)-dimensional, carrying values for state-action pairs.
                certificate = heur_result.result_vector[:C-2]
                witness = Subsystem(reach_form, certificate, "max")

                indicator_weights = heur_result.result_vector[C-2:]
                no_nonzero_groups = len([i for i in indicator_weights if i > 0])

                yield ProblemResult("success", witness, no_nonzero_groups, certificate)

                current_objective = updater.update(heur_result.result_vector)
            else:
                # failed to optimize LP
                yield ProblemResult(heur_result.status, None,None,None)
                break
