from . import ProblemFormulation, ProblemResult, MinimalWitness
from farkas.solver import LP
from .qsheurparams import AllOnesInitializer, InverseResultUpdater
import numpy as np

class QSHeur(ProblemFormulation):
    """The class QSHeur implements a class of iterative heuristics for computing small witnessing subsystems.
    Its goal is to find points in the corresponding Farkas-polytope with a small number of positive entries.
    It works as follows.
    Given a reachability form, let :math:`\\mathcal{F}(\\lambda)` be its Farkas (min- or max-)polytope for a given threshold :math:`\\lambda`.
    Then, the vector :math:`QS(i)` is an optimal solution of the LP:

    .. math::

       \min \mathbf{o}_i \cdot \mathbf{x} \quad \\text{ subj. to } \quad \mathbf{x} \in \\mathcal{F}(\\lambda)

    where :math:`\mathbf{o}_0` is a vector of initial weights (:math:`(1,\ldots,1)` is the default) and :math:`\mathbf{o}_i = \operatorname{upd}(QS(i-1))` for a given update function :math:`\operatorname{upd}` (where pointwise :math:`1/x` if :math:`x \\neq 0`, and a big constant otherwise, is the default). """
    def __init__(self,
                 threshold,
                 mode,
                 iterations = 3,
                 initializer = AllOnesInitializer(),
                 updater = InverseResultUpdater(),
                 solver_name="cbc"):
        super().__init__()
        assert mode in ["min","max"]
        assert solver_name in ["gurobi","cbc"]
        assert (threshold >= 0) and (threshold <= 1)

        self.mode = mode
        self.threshold = threshold
        self.iterations = iterations
        self.solver = solver_name
        self.updater = updater
        self.initializer = initializer

    def __repr__(self):
        return "QSHeur(threshold=%s, mode=%s, solver=%s, iterations=%s, initializer=%s, updater=%s)" % (
            self.threshold, self.mode, self.solver, self.iterations, self.initializer, self.updater)

    def solve(self, reach_form):
        """Runs the QSheuristic on the Farkas (min- or max-) polytope depending on the value in mode."""
        if self.mode == "min":
            return self.solve_min(reach_form)
        else:
            return self.solve_max(reach_form)

    def solve_min(self, reach_form):
        """Runs the QSheuristic on the Farkas min-polytope of a given reachability form for a given threshold."""
        C,N = reach_form.P.shape

        current_weights = self.initializer.initialize(reach_form, self.mode)

        # computes the constraints for the Farkas min-polytope of the given reachability form
        fark_matr,fark_rhs = reach_form.fark_min_constraints(self.threshold)
        print(fark_matr.shape, fark_rhs.shape, current_weights.shape,(C,N))

        # iteratively solves the corresponding LP, and computes the next objective function
        # from the result of the previous round according to the given update function
        for i in range(self.iterations):

            heur_i_lp = LP.from_coefficients(fark_matr,fark_rhs,current_weights)
            for idx in range(fark_matr.shape[1]):
                heur_i_lp.add_constraint([(idx,1)], ">=", 0)
                heur_i_lp.add_constraint([(idx,1)], "<=", 1)

            heur_i_result = heur_i_lp.solve(self.solver)
            
            if heur_i_result.status == "optimal":
                res_vector = heur_i_result.result
                res_vector = np.clip(res_vector, 0, 1)
                witness = MinimalWitness(reach_form, res_vector)

                yield ProblemResult("success", witness)
                
                current_weights = self.updater.update(heur_i_result.result, self.mode)
            else:
                # failed to optimize LP
                yield ProblemResult(heur_i_result.status, None)

    def solve_max(self, reach_form, initial_weights = None):
        """Runs the QSheuristic on the Farkas max-polytope of a given reachability form for a given threshold."""
        C,N = reach_form.P.shape

        current_weights = self.initializer.initialize(reach_form, self.mode)

        # computes the constraints for the Farkas max-polytope of the given reachability form
        fark_matr,fark_rhs = reach_form.fark_max_constraints(self.threshold)

        # iteratively solves the corresponding LP, and computes the next objective function
        # from the result of the previous round according to the given update function
        for i in range(0,self.iterations):
            # print(fark_matr.shape, fark_rhs.shape, current_weights.shape,(C,N))
            heur_i_lp = LP.from_coefficients(fark_matr,fark_rhs,current_weights)
            for idx in range(fark_matr.shape[1]):
                heur_i_lp.add_constraint([(idx,1)], ">=", 0)

            heur_i_result = heur_i_lp.solve(self.solver)
            
            if heur_i_result.status == "optimal":
                res_vector = heur_i_result.result
                res_vector = np.clip(res_vector, 0, 1)
                witness = MinimalWitness(reach_form, res_vector)

                yield ProblemResult("success", witness)

                current_weights = self.updater.update(heur_i_result.result, self.mode)

            else:
                # failed to optimize LP
                yield ProblemResult(heur_i_result.status, None)
