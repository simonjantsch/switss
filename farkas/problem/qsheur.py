from . import ProblemFormulation, ProblemResult, Subsystem, var_groups_program, var_groups_from_state_groups
from farkas.solver import LP
from .qsheurparams import AllOnesInitializer, InverseResultUpdater
import numpy as np

class QSHeur(ProblemFormulation):
    """The class QSHeur implements a class of iterative heuristics for
    computing small witnessing subsystems.
    Its goal is to find points in the corresponding Farkas-polytope
    with a small number of positive entries.
    It works using a similar linear program to the one in MILPExact.
    Let :math:`\\mathcal{F}(\\lambda)` be the Farkas (y- or z-)polytope
    for a threshold :math:`\\lambda` of some reachability form.
    The i-th solution is the vector :math:`QS_{\mathbf{x}}(i)`, namely
    the :math:`\mathbf{x}` part of an optimal solution of the LP:

    .. math::

       \min \mathbf{o}_i \cdot \sigma \;\; \\text{ subj. to } \quad \mathbf{x} \in \\mathcal{F}(\\lambda) \;\; \\text{ and }  \;\; \mathbf{x}(i) \leq K \cdot \sigma(g(i))

    where :math:`\mathbf{o}_0` is a vector of initial weights
    (:math:`(1,\ldots,1)` is the default) and
    :math:`\mathbf{o}_i = \operatorname{upd}(QS_{\sigma}(i-1))` for a given
    update function :math:`\operatorname{upd}` (where pointwise :math:`1/x`
    if :math:`x \\neq 0`, and a big constant otherwise, is the default).
    The vector :math:`QS_{\sigma}(i-1)` is the :math:`\sigma` part of the
    :math:`i-1`-th iteration of the heuristic.

    The function :math:`g` is specified using the state_groups parameter
    and is used to "group" states as in MILPExact.
    The default setting :math:`g = id` results in the standard minimization
    where all states are considered equally.
    """
    def __init__(self,
                 threshold,
                 mode,
                 iterations = 3,
                 state_groups = None,
                 initializer = AllOnesInitializer(),
                 updater = InverseResultUpdater(),
                 solver_name="cbc"):
        super().__init__()
        assert mode in ["min","max"]
        assert (threshold >= 0) and (threshold <= 1)

        self.mode = mode
        self.threshold = threshold
        self.iterations = iterations
        self.solver = solver_name
        self.updater = updater
        self.initializer = initializer
        self.state_groups = state_groups

    def __repr__(self):
        return "QSHeur(threshold=%s, mode=%s, solver=%s, iterations=%s, initializer=%s, updater=%s)" % (
            self.threshold, self.mode, self.solver, self.iterations, self.initializer, self.updater)

    def solve(self, reach_form):
        """Runs the QSheuristic using the Farkas (y- or z-) polytope
        depending on the value in mode."""
        if self.mode == "min":
            return self.solve_min(reach_form)
        else:
            return self.solve_max(reach_form)

    def solve_min(self, reach_form):
        """Runs the QSheuristic using the Farkas z-polytope of a given
        reachability form for a given threshold."""
        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_z_constraints(self.threshold)

        var_groups = var_groups_from_state_groups(
            reach_form,self.state_groups,"min")

        heur_lp, ind_to_grp_idx = var_groups_program(fark_matr,
                                                     fark_rhs,
                                                     var_groups,
                                                     upper_bound=1,
                                                     indicator_type="real")

        indicator_idx = ind_to_grp_idx.keys()

        current_objective = self.initializer.initialize(
            indicator_idx)

        # iteratively solves the corresponding LP, and computes the next
        # objective function from the result of the previous round
        # according to the given update function
        for i in range(self.iterations):

            heur_lp.set_objective_function(current_objective)

            heur_result = heur_lp.solve(self.solver)

            if heur_result.status == "optimal":
                state_weights = heur_result.result_vector[:N]

                # this creates a new C-dimensional vector which carries
                # values for state-action pairs.
                # every state-action pair is assigned the weight the state
                # has.
                # this "blowing-up" will make visualizing subsystems easier.
                state_action_weights = np.zeros(C)
                for idx in range(C):
                    state,_ = reach_form.index_by_state_action.inv[idx]
                    state_action_weights[idx] = state_weights[state]

                witness = Subsystem(reach_form, state_action_weights)

                indicator_weights = heur_result.result_vector[N:]
                no_nonzero_groups = len(
                    [i for i in indicator_weights if i > 0])
                yield ProblemResult("success", witness, no_nonzero_groups)

                current_objective = self.updater.update(
                    heur_result.result_vector,
                    indicator_idx)
            else:
                # failed to optimize LP
                yield ProblemResult(heur_result.status, None,None)

    def solve_max(self, reach_form, initial_weights = None):
        """Runs the QSheuristic using the Farkas y-polytope of a given reachability form for a given threshold."""
        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_y_constraints(self.threshold)

        var_groups = var_groups_from_state_groups(reach_form,self.state_groups,"max")

        heur_lp, ind_to_grp_idx = var_groups_program(fark_matr,
                                                     fark_rhs,
                                                     var_groups,
                                                     upper_bound=None,
                                                     indicator_type="real")

        indicator_idx = ind_to_grp_idx.keys()
        current_objective = self.initializer.initialize(
            indicator_idx)

        # iteratively solves the corresponding LP, and computes the
        # next objective function
        # from the result of the previous round according to the given
        # update function
        for i in range(0,self.iterations):
            heur_lp.set_objective_function(current_objective)

            heur_result = heur_lp.solve(self.solver)

            if heur_result.status == "optimal":
                # for the max-form, the resulting vector will be
                # C-dimensional, carrying values for state-action pairs.
                state_action_weights = heur_result.result_vector[:C]
                witness = Subsystem(reach_form, state_action_weights)

                indicator_weights = heur_result.result_vector[N:]
                no_nonzero_groups = len(
                    [i for i in indicator_weights if i > 0])

                yield ProblemResult("success", witness, no_nonzero_groups)

                current_objective = self.updater.update(
                    heur_result.result_vector,
                    indicator_idx)

            else:
                # failed to optimize LP
                yield ProblemResult(heur_result.status, None,None)
