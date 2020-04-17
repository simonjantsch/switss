from . import ProblemFormulation, ProblemResult, Subsystem
from farkas.solver import LP, MILP, SolverResult
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
    mapping from the variables into a finite set of groups,
    :math:`\mathcal{F}(\lambda)` is the Farkas (y- or z-)polytope for
    threshold :math:`\lambda` and :math:`K` is a upper bound on the variables
    :math:`\mathbf{x}` in :math:`\mathcal{F}(\lambda)`.

    It follows that in any solution :math:`\sigma(G)` is one iff one of the variables :math:`\mathbf{x}(i)` such that :math:`g(i) = G` is strictly positive.
    """
    def __init__(self, threshold, mode, solver):
        super().__init__()
        assert (threshold >= 0) and (threshold <= 1)
        assert mode in ["min","max"]

        self.solver = solver
        self.threshold = threshold
        self.mode = mode

    def __repr__(self):
        return "MILPExact(threshold=%s, mode=%s, solver=%s)" % (
            self.threshold, self.mode, self.solver)

    def solve(self, reach_form, state_groups=None):
        if self.mode == "min":
            return self.solve_min(reach_form,state_groups)
        else:
            return self.solve_max(reach_form,state_groups)

    def solve_min(self, reach_form,state_groups=None):

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_z_constraints(self.threshold)

        if state_groups == None:
            var_groups = InvertibleDict(dict([(i,i) for i in range(N)]))
        else:
            var_groups = state_groups

        milp_result = min_nonzero_entries(fark_matr,fark_rhs,var_groups,upper_bound=1,solver=self.solver)

        # this creates a new C-dimensional vector which carries values for state-action pairs.
        # every state-action pair is assigned the weight the state has.
        # this "blowing-up" will make it easier for visualizing subsystems.
        state_action_weights = np.zeros(C)
        for idx in range(C):
            state,_ = reach_form.index_by_state_action.inv[idx]
            state_action_weights[idx] = milp_result.result_vector[state]

        return Subsystem(reach_form, state_action_weights)

    def solve_max(self, reach_form,state_groups=None):

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_y_constraints(self.threshold)

        var_groups = InvertibleDict(dict())

        for st_act_idx in range(C):
            (state,action) = reach_form.index_by_state_action.inv[st_act_idx]
            if state_groups != None:
                if state in state_groups.keys():
                    var_groups[st_act_idx] = state_groups[state]
            else:
                var_groups[st_act_idx] = state

        milp_result = min_nonzero_entries(fark_matr,fark_rhs,var_groups,upper_bound=None,solver=self.solver)

        return Subsystem(reach_form, milp_result.result_vector)


def min_nonzero_entries(matrix,
                        rhs,
                        var_groups,
                        upper_bound = None,
                        solver = "cbc"):
    C,N = matrix.shape

    # TODO assertion on var_groups and rhs

    print(var_groups)

    min_nonzero_milp = MILP.from_coefficients(matrix,rhs,np.zeros(N),["real"]*N,objective="min")
    objective_expr = []

    if upper_bound == None:
        upper_obj = np.ones(N)
        upper_bound_LP = LP.from_coefficients(matrix,rhs,upper_obj,objective="max")
        for idx in range(N):
            upper_bound_LP.add_constraint([(idx,1)],">=",0)
        lp_result = upper_bound_LP.solve(solver=solver)
        upper_bound = lp_result.result_value

    for idx in range(N):
        min_nonzero_milp.add_constraint([(idx,1)], ">=", 0)
        min_nonzero_milp.add_constraint([(idx,1)], "<=", upper_bound)

    indicator_var_to_vargroup_idx = bidict()
    for (var_idx,group_idx) in var_groups.items():
        if group_idx not in indicator_var_to_vargroup_idx.values():
            indicator_var = min_nonzero_milp.add_variables(*["binary"])
            indicator_var_to_vargroup_idx[indicator_var] = group_idx
            objective_expr.append((indicator_var,1))
        min_nonzero_milp.add_constraint([(var_idx,1),(indicator_var,-upper_bound)],"<=",0)

    min_nonzero_milp.set_objective_function(objective_expr)

    milp_result = min_nonzero_milp.solve(solver)

    result_projected = np.zeros(N)
    handled_vars = dict()
    group_idx_to_vars = var_groups.inv
    for (indicator,group_idx) in indicator_var_to_vargroup_idx.items():
        if milp_result.result_vector[indicator] == 1:
            for var_idx in group_idx_to_vars[group_idx]:
                result_projected[var_idx] = milp_result.result_vector[var_idx]
                handled_vars[var_idx] = True
        else:
            for var_idx in group_idx_to_vars[group_idx]:
                result_projected[var_idx] = 0
                handled_vars[var_idx] = True

    for n in range(N):
        if n not in handled_vars.keys():
            result_projected[n] = milp_result.result_vector[n]
        elif not handled_vars[n]:
            result_projected[n] = milp_result.result_vector[n]

    return SolverResult(milp_result.status, result_projected, milp_result.result_value)
