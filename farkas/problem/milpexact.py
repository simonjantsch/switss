from . import ProblemFormulation, ProblemResult, Subsystem
from farkas.solver import LP, MILP, SolverResult

from bidict import bidict
import numpy as np


class MILPExact(ProblemFormulation):
    def __init__(self, threshold, mode, solver):
        super().__init__()
        assert (threshold >= 0) and (threshold <= 1)
        assert mode in ["min","max"]

        self.solver = solver
        self.threshold = threshold
        self.mode = mode

    def solve(self, reach_form):
        if self.mode == "min":
            return self.solve_min(reach_form)
        else:
            return self.solve_max(reach_form)

    def solve_min(self, reach_form):

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_min_constraints(self.threshold)

        var_groups = dict([(i,[i]) for i in range(N)])
        milp_result = min_nonzero_entries(fark_matr,fark_rhs,var_groups,upper_bound=1,solver=self.solver)

        # this creates a new C-dimensional vector which carries values for state-action pairs. 
        # every state-action pair is assigned the weight the state has.
        # this "blowing-up" will make it easier for visualizing subsystems.
        state_action_weights = np.zeros(C)
        for idx in range(C):
            state,_ = reach_form.index_by_state_action.inv[idx]
            state_action_weights[idx] = milp_result.result_vector[state]

        return Subsystem(reach_form, milp_result.result_vector)

    def solve_max(self, reach_form):

        C,N = reach_form.P.shape

        fark_matr,fark_rhs = reach_form.fark_max_constraints(self.threshold)

        var_groups = dict([(i,[i]) for i in range(C)])
        milp_result = min_nonzero_entries(fark_matr,fark_rhs,var_groups,upper_bound=None,solver=self.solver)

        return Subsystem(reach_form, milp_result.result_vector)


    def __repr__(self):
        return "MILPExact(solver=%s)" % (self.solver)


def min_nonzero_entries(matrix,
                        rhs,
                        var_groups,
                        upper_bound = None,
                        solver = "cbc"):
    C,N = matrix.shape

    # TODO assertion on var_groups and rhs

    min_nonzero_milp = MILP.from_coefficients(matrix,rhs,np.zeros(N),["real"]*C,objective="min")
    objective_expr = []

    if upper_bound == None:
        upper_obj = np.ones(N)
        upper_bound_LP = LP.from_coefficients(matrix,rhs,upper_obj,objective="max")
        lp_result = upper_bound_LP.solve(solver=solver)
        upper_bound = lp_result.result_value

    for idx in range(N):
        min_nonzero_milp.add_constraint([(idx,1)], ">=", 0)
        min_nonzero_milp.add_constraint([(idx,1)], "<=", upper_bound)

    indicator_var_to_vargroup_idx = bidict()
    for (group_idx, var_indices) in var_groups.items():
        indicator_var = min_nonzero_milp.add_variables(*["binary"])
        indicator_var_to_vargroup_idx[indicator_var] = group_idx
        objective_expr.append((indicator_var,1))
        for var_idx in var_indices:
            min_nonzero_milp.add_constraint([(var_idx,1),(indicator_var,-upper_bound)],"<=",0)

    min_nonzero_milp.set_objective_function(objective_expr)

    milp_result = min_nonzero_milp.solve(solver)

    result_projected = np.zeros(N)
    handled_vars = dict()
    for (indicator, group_idx) in indicator_var_to_vargroup_idx.items():
        if milp_result.result_vector[indicator] == 1:
            for var_idx in var_groups[group_idx]:
                result_projected[var_idx] = milp_result.result_vector[var_idx]
                handled_vars[var_idx] = True
        else:
            for var_idx in var_groups[group_idx]:
                result_projected[var_idx] = 0
                handled_vars[var_idx] = True

    for n in range(N):
        if not handled_vars[n]:
            result_projected[n] = milp_result.result_vector[n]

    return SolverResult(milp_result.status, result_projected, milp_result.result_value)
