from farkas.solver import LP, MILP
from farkas.utils import InvertibleDict

from bidict import bidict
import numpy as np

def __compute_upper_bound(matr,rhs,lower_bound=0,solver="cbc"):
    C,N = matr.shape

    upper_obj = np.ones(N)
    upper_bound_LP = LP.from_coefficients(
        matr,rhs,upper_obj,objective="max")

    for idx in range(N):
        upper_bound_LP.add_constraint([(idx,1)],">=",0)

    lp_result = upper_bound_LP.solve(solver=solver)
    return lp_result.value

def var_groups_program(matr,
                       rhs,
                       var_groups,
                       upper_bound = None,
                       indicator_type="binary"):
    assert indicator_type in ["binary","real"]

    C,N = matr.shape

    if upper_bound == None:
        upper_bound = __compute_upper_bound(matr,rhs)

    if indicator_type == "binary":
        var_groups_program = MILP.from_coefficients(
            matr,rhs,np.zeros(N),["real"]*N,sense="<=",objective="min")
    else:
        var_groups_program = LP.from_coefficients(
            matr,rhs,np.zeros(N),sense="<=",objective="min")

    indicator_var_to_vargroup_idx = bidict()
    objective_expr = []

    for (var_idx,group_idx) in var_groups.items():
        group_idx = next(iter(group_idx))
        if group_idx not in indicator_var_to_vargroup_idx.inv.keys():
            indicator_var = var_groups_program.add_variables(*[indicator_type])
            if indicator_type != "binary":
                var_groups_program.add_constraint([(indicator_var,1)],"<=",1)
                var_groups_program.add_constraint(
                    [(indicator_var,1)],">=",0)
            indicator_var_to_vargroup_idx[indicator_var] = group_idx
            objective_expr.append((indicator_var,1))
        var_groups_program.add_constraint(
            [(var_idx,1),(indicator_var,-upper_bound)],"<=",0)

    for idx in range(N):
        var_groups_program.add_constraint([(idx,1)], ">=", 0)
        var_groups_program.add_constraint([(idx,1)], "<=", upper_bound)

    var_groups_program.set_objective_function(objective_expr)

    return var_groups_program,indicator_var_to_vargroup_idx

def project_from_binary_indicators(result_vector,
                                   projected_length,
                                   var_groups,
                                   indicator_var_to_vargroup_idx):
    result_projected = np.zeros(projected_length)
    handled_vars = dict()
    group_idx_to_vars = var_groups.inv
    for (indicator,group_idx) in indicator_var_to_vargroup_idx.items():
        if result_vector[indicator] == 1:
            for var_idx in group_idx_to_vars[group_idx]:
                result_projected[var_idx] = result_vector[var_idx]
                handled_vars[var_idx] = True
        else:
            for var_idx in group_idx_to_vars[group_idx]:
                result_projected[var_idx] = 0
                handled_vars[var_idx] = True

    for n in range(projected_length):
        if n not in handled_vars.keys():
            result_projected[n] = result_vector[n]
        elif not handled_vars[n]:
            result_projected[n] = result_vector[n]

    return result_projected

def var_groups_from_state_groups(reach_form,state_groups,mode):
    assert mode in ["min","max"]

    C,N = reach_form.P.shape

    if mode == "min":
        if state_groups == None:
            return bidict({ i : i for i in range(N) })
        else:
            return state_groups
    else:
        var_groups = InvertibleDict({}, is_default=True)
        for st_act_idx in range(C):
            (state,action) = reach_form.index_by_state_action.inv[st_act_idx]
            if state_groups != None:
                if state in state_groups.keys():
                    var_groups[st_act_idx] = state_groups[state]
            else:
                var_groups[st_act_idx] = state
        return var_groups
