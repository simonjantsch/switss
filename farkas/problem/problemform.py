from abc import ABC, abstractmethod
from bidict import bidict
import numpy as np

from ..solver import MILP,LP
from ..utils import InvertibleDict

class ProblemFormulation:
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, reachability_form, threshold):
        pass

    @staticmethod
    def _compute_upper_bound(matr,rhs,lower_bound=0,solver="cbc"):
        C,N = matr.shape

        upper_obj = np.ones(N)
        upper_bound_LP = LP.from_coefficients(
            matr,rhs,upper_obj,objective="max")

        for idx in range(N):
            upper_bound_LP.add_constraint([(idx,1)],">=",0)

        lp_result = upper_bound_LP.solve(solver=solver)
        return lp_result.value

    @staticmethod
    def _var_groups_program(matr,
                            rhs,
                            var_groups,
                            upper_bound = None,
                            indicator_type="binary"):
        assert indicator_type in ["binary","real"]

        C,N = matr.shape

        if upper_bound == None:
            upper_bound = ProblemFormulation._compute_upper_bound(matr,rhs)

        if indicator_type == "binary":
            var_groups_program = MILP.from_coefficients(
                matr,rhs,np.zeros(N),["real"]*N,sense="<=",objective="min")
        else:
            var_groups_program = LP.from_coefficients(
                matr,rhs,np.zeros(N),sense="<=",objective="min")
        indicator_var_to_vargroup = bidict()
        objective_expr = []
        for (group,var_indices) in var_groups.items():
            indicator_var = var_groups_program.add_variables(
                *[indicator_type])
            indicator_var_to_vargroup[indicator_var] = group
            for var_idx in var_indices:
                if indicator_type != "binary":
                    var_groups_program.add_constraint(
                        [(indicator_var,1)],"<=",1)
                    var_groups_program.add_constraint(
                        [(indicator_var,1)],">=",0)
                objective_expr.append((indicator_var,1))
                var_groups_program.add_constraint(
                    [(var_idx,1),(indicator_var,-upper_bound)],"<=",0)

        for idx in range(N):
            var_groups_program.add_constraint([(idx,1)], ">=", 0)
            var_groups_program.add_constraint([(idx,1)], "<=", upper_bound)

        var_groups_program.set_objective_function(objective_expr)

        return var_groups_program,indicator_var_to_vargroup

    @staticmethod
    def _project_from_binary_indicators(result_vector,
                                        projected_length,
                                        var_groups,
                                        indicator_var_to_vargroup):
        result_projected = np.zeros(projected_length)
        handled_vars = dict()
        for (indicator,group) in indicator_var_to_vargroup.items():
            if result_vector[indicator] == 1:
                for var_idx in var_groups[group]:
                    result_projected[var_idx] = result_vector[var_idx]
                    handled_vars[var_idx] = True
            else:
                for var_idx in var_groups[group]:
                    result_projected[var_idx] = 0
                    handled_vars[var_idx] = True

        for n in range(projected_length):
            if n not in handled_vars.keys():
                result_projected[n] = result_vector[n]
            elif not handled_vars[n]:
                result_projected[n] = result_vector[n]

        return result_projected

    @staticmethod
    def _var_groups_from_labels(reach_form,labels,mode):
        assert mode in ["min","max"]

        C,N = reach_form.P.shape

        sys_st_by_label = reach_form.system.states_by_label
        min_labels = InvertibleDict(
            { l : sys_st_by_label["."+l] for l in labels})

        if mode == "min":
            return min_labels
        else:
            var_groups = InvertibleDict({})
            min_labels.inv
            for st_act_idx in range(C):
                (st,act) = reach_form.index_by_state_action.inv[st_act_idx]
                if st in min_labels.i.keys():
                    st_labels = min_labels.i[st]
                    for l in st_labels:
                        # if g not in var_groups.keys():
                        #     var_groups[g] = set()
                        # TODO change __set_item__ for InvertibleDict
                        var_groups[l] = st_act_idx
        return var_groups
