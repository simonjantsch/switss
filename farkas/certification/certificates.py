from farkas.problem import LP
import numpy as np

def generate_farkas_certificate(reach_form, mode, sense, threshold,solver="cbc"):
    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(mode,sense,threshold)

    D = rhs.shape[0]
    objective_fct = np.zeros(D)
    objective_fct[D] = -1

    if sense in ["<=","<"]:
        optimization_mode = "min"
    elif sense in [">=",">"]:
        optimization_mode = "max"

    fark_lp = LP.from_coefficients(farkas_matr,
                                   rhs,
                                   objective_fct,
                                   objective = optimization_mode)

    lp_result = fark_lp.solve(solver=solver)

    if lp_result.status == "optimal":
        if sense == ">":
            if lp_result.result_value > threshold:
                return lp_result.result_vector
            else:
                return None
        elif sense == "<":
            if lp_result.result_value < threshold:
                return lp_result.result_vector
            else:
                return None
        else:
            return lp_result.vector
    else:
        return None

def check_farkas_certificate(reach_form, mode, sense, threshold, farkas_vec):

    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(mode,sense,threshold)

    D = farkas_vec.shape[0]
    assert fark_matr.shape[1] == D

    res_vec = np.matmul(fark_matr,farkas_vec)

    if sense == "<=":
        return (res_vec >= rhs).all
    elif sense == "<":
        return (res_vec >= rhs).all and res_vec[D] > rhs[D]
    elif sense == ">=":
        return (res_vec <= rhs).all
    elif sense == ">":
        return (res_vec <= rhs).all and res_vec[D] < rhs[D]

    assert False


def __get_right_constraint_set(mode,sense,threshold):

    assert mode in ["min","max"]
    assert sense in ["<=","<",">=",">"]

    if mode == "min" and sense in ["<=","<"]:
        return reach_form.fark_z_constraints(threshold)

    elif mode == "max" and sense in [">=",">"]:
        return reach_form.fark_z_constraints(threshold)

    elif mode == "min" and sense in [">=",">"]:
        return reach_form.fark_y_constraints(threshold)

    elif mode == "max" and sense in ["<=","<"]
        return reach_form.fark_y_constraints(threshold)

    assert False
