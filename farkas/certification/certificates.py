from farkas.solver import LP
import numpy as np

def generate_farkas_certificate(reach_form, mode, sense, threshold,solver="cbc"):
    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(reach_form,mode,sense,threshold)

    D = rhs.shape[0]

    if sense in ["<=","<"]:
        optimization_mode = "min"
        lp_sense = ">="
        if mode == "min":
            objective_fct = reach_form.to_target
        elif mode == "max":
            objective_fct = np.zeros(D)
            objective_fct[reach_form.initial] = 1
    elif sense in [">=",">"]:
        optimization_mode = "max"
        lp_sense = "<="
        if mode == "max":
            objective_fct = reach_form.to_target
        elif mode == "min":
            objective_fct = np.zeros(D)
            objective_fct[reach_form.initial] = 1

    fark_lp = LP.from_coefficients(farkas_matr,
                                   rhs,
                                   objective_fct,
                                   sense = lp_sense,
                                   objective = optimization_mode)

    lp_result = fark_lp.solve(solver=solver)

    if lp_result.status == "optimal":
        if sense == ">":
            if lp_result.result_value > threshold:
                return lp_result.result_vector
            else:
                print("Property is not satisfied!")
                return None
        elif sense == "<":
            if lp_result.result_value < threshold:
                return lp_result.result_vector
            else:
                print("Property is not satisfied!")
                return None
        else:
            return lp_result.result_vector
    else:
        print("Property is not satisfied!")
        return None

def check_farkas_certificate(reach_form, mode, sense, threshold, farkas_vec, prec=9):

    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(reach_form,mode,sense,threshold)

    D = farkas_vec.shape[0]
    assert farkas_matr.shape[1] == D

    res_vec = farkas_matr.dot(farkas_vec)

    res_vec_rounded = np.round(res_vec,prec)

    if sense == "<=":
        return (res_vec_rounded >= rhs).all()
    elif sense == "<":
        return (res_vec_rounded >= rhs).all() and res_vec_rounded[D] > rhs[D]
    elif sense == ">=":
        return (res_vec_rounded <= rhs).all()
    elif sense == ">":
        return (res_vec_rounded <= rhs).all() and res_vec_rounded[D] < rhs[D]

    assert False


def __get_right_constraint_set(reach_form,mode,sense,threshold):

    assert mode in ["min","max"]
    assert sense in ["<=","<",">=",">"]

    if mode == "min" and sense in ["<=","<"]:
        return reach_form.fark_y_constraints(threshold)

    elif mode == "max" and sense in [">=",">"]:
        return reach_form.fark_y_constraints(threshold)

    elif mode == "min" and sense in [">=",">"]:
        return reach_form.fark_z_constraints(threshold)

    elif mode == "max" and sense in ["<=","<"]:
        return reach_form.fark_z_constraints(threshold)

    assert False
