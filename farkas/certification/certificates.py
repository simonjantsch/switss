from farkas.solver import LP
import numpy as np

def generate_farkas_certificate(reach_form, mode, sense, threshold,solver="cbc"):
    """ Generates Farkas certificates for a given reachability form, mode, sense and threshold using the characterizations in Table 1 of [FJB19]_.
    To this end uses an LP solver to find a satisfying vector of the corresponding polytope.
    For strict inequalities, maximizes (minimizes) the probability and then checks whether the result value is strictly greater than the threshold.

    As the solvers that are currently supported do not use precise arithmetic the resulting vectors may deviate from the desired result slightly.
    """

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
            if lp_result.value > threshold:
                return lp_result.result_vector
            else:
                print("Property is not satisfied!")
                return None
        elif sense == "<":
            if lp_result.value < threshold:
                return lp_result.result_vector
            else:
                print("Property is not satisfied!")
                return None
        else:
            return lp_result.result_vector
    else:
        print("Property is not satisfied!")
        return None

def check_farkas_certificate(reach_form, mode, sense, threshold, farkas_vec, prec=5):
    """Given a reachability form, mode, sense, threshold and candidate vector, checks
    whether the vector is a legal Farkas certificate for the reachability constraint.

    To cope with the problem of imprecise LP-solver results, the function allows to
    define a precision :math:`prec` (the default value is 5).
    It will then round the vector pointwise to :math:`prec` decimals and check the
    certificate conditions afterwards.
    """

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
