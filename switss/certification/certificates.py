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

    D = farkas_matr.shape[1]

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

    for idx in range(D):
        fark_lp.add_constraint([(idx,1)], ">=" ,0)

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

def check_farkas_certificate(reach_form, mode, sense, threshold, farkas_vec, tol=1e-8):
    """Given a reachability form, mode, sense, threshold and candidate vector, checks
    whether the vector is a legal Farkas certificate for the reachability constraint.

    To allow small deviations when checking the certificate conditions one can set the
    tol (for tolerance) parameter (defaults to 1e-8).
    It is then checked that any constraint deviates by at most the value in tol.
    """

    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(reach_form,mode,sense,threshold)

    N,D = farkas_matr.shape
    assert farkas_vec.shape[0] == D

    res_vec = farkas_matr.dot(farkas_vec)

    if sense == "<=":
        return (res_vec + tol >= rhs).all()
    elif sense == "<":
        return (res_vec + tol >= rhs).all() and (res_vec[N-1] + tol) > rhs[N-1]
    elif sense == ">=":
        return (res_vec - tol <= rhs).all()
    elif sense == ">":
        return (res_vec - tol <= rhs).all() and (res_vec[N-1] - tol) < rhs[N-1]

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
