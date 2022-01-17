from switss.solver import LP
from scipy.sparse import dok_matrix, identity
import numpy as np

def find_interior_point(A, b, xgeq0=False, zero_vars = [], solver="cbc"):
    """Finds a point :math:`x` that is (strictly) inside a convex region (made up of half-spaces), i.e. a vector :math:`x \in \mathbb{R}^n` such that :math:`A x \leq b` if there are any. The algorithm finds the solution to the problem 

    .. :math:

        \min_{(x,s)} s \\text{ subject to } Ax \leq b + s

    :param A: :math:`N \\times M` matrix :math:`A`
    :type A: scipy.sparse.dok_matrix
    :param b: :math:`N` vector :math:`b`
    :type b: np.ndarray
    :param xgeq0: If true, adds additional constraints :math:`x \geq 0`, defaults to False
    :type xgeq0: bool, optional
    :param zero_vars: A list of variables which should be forced to have value zero.
    :type zero_vars: [int], optional
    :param solver: the used solver, may be "gurobi", "cbc", "cplex" or "glpk", defaults to "cbc"
    :type solver: str, optional
    :return: a 3d-tuple (x, optimal, strict) where `x` is the :math:`M`-d result vector :math:`x^*`, `optimal` indicates whether the LP-solution is optimal (i.e. not unbounded or infeasible) and `strict` whether :math:`A x^* < b` is satisfied.
    :rtype: Tuple[np.ndarray, Bool, Bool]
    """
    # A' = [ a11 ... a1M -1 
    #        ...
    #        aN1     aNM -1 ] is a Nx(M+1) matrix
    # x' = (x1 ... xM s)
    # b' = b
    A_ = dok_matrix(A.copy())
    b_ = b.copy()
    A_.resize((A.shape[0], A.shape[1]+1))
    A_[:,-1] = -1

    opt = np.zeros(A_.shape[1])
    opt[-1] = 1

    lp = LP.from_coefficients(A_, b_, opt,"<=","min")
    if xgeq0:
        for idx in range(A.shape[1]):
            lp.add_constraint([(idx,1),(A.shape[1],1)],">=",0)

    for idx in zero_vars:
        lp.add_constraint([(idx,1)],"=",0)

    result = lp.solve(solver)
    sres = result.result_vector[-1]

    return result.result_vector[:-1], (result.status == "optimal" and sres <= 0), sres < 0

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

def check_farkas_certificate(reach_form, mode, sense, threshold, farkas_vec, tol=1e-8):
    """Given a reachability form, mode, sense, threshold and candidate vector, checks
    whether the vector is a legal Farkas certificate for the reachability constraint.

    To allow small deviations when checking the certificate conditions one can set the
    tol (for tolerance) parameter (defaults to 1e-8). It is then checked that any constraint 
    deviates by at most the value in tol.

    :param reach_form: RF the certificate should be checked for
    :type reach_form: model.ReachabilityForm
    :param mode: either "min" or "max"
    :type mode: str
    :param sense: either "<=", ">=", "<" or ">"
    :type sense: str
    :param threshold: The threshold that should be used
    :type threshold: float
    :param farkas_vec: :math:`N` or :math:`C` dimensional certificate vector, dependent on mode
    :type farkas_vec: np.ndarray[float]
    :param tol: The used tolerance, defaults to 1e-8
    :type tol: float, optional
    :return: Either True (certificate is valid for used tolerance) or False (that is not the case)
    :rtype: bool
    """       
    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(reach_form,mode,sense,threshold)

    N,D = farkas_matr.shape
    assert farkas_vec.shape[0] == D

    if mode == "min":
        N, D = farkas_matr.shape
        if sense in [">=",">"]:
            for idx in range(D):
                if reach_form.in_proper_ec(idx) and not (farkas_vec[idx] == 0):
                    return False

        else:
            for idx in range(D):
                (s,a) = reach_form.index_by_state_action.inv[idx]
                if reach_form.in_proper_ec(s) and not (farkas_vec[idx] == 0):
                    return False

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

def generate_farkas_certificate(reach_form, mode, sense, threshold,tol=1e-5,solver="cbc"):
    """Generates Farkas certificates for a given reachability form, mode, sense and threshold using the characterizations in Table 1 of [FJB19]_. To this end uses an LP solver to find a satisfying vector of the corresponding polytope.

    :param reach_form: RF the certificate should be generated for
    :type reach_form: model.ReachabilityForm
    :param mode: must be either "min" or "max"
    :type mode: str
    :param sense: must be either "<=", ">=", "<" or ">".
    :type sense: str 
    :param threshold: threshold the certificate should be generated for
    :type threshold: float
    :param tol: the tolerance the certificate should be checked for
    :type tol: float
    :param solver: used solver, must be either "gurobi", "cbc", "glpk" or "cplex", defaults to "cbc"
    :type solver: str, optional
    :return: :math:`N` or :math:`C` dimensional vector, dependent on mode
    :rtype: numpy.ndarray[float]
    """    

    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(reach_form,mode,sense,threshold)
    if sense in ["<=", "<"]:
        # if sense is "<=" or "<", then the certificate condition is Ax >=/> b
        # therefore, apply the transformation: Ax >=/> b <=> (-A)x <=/< -b
        farkas_matr, rhs = -farkas_matr, -rhs

    #for "min"-problems all variables corresponding to states in proper end components have to be set to zero
    zero_vars = []
    if mode == "min":
        N, D = farkas_matr.shape
        if sense in [">=",">"]:
            for idx in range(D):
                if reach_form.in_proper_ec(idx):
                    zero_vars.append(idx)
        else:
            for idx in range(D):
                (s,a) = reach_form.index_by_state_action.inv[idx]
                if reach_form.in_proper_ec(s):
                    zero_vars.append(s)

    lp_result, optimal, is_strict = find_interior_point(farkas_matr,rhs,xgeq0=True,zero_vars=zero_vars,solver=solver)
    
    check = check_farkas_certificate(reach_form, mode, sense, threshold, lp_result,tol=tol)
    return lp_result if check else None
