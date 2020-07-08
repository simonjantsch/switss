from switss.solver import LP
from scipy.sparse import dok_matrix, identity
import numpy as np

def find_interior_point(A, b, xgeq0=False, solver="cbc"):
    """Finds a point :math:`x` that is (strictly) inside a convex region (made up of half-spaces), i.e. a vector :math:`x \in \mathbb{R}^n` such that :math:`A x \leq b` if there are any. The algorithm finds the solution to the problem 

    .. :math:

        \min_{(x,s)} s \\text{ subject to } Ax \leq b + s

    :param A: :math:`N \\times M` matrix :math:`A`
    :type A: scipy.sparse.dok_matrix
    :param b: :math:`N` vector :math:`b`
    :type b: np.ndarray
    :param xgeq0: If true, adds additional constraints :math:`x \geq 0`, defaults to False
    :type xgeq0: bool, optional
    :param solver: the used solver, may be "gurobi", "cbc", "cplex" or "glpk", defaults to "cbc"
    :type solver: str, optional
    :return: a 3d-tuple (x, optimal, strict) where `x` is the :math:`M`-d result vector :math:`x^*`, `optimal` indicates whether the LP-solution is optimal (i.e. not unbounded or infeasible) and `strict` whether :math:`A x^* < b` is satisfied.
    :rtype: Tuple[np.ndarray, Bool, Bool]
    """    
    # if xgeq0 is False, then 
    # A' = [ a11 ... a1M -1 
    #        ...
    #        aN1     aNM -1 ] is a Nx(M+1) matrix
    # x' = (x1 ... xM s)
    # b' = b
    # if xgeq0 is True, then
    # A' = [ a11 ... a1M -1 
    #        ...
    #        aN1 ... aNM -1 
    #         -1 0 ... 0 -1
    #         0 -1 ... 0 -1
    #         ...
    #         0   ... -1 -1 ] is a (N+M)x(M+1) matrix
    # x' = (x1 ... xM s)
    # b' = (b1 ... bN 0 ... 0) is a (N+M) vector
    A_ = dok_matrix(A.copy())
    b_ = b.copy()
    if xgeq0:
        A_.resize((A.shape[0]+A.shape[1], A.shape[1]+1))
        A_[A.shape[0]:,:-1] = -identity(A.shape[1])
        b_ = np.hstack((b_, np.zeros(A.shape[1])))
    else:
        A_.resize((A.shape[0], A.shape[1]+1))
    A_[:,-1] = -1

    opt = np.zeros(A_.shape[1])
    opt[-1] = 1

    lp = LP.from_coefficients(A_, b_, opt,"<=","min")

    result = lp.solve(solver)
    sres = result.result_vector[-1]
    return result.result_vector[:-1], (result.status != "optimal" or sres > 0), sres < 0

def generate_farkas_certificate(reach_form, mode, sense, threshold,solver="cbc"):
    """Generates Farkas certificates for a given reachability form, mode, sense and threshold using the characterizations in Table 1 of [FJB19]_. To this end uses an LP solver to find a satisfying vector of the corresponding polytope.

    :param reach_form: RF the certificate should be generated for
    :type reach_form: model.ReachabilityForm
    :param mode: must be either "min" or "max"
    :type mode: str
    :param sense: must be either "<=", ">=", "<" or ">".
    :type sense: str 
    :param threshold: threshold the certificate should be generated for
    :type threshold: float
    :param solver: used solver, must be either "gurobi", "cbc", "glpk" or "cplex", defaults to "cbc"
    :type solver: str, optional
    :return: :math:`N` or :math:`C` dimensional vector, dependent on mode
    :rtype: numpy.ndarray[float]
    """    

    assert (threshold >= 0) and (threshold <= 1)

    farkas_matr,rhs = __get_right_constraint_set(reach_form,mode,sense,threshold)
    if sense in [">=", ">"]:
        # Ax >=/> b <=> (-A)x <=/< -b
        farkas_matr, rhs = -farkas_matr, -rhs
    lp_result, optimal, is_strict = find_interior_point(farkas_matr,rhs,True,solver)

    if optimal:
        if sense == ">":
            if lp_result[reach_form.initial] > threshold:
                return lp_result
            else:
                print("Property is not satisfied!")
                return None
        elif sense == "<":
            if lp_result[reach_form.initial] < threshold:
                return lp_result
            else:
                print("Property is not satisfied!")
                return None
        else:
            return lp_result
    else:
        print("Property is not satisfied!")
        return None

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
