## this file returns MILPs/LPs as follows:
from switss.model import ReachabilityForm
from switss.solver import MILP, LP, GurobiMILP
from switss.utils import InvertibleDict, Graph
from . import AllOnesInitializer

import numpy as np
from scipy.sparse import dok_matrix
from bidict import bidict

def certificate_size(rf, mode):
    """returns the certificate dimension w.r.t. a given mode and RF

    :param rf: the RF
    :type rf: model.ReachabilityForm
    :param mode: either 'min' or 'max'
    :type mode: str
    :return: certificate dimension
    :rtype: int
    """
    assert mode in ["min", "max"]
    C, N = rf.system.P.shape
    if mode == "min":
        return N-2
    else:
        return C-2

def compute_upper_bounds(matr, rhs, solver="cbc",timeout=600):
    """
    computes point wise upper bounds for LPs/MILPs. Solves the LP

    .. math::

        K := \max \sum_{v \in V} \mathbf{x}(v) \; \\text{s.t.} \; 
        \mathbf{A} \mathbf{x} \leq \mathbf{b},\; \mathbf{x} \geq 0

    Returns either [K,K,...,K], for MDPs, and the solution vector of the above LP for DTMCs.

    :param matr: left hand side matrix :math:`\mathbf{A}` for constraining polytope
    :type matr: scipy.sparse.dok_matrix
    :param rhs: right hand side vector :math:`\mathbf{b}` for constraining polytope
    :type rhs: np.array
    :param solver: the used solver, defaults to "cbc"
    :type solver: str, optional
    :return: [K,K,...,K] for MDPs and the solution vector of the LP for DTMCs, and the status of the LP
    :rtype: Tuple[str, [float]]
    """
    rows, certsize = matr.shape
    upper_obj = np.ones(certsize)
    upper_bound_LP = LP.from_coefficients(matr, rhs, upper_obj, objective="max")

    for idx in range(certsize):
        upper_bound_LP.add_constraint([(idx, 1)], ">=", 0)

    lp_result = upper_bound_LP.solve(solver=solver,timeout=timeout)
    assert lp_result.status != "unbounded"

    if (certsize + 1 == rows):
        result = lp_result.result_vector
    else:
        result = np.full(shape=certsize, fill_value=lp_result.value)

    return lp_result.status, result


def groups_from_labels(rf, mode, labels=None, group_actions = False):
    """computes variable groups from a given mode and a set of labels.
    if the labels are 'None', then returns the identity mapping.

    :param rf: the RF
    :type rf: model.ReachabilityForm
    :param mode: either 'min' or 'max'
    :type mode: str
    :param labels: labels that group states, defaults to None
    :type labels: List[str], optional
    :param group_actions: returns the labeling in which all actions belong to the same group. Only applicable if mode="max" and labels=None. Defaults to False.
    :type labels: Bool, optional
    :return: the state/state-action-pair groupings
    :rtype: InvertibleDict[int, Set[int]] 
    """    
    assert mode in ["min", "max"]
    if labels is None:
        if mode == "max" and group_actions == True:
            groups = InvertibleDict({})
            for i in range(certificate_size(rf, mode)):
                (s,a) = rf.index_by_state_action.inv[i]
                groups.add(s, i)
            return groups
        else:
            return InvertibleDict({ idx: {idx} for idx in range(certificate_size(rf, mode))})
    elif mode == "min":
        groups = {}
        for label in labels:
            states = rf.system.states_by_label[label]
            groups[label] = states
        return InvertibleDict(groups)
    else:
        groups = InvertibleDict({})
        for label in labels:
            states = rf.system.states_by_label[label]
            for state in states:
                actions = rf.system.actions_by_state[state]
                for action in actions:
                    sap = rf.system.index_by_state_action[(state, action)]
                    groups.add(label, sap)
        return groups

def add_indicator_constraints(model, variables, upper_bounds, mode, groups, use_real_indicator_constrs=False, indicator_domain="real"):
    """
    adds new variables and constraints of the form 

    .. math:: 

        \mathbf{x}(v) \leq UB(v) \sigma(l), \quad \\text{for all}\; v \in V, l \in\Lambda(v) 

    to a given MILP/LP. Introduces a new variable :math:`\sigma(l)` for every label :math:`l` and, dependent on the given indicator domain, the additional constraint :math:`\sigma(l) \in \{0,1\}` or :math:`0 \leq \sigma(l) \leq 1`. 

    :param model: the given MILP/LP
    :type model: solver.MILP or solver.LP
    :param variables: set of variables :math:`V`.
    :type variables: Iterable[int]
    :param upper_bounds: point wise upper bound UB(i) for all variables i
    :type upper_bounds: [float], optional
    :param mode: either 'min' or 'max'
    :type mode: str
    :param groups: mapping :math:`\Lambda` grouping subsets of variables :math:`V` together
    :type groups: Dict[\*, Set[int]]
    :param use_real_indicator_constrs: True iff "real" boolean indicator constrs of type "s = 0 ==> x = 0" should be used. Only possible with gurobi, atm.
    :type  use_real_indicator_constrs: bool
    :param indicator_domain: domain of every :math:`\sigma(l)`, defaults to "real"
    :type indicator_domain: str, optional
    :return: the mapping of new indicator variables (:math:`\sigma(l)`) to their corresponding sets of variables (the set that contains all :math:`\mathbf{x}(v)` where :math:`l \in\Lambda(v)`).
    :rtype: utils.InvertibleDict[int, Set[int]]
    """
    if use_real_indicator_constrs:
        assert model.isinstance(GurobiMILP)

    indicator_to_group = {}

    if upper_bounds is not None:
        ub = 1
    else:
        ub = None

    for _, group in groups.items():
        indicator_var = model.add_variables_w_bounds([(indicator_domain,0,ub)])
        indicator_to_group[indicator_var] = group
        for var in group:
            if use_real_indicator_constrs:
                model.add_indicator_constraint(indicator_var,var)

            elif upper_bounds is not None:
                model.add_constraint([(var, 1), (indicator_var, -upper_bounds[var])], "<=", 0)
            else:
                model.add_constraint([(var, 1), (indicator_var, -1)], "<=", 0)

    indicator_to_group = InvertibleDict(indicator_to_group)

    return indicator_to_group


def construct_MILP(rf, threshold, mode, labels=None, relaxed=False, known_upper_bounds = None, upper_bound_solver="cbc", modeltype_str="pulp"):
    """
    constructs a MILP in the following form:

    .. math::

        \min \sum_{l \in L} \sigma(l) \; \\text{s.t.}\; \mathbf{x}(v) \in \mathcal{F}(\lambda),\; \mathbf{x}(v) \leq K \sigma(l),\; \sigma(l) \in \{0,1\},\; \\text{for all}\; v \in V, l \in\Lambda(v)

    where either :math:`V = \mathcal{S}` and :math:`\mathcal{F} = \mathcal{P}^{\mathrm{min}}` or :math:`V = \mathcal{M}` and :math:`\mathcal{F} = \mathcal{P}^{\mathrm{max}}` respectively.

    :param rf: the RF that induces the polytope :math:`\mathcal{F}`
    :type rf: model.ReachabilityForm
    :param threshold: the threshold :math:`\lambda`
    :type threshold: float
    :param mode: the chosen mode; either 'min' or 'max', defaults to "min"
    :type mode: str, optional
    :param labels: set of labels grouping states or state-action-pairs together. If None, then every 
        state/state-action-pair is considered separately, defaults to None
    :type labels: Dict[\*, Set[int]], optional
    :param relaxed: if set to True, then the last condition is relaxed to :math:`0 \leq \sigma \leq 1`, 
        defaults to False
    :type relaxed: bool, optional
    :param known_upper_bounds: vector of point wise upper bounds
    :type knwon_upper_bounds: [int] , optional
    :param upper_bound_solver: if the max-form is considered, :math:`K` needs to be computed by a solver, 
        defaults to "cbc"
    :type upper_bound_solver: str, optional
    :param modeltype: returns either a PuLP or Gurobi-MILP. Needs to be either 'gurobi' or 'pulp'
    :type modeltype: str
    :return: the resulting MILP. If the upper bound calculation fails, returns (None, None)
    :rtype: Tuple[solver.MILP, utils.InvertibleDict[int, Set[int]]]
    """
    assert modeltype_str in ["gurobi", "pulp"]
    modeltype = { "gurobi": GurobiMILP, "pulp": MILP }[modeltype_str]

    # construct constraining polytope matrices according to chosen mode
    fark_matr, fark_rhs = rf.fark_constraints(threshold, mode)
    certsize = certificate_size(rf, mode)

    upper_bounds = known_upper_bounds
    use_real_indicator_constrs = False

    if mode == "min" and isinstance(rf,ReachabilityForm) and upper_bounds is None:
        upper_bounds = np.ones(certsize)

    elif rf.is_ec_free:
        # try to compute (if there is no result within 10 minutes, stop)
        status, up_res = compute_upper_bounds(fark_matr, fark_rhs, solver=upper_bound_solver,timeout=600)
        if status == "infeasible":
            return None,None
        elif status == "optimal":
            upper_bounds = up_res

    if not relaxed and mode == "max" and upper_bounds is None:
        assert modeltype_str == "gurobi", "warning: exact minimization of max-properties only possible via gurobi interface at the moment."

        use_real_indicator_constrs = True

    # obtain variable groups from labels
    # for exact minimization in max case, make a group per state
    if mode == "max" and not relaxed:
        groups = groups_from_labels(rf, mode, labels=labels,group_actions=True)
    else:
        groups = groups_from_labels(rf, mode, labels=labels,group_actions=False)

    # compute lower/upper bounds
    bounds = []
    for varidx in range(certsize):
        #model.add_constraint([(varidx, 1)], ">=", 0)
        if mode == "min" and rf.in_proper_ec(varidx):
            ub = 0
        elif upper_bounds is not None:
            ub = upper_bounds[varidx]
        else:
            ub = None
        bounds.append((0,ub))

    # construct MILP
    certsize = certificate_size(rf, mode)
    # initialize model
    model = modeltype.from_coefficients(fark_matr, fark_rhs, np.zeros(certsize), ["real"]*certsize, sense="<=", objective="min", bounds=bounds)

    # add indicator variables, which are either binary or real, dependent on what relaxed was set to
    indicator_domain = "real" if relaxed else "binary"
    indicators = add_indicator_constraints(model, np.arange(certsize),
                                           upper_bounds, mode, groups,
                                           use_real_indicator_constrs,
                                           indicator_domain=indicator_domain)

    # make objective function opt=(0,...,0, 1,...,1) where the (0,..,0) part
    # corresponds to the x-variables and the (1,..,1) part to the indicators 
    objective = AllOnesInitializer(indicators).initialize()
    model.set_objective_function(objective)
    return model, indicators
