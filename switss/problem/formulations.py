## this file returns MILPs/LPs as follows:
from switss.solver import MILP, LP
from . import AllOnesInitializer
from switss.utils import InvertibleDict
import numpy as np

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

def compute_upper_bound(matr, rhs, solver="cbc"):
    """
    computes upper bound :math:`K` for LPs/MILPs. Solves the LP

    .. math::

        K := \max \sum_{v \in V} \mathbf{x}(v) \; \\text{s.t.} \; 
        \mathbf{A} \mathbf{x} \leq \mathbf{b},\; \mathbf{x} \geq 0

    :param matr: left hand side matrix :math:`\mathbf{A}` for constraining polytope
    :type matr: scipy.sparse.dok_matrix
    :param rhs: right hand side vector :math:`\mathbf{b}` for constraining polytope
    :type rhs: np.array
    :param solver: the used solver, defaults to "cbc"
    :type solver: str, optional
    :return: the optimal value :math:`K` and status of LP
    :rtype: Tuple[str, float]
    """
    _, certsize = matr.shape
    upper_obj = np.ones(certsize)
    upper_bound_LP = LP.from_coefficients(matr, rhs, upper_obj, objective="max")

    for idx in range(certsize):
        upper_bound_LP.add_constraint([(idx, 1)], ">=", 0)

    lp_result = upper_bound_LP.solve(solver=solver)
    assert lp_result.status != "unbounded"
    return lp_result.status, lp_result.value


def groups_from_labels(rf, mode, labels=None):
    """computes variable groups from a given mode and a set of labels.
    if the labels are 'None', then returns the identity mapping.

    :param rf: the RF
    :type rf: model.ReachabilityForm
    :param mode: either 'min' or 'max'
    :type mode: str
    :param labels: labels that group states, defaults to None
    :type labels: List[str], optional
    :return: the state/state-action-pair groupings
    :rtype: InvertibleDict[int, Set[int]] 
    """    
    assert mode in ["min", "max"]
    if labels is None:
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

def add_indicator_constraints(model, variables, upper_bound, mode, groups, indicator_domain="real"):
    """
    adds new variables and constraints of the form 

    .. math:: 

        \mathbf{x}(v) \leq K \sigma(l), \quad \\text{for all}\; v \in V, l \in\Lambda(v) 

    to a given MILP/LP. Introduces a new variable :math:`\sigma(l)` for every label :math:`l` and, dependent on the given indicator domain, the additional constraint :math:`\sigma(l) \in \{0,1\}` or :math:`0 \leq \sigma(l) \leq 1`. 

    :param model: the given MILP/LP
    :type model: solver.MILP or solver.LP
    :param variables: set of variables :math:`V`.
    :type variables: Iterable[int]
    :param upper_bound: value for :math:`K`
    :type upper_bound: float
    :param mode: either 'min' or 'max'
    :type mode: str
    :param groups: mapping :math:`\Lambda` grouping subsets of variables :math:`V` together
    :type groups: Dict[\*, Set[int]]
    :param indicator_domain: domain of every :math:`\sigma(l)`, defaults to "real"
    :type indicator_domain: str, optional
    :return: the mapping of new indicator variables (:math:`\sigma(l)`) to their corresponding sets of variables (the set that contains all :math:`\mathbf{x}(v)` where :math:`l \in\Lambda(v)`).
    :rtype: utils.InvertibleDict[int, Set[int]]
    """

    indicator_to_group = {}
    for _, group in groups.items():
        indicator_var = model.add_variables(indicator_domain)
        indicator_to_group[indicator_var] = group
        for var in group:
            model.add_constraint([(var, 1), (indicator_var, -upper_bound)], "<=", 0)
    indicator_to_group = InvertibleDict(indicator_to_group)

    if indicator_domain != "binary":
        for indicator_var in indicator_to_group.keys():
            model.add_constraint([(indicator_var, 1)], ">=", 0)
            model.add_constraint([(indicator_var, 1)], "<=", 1)

    return indicator_to_group


def construct_MILP(rf, threshold, mode, labels=None, relaxed=False, upper_bound_solver="cbc"):
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
    :param upper_bound_solver: if the max-form is considered, :math:`K` needs to be computed by a solver, 
        defaults to "cbc"
    :type upper_bound_solver: str, optional
    :return: the resulting MILP. If the upper bound calculation fails, returns (None, None)
    :rtype: Tuple[solver.MILP, Dict[int,Set[int]]]
    """
    # construct constraining polytope matrices according to chosen mode
    fark_matr, fark_rhs = rf.fark_constraints(threshold, mode)
    
    # compute the upper bound K
    if mode == "min":
        upper_bound = 1. 
    else:
        status, upper_bound = compute_upper_bound(fark_matr, fark_rhs, solver=upper_bound_solver)
        if status != "optimal":
            return None, None
    
    # obtain variable groups from labels
    groups = groups_from_labels(rf, mode, labels=labels)
    
    # construct MILP
    certsize = certificate_size(rf, mode)
    model = MILP.from_coefficients(fark_matr, fark_rhs, np.zeros(certsize), ["real"]*certsize) # initialize model
    for varidx in range(certsize):
        model.add_constraint([(varidx, 1)], ">=", 0)
        model.add_constraint([(varidx, 1)], "<=", upper_bound) # isn't this constraint unnecessary?
    # add indicator variables, which are either binary or real, dependent on what relaxed was set to
    indicator_domain = "real" if relaxed else "binary"
    indicators = add_indicator_constraints(model, np.arange(certsize), 
                                           upper_bound, mode, groups, 
                                           indicator_domain=indicator_domain)
    # make objective function opt=(0,...,0, 1,...,1) where the (0,..,0) part
    # corresponds to the x-variables and the (1,..,1) part to the indicators 
    objective = AllOnesInitializer(indicators).initialize()
    model.set_objective_function(objective)
    return model, indicators