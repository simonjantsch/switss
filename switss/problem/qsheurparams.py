from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np

class Initializer(ABC):
    """Abstract base class for QSHeur-initializers. An initializer 
    computes the initial objective function :math:`\mathbf{o}_0` of a QSHeur-problem.
    """

    def __init__(self, reachability_form, mode, indicator_to_group, weights = None, **kwargs):
        """
        :param reachability_form: The reachability-form that should be minimized.
            Computation of objective function may or may not be dependent on the reachability-form.
        :type reachability_form: model.ReachabilityForm
        :param mode: QSHeur-mode, i.e. "max" or "min"
        :type mode: str
        :param indicator_to_group: mapping from group indices to sets of states/state-action pairs. If label-based
            minimization was not choosen, every group consists of one state/state-action pair only.
            If label-based minimization was choosen, every group corresponds to one of the specified labels
            and yields all states that belong to this label.
        :type indicator_to_group: utils.InvertibleDict
        :param weights: a mapping from groups to weights. If None, then each group is assigned weight one.
        :type mode: Dict[int,float]
        """
        assert mode in ["max", "min"]
        self.reachability_form = reachability_form
        self.mode = mode
        self.indicator_to_group = indicator_to_group
        self.groups = self.indicator_to_group.keys()
        self.variables = self.indicator_to_group.inv.keys()
        if weights is None:
            self.weights = {gr : 1 for gr in self.groups }
        else:
            self.weights = weights

    @abstractmethod
    def initialize(self):
        """Computes the initial objective function :math:`\mathbf{o}_0` for a QSHeur-run as a list of index/
        coefficient pairings. Every entry is a tuple :math:`(v, \mathbf{o}_{0}(v))` where :math:`v` corresponds
        to some group index :math:`v \in V`.

        :return: The initial objective function 
        :rtype: List[Tuple[int,float]]
        """
        pass

    def __repr__(self):
        return type(self).__name__

class Updater(ABC):
    """Abstract base class for QSHeur-updaters. An updater 
    computes the new objective function :math:`\mathbf{o}_{i+1}` after each QSHeur-iteration. Computation may
    or may not be dependent on the last result vector :math:`QS(i)`.
    """    

    def __init__(self, reachability_form, mode, indicator_to_group, weights=None, **kwargs):
        """
        :param reachability_form: The reachability-form that should be minimized.
            Computation of objective function may or may not be dependent on the reachability-form.
        :type reachability_form: model.ReachabilityForm
        :param mode: QSHeur-mode, i.e. "max" or "min"
        :type mode: str
        :param indicator_to_group: mapping from group indices to sets of states/state-action pairs. If label-based
            minimization was not choosen, every group consists of one state/state-action pair only.
            If label-based minimization was choosen, every group corresponds to one of the specified labels
            and yields all states that belong to this label.
        :type indicator_to_group: utils.InvertibleDict
        :param weights: a vector of weights associated to the groups. If None, then implicitly every group has weight one.
        :type mode: [float]
        """
        assert mode in ["max", "min"]
        self.reachability_form = reachability_form
        self.mode = mode
        self.indicator_to_group = indicator_to_group
        self.groups = self.indicator_to_group.keys()
        self.variables = self.indicator_to_group.inv.keys()
        if weights is None:
            self.weights = {gr : 1 for gr in self.groups }
        else:
            self.weights = weights

    @abstractmethod
    def update(self, last_result):
        """ 
        Computes the updated objective function :math:`\mathbf{o}_{i+1}` for a QSHeur-run as a list of index/
        coefficient pairings. Every entry is a tuple :math:`(v, \mathbf{o}_{0}(v))` where :math:`v` corresponds
        to some group index :math:`v \in V`. 

        :param last_result: The past result vector :math:`QS(i)`.
        :return: The updated objective function 
        :rtype: List[Tuple[int,float]        
        """
        pass

    @abstractmethod
    def constraints(self, last_result):
        """
        Computes a list of linear constraints that are added to the LP.
        Every linear constraint is given as a tuple :math:`(((i_1,a_1),\dots,(i_m,a_m)),\circ,b)`,
        where :math:`\circ \in \{ \leq, \geq, = \}`, :math:`i_1,\dots,i_m \in \mathbb{N}` are the variable indicies 
        that influence the constraint and :math:`a_1,\dots,a_m \in \mathbb{R}` are their respective coefficients.
        :math:`b` is the right hand side of the resulting equation. This expresses the constraint

        .. math::

            \sum_{j=1}^m a_j \mathbf{x}_{i_j} \circ b

        where :math:`\mathbf{x}` is the vector of variables.

        :param last_result: The past result vector :math:`QS(i)`.
        :type last_result: List[Tuple[List[Tuple[int,float]], str, float]]
        """
        pass

    def __repr__(self):
        return type(self).__name__

class AllOnesInitializer(Initializer):
    """Initializes each group by its weight, i.e.
    
    .. math::
    
        \mathbf{o}_0(v) = wgt(v), \quad \\forall v \in V 
    
    """
    def __init__(self, indicator_to_group, weights=None, **kwargs):
        super(AllOnesInitializer, self).__init__(None, "min", indicator_to_group, weights=weights)

    def initialize(self):
        return [(group,self.weights[group]) for group in self.groups]

class InverseResultUpdater(Updater):
    """Gives most weight to groups that were removed in the last iteration (i.e. :math:`QS_{\sigma}(i)(v) = 0`)
    and increases weight of groups that are already close to beeing removed (i.e. small :math:`QS_{\sigma}(i)(v)`):

    .. math::

        \mathbf{o}_{i+1}(v) = \\begin{cases} 
            1/QS_{\sigma}(i)(v) & QS_{\sigma}(i)(v) > 0, \\\ 
            C & QS_{\sigma}(i)(v) = 0 
        \end{cases}, \quad \\forall v \in V. 

    where :math:`C \gg 0`.

    """    

    def update(self, last_result):
        C = np.min([np.max([0,1e8] + [1/last_result[group] for group in self.groups if last_result[group] != 0]),1e9])
        objective = [(group, self.weights[group] * np.min([1/last_result[group],C]) if last_result[group] > 0 else C) for group in self.groups]
        return objective

    def constraints(self, last_result):
        return []

class InverseResultFixedZerosUpdater(Updater):

    def update(self, last_result):
        return [((group, self.weights[group]/last_result[group]) if last_result[group] > 0 else (group,0.)) for group in self.groups]

    def constraints(self, last_result):
        # if fix_zero_states is enabled, every group that has a 0-entry in the 
        # result vector will be constrained to equal 0 in the coming LPs
        ret = []
        for group in self.groups:
            if last_result[group] == 0:
                constraint = ([(group,1.)], "=", 0)
                ret.append(constraint)
        return ret

class InverseReachabilityInitializer(Initializer):
    """Gives groups the most weight that have a low probability of reaching the goal state.

    If the objective function has to be computed for the z-Form, we compute the average goal reachability
    probability over all states that are in the group and then return the inverse value:

    .. math::

        \mathbf{o}_0(v) = \left( \\frac{1}{|S_v|} \sum_{s \in S_v} 
        \mathbf{Pr}_s^{\\text{min}}(\diamond \\text{goal}) \\right)^{-1} 

    This gives a high weight to groups that have a low probability of reaching the goal state.

    If the objective function has to be computed for the y-Form, we compute the average goal reachability
    probability over the states that are yield by the state-action pairs.

    Let 

    .. math::

        \mathbf{x}((s,a)) = \mathbf{P}((s,a),\\text{goal}) + \sum_{d \in S} 
            \mathbf{P}((s,a),d) \mathbf{Pr}^{\\text{min}}_d(\diamond \\text{goal})

    We then define

    .. math::

        \mathbf{o}_0(v) = \left(  \\frac{1}{|\mathcal{M}_v|} \sum_{(s,a) \in \mathcal{M}_v}  \mathbf{x}((s,a)) \\right)^{-1}

    The reasoning is similar to the z-Form; state-action pairs that yield states which have a low probability
    of reaching the goal state get a high weight.
    """    
    def __init__(self, reachability_form, mode, indicator_to_group, solver="cbc"):
        super(InverseReachabilityInitializer, self).__init__(reachability_form, mode, indicator_to_group)
        self.solver = solver

        self.Pr = None
        if self.mode == "min":
            # if mode is min, each variable in a group corresponds to a state
            Pr_x = self.reachability_form.max_z_state(solver=self.solver)
            self.Pr = Pr_x
        else:
            # if mode is max, each variable in a group corresponds to a state-action pair index
            Pr_x_a = self.reachability_form.max_z_state_action(solver=self.solver)
            self.Pr = Pr_x_a

    def initialize(self):
        ret = []

        for group in self.groups:
            variables = self.indicator_to_group[group]
            variablecount = len(variables)
            weighted_probability = sum([self.Pr[var] for var in variables])/variablecount
            ret.append((group, self.weights[group] * np.min([1e9,1/weighted_probability if weighted_probability > 0 else 1e9])))

        return ret

class InverseFrequencyInitializer(Initializer):
    """"""

    def __init__(self, reachability_form, mode, indicator_to_group, solver="cbc"):
        super(InverseFrequencyInitializer, self).__init__(reachability_form, mode, indicator_to_group)
        self.solver = solver

        self.E = None
        if self.mode == "min":
            # if mode is min, each variable in a group corresponds to a state
            E_x = self.reachability_form.max_y_state(solver=self.solver)
            self.E = E_x
        else:
            # if mode is max, each variable in a group corresponds to a state-action pair index
            E_x_a = self.reachability_form.max_y_state_action(solver=self.solver)
            self.E = E_x_a


    def initialize(self):
        ret = []

        for group in self.groups:
            variables = self.indicator_to_group[group]
            expected_val_sum = sum([self.E[var] for var in variables])
            ret.append((group, self.weights[group] * np.min([1e9,1/expected_val_sum]) if expected_val_sum > 0 else 1e8))

        return ret


class InverseCombinedInitializer(Initializer):
    def __init__(self, reachability_form, mode, indicator_to_group, solver="cbc"):
        super(InverseCombinedInitializer, self).__init__(reachability_form, mode, indicator_to_group)
        self.solver = solver

        self.E = None
        if self.mode == "min":
            # if mode is min, each variable in a group corresponds to a state
            E_x = self.reachability_form.max_y_state(solver=self.solver)
            Pr_x = self.reachability_form.max_z_state(solver=self.solver)
            self.V = E_x*Pr_x
        else:
            # if mode is max, each variable in a group corresponds to a state-action pair index
            E_x_a = self.reachability_form.max_y_state_action(solver=self.solver)
            Pr_x_a = self.reachability_form.max_z_state_action(solver=self.solver)
            self.V = E_x_a*Pr_x_a


    def initialize(self):
        ret = []

        for group in self.groups:
            variables = self.indicator_to_group[group]
            expected_val_sum = sum([self.V[var] for var in variables])
            ret.append((group, self.weights[group] * np.min([1e9,1/expected_val_sum]) if expected_val_sum > 0 else 1e8))

        return ret
