
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np

class Initializer(ABC):
    """Abstract base class for QSHeur-initializers. An initializer 
    computes the initial objective function :math:`\mathbf{o}_0` of a QSHeur-problem.
    """

    def __init__(self, reachability_form, mode, indicator_to_group, **kwargs):
        """
        :param reachability_form: The reachability-form that should be minimized.
            Computation of objective function may or may not be dependent on the reachability-form.
        :type reachability_form: model.ReachabilityForm
        :param mode: QSHeur-mode, i.e. "max" or "min"
        :type mode: str
        """
        assert mode in ["max", "min"]
        self.reachability_form = reachability_form
        self.mode = mode
        self.indicator_to_group = indicator_to_group
        self.groups = self.indicator_to_group.keys()
        self.variables = self.indicator_to_group.inv.keys()

    @abstractmethod
    def initialize(self):
        """Computes the initial objective function :math:`\mathbf{o}_0` for a QSHeur-run. 

        :param indicator_keys: Set of indices :math:`v_1,\dots,v_m` where each index :math:`v` corresponds to one
            state group variable :math:`\sigma(v)`. Only indices that occur in this set should be considered in the objective function.
        :type indicator_keys: Iterable[int]
        :return: A list of index/coefficient pairings. Each entry is a tuple :math:`(v, \mathbf{o}_{0}(v))` 
            where :math:`v \in \{ v_1,\dots,v_m\}`.
        :rtype: List[Tuple[int,float]]
        """
        pass

    def __repr__(self):
        return type(self).__name__

class Updater(ABC):
    """Abstract base class for QSHeur-updaters. An updater 
    computes the new objective function :math:`\mathbf{o}_{i+1}` after each QSHeur-iteration. Computation may
    or may not be dependent on the last result vector :math:`QS(i) = (QS_{\mathbf{x}}(i)\ QS_{\sigma}(i))`.
    """    

    def __init__(self, reachability_form, mode, indicator_to_group, **kwargs):
        """
        :param reachability_form: The reachability-form that should be minimized.
            Computation of objective function may or may not be dependent on the reachability-form.
        :type reachability_form: model.ReachabilityForm
        :param mode: QSHeur-mode, i.e. "max" or "min"
        :type mode: str
        """
        assert mode in ["max", "min"]
        self.reachability_form = reachability_form
        self.mode = mode
        self.indicator_to_group = indicator_to_group
        self.groups = self.indicator_to_group.keys()
        self.variables = self.indicator_to_group.inv.keys()

    @abstractmethod
    def update(self, last_result):
        """Computes the updated objective function :math:`\mathbf{o}_{i+1}`.

        :param last_result: The past result vector :math:`QS(i)`.
        :param indicator_keys: Set of indices :math:`v_1,\dots,v_m` where each index :math:`v` corresponds to one
            state group variable :math:`\sigma(v)`. Only indices that occur in this set should be considered in the objective function.
        :type indicator_keys: Iterable[int]
        :return: A list of index/coefficient pairings. Each entry is a tuple :math:`(v, \mathbf{o}_{i+1}(v))` 
            where :math:`v \in \{ v_1,\dots,v_m\}`.
        :rtype: List[Tuple[int,float]]
        """        
        pass

    def __repr__(self):
        return type(self).__name__

class AllOnesInitializer(Initializer):
    """Gives each group the same weight, i.e.
    
    .. math::
    
        \mathbf{o}_0(v) = 1, \quad \\forall v \in \{ v_1, \dots v_m \} 
    
    """
    def __init__(self, indicator_to_group, **kwargs):
        super(AllOnesInitializer, self).__init__(None, "min", indicator_to_group)

    def initialize(self):
        return [(group,1) for group in self.groups]

class InverseResultUpdater(Updater):
    """Gives most weight to groups that were removed in the last iteration (i.e. :math:`QS_{\sigma}(i)(v) = 0`)
    and increases weight of groups that are already close to beeing removed (i.e. small :math:`QS_{\sigma}(i)(v)`):

    .. math::

        \mathbf{o}_{i+1}(v) = \\begin{cases} 
            1/QS_{\sigma}(i)(v) & QS_{\sigma}(i)(v) > 0, \\\ 
            C & QS_{\sigma}(i)(v) = 0 
        \end{cases}, \quad \\forall v \in \{v_1,\dots,v_m\}.

    where :math:`C \gg 0`.

    """    
    def update(self, last_result):
        C = np.max([1/last_result[group] for group in self.groups if last_result[group] != 0]) + 1e8
        objective = [(group, 1/last_result[group] if last_result[group] > 0 else C) for group in self.groups]
        return objective

class InverseReachabilityInitializer(Initializer):
    """Gives groups the most weight that have a low probability of reaching the goal state.

    .. math::

        \mathbf{o}_0^{(v)} = 1/Pr_{v}(\diamond goal), \quad \\forall v \in \{ v_1, \dots v_m \} 
 
    """    
    def __init__(self, reachability_form, mode, indicator_to_group, solver="cbc"):
        super(InverseReachabilityInitializer, self).__init__(reachability_form, mode, indicator_to_group)
        self.solver = solver

        self.Pr = None
        if self.mode == "min":
            # if mode is min, each variable in a group corresponds to a state
            Pr_x = self.reachability_form.max_z_state(solver=self.solver)
            assert (Pr_x > 0).all()
            self.Pr = Pr_x
        else:
            # if mode is max, each variable in a group corresponds to a state-action pair index
            Pr_x_a = self.reachability_form.max_z_state_action(solver=self.solver)
            assert (Pr_x_a > 0).all()
            self.Pr = Pr_x_a

    def initialize(self):
        ret = []

        for group in self.groups:
            variables = self.indicator_to_group[group]
            variablecount = len(variables)
            weighted_probability = sum([self.Pr[var] for var in variables])/variablecount
            ret.append((group, 1/weighted_probability))

        return ret

class InverseFrequencyInitializer(Initializer):
    """Gives groups the most weight that have a low expected frequency.

    .. math::

        \mathbf{o}_0^{(v)} = 1/\mathbb{E}[v], \quad \\forall v \in \{ v_1, \dots, v_m \}

    """

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
            ret.append((group, 1/expected_val_sum if expected_val_sum > 0 else 1e8))

        return ret
