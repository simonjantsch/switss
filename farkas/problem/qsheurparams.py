
from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    """Abstract base class for QSHeur-initializers. An initializer 
    computes the initial objective function :math:`\mathbf{o}_0` of a QSHeur-problem.
    """

    def __init__(self, reachability_form, mode, indicator_to_group):
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

    def __init__(self, reachability_form, mode, indicator_to_group):
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
        C = np.max([1/last_result[group] for group in self.groups if last_result[group] != 0]) + 1
        objective = [(group, 1/last_result[group] if last_result[group] > 0 else C) for group in self.groups]
        return objective

# class InverseReachabilityInitializer(Initializer):
#     """Gives states the most weight that have a low probability of reaching the goal state.
#     Currently only works for min- and max-form if model is a DTMC.
# 
#     .. math::
# 
#         \mathbf{o}_0^{(v)} = 1/(Pr_{v}(\diamond goal)+c), \quad \\forall v \in \{ v_1, \dots v_m \} 
# 
#     where :math:`c` is positive but close to zero. 
#     """    
#     def initialize(self, indicator_keys):
#         P = self.reachability_form.P
#         to_target = self.reachability_form.to_target
#         I = np.identity(P.shape[0])
#         reachability = (I-P)**-1 * np.matrix(to_target).T
#         print(indicator_keys)
#         return [(i,1/(reachability[i]+1e-12)) for i in indicator_keys]
