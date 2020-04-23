
from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    """Abstract base class for QSHeur-initializers. An initializer 
    computes the initial objective function :math:`\sigma_0` of a QSHeur-problem.
    """

    def __init__(self, reachability_form, mode):
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

    @abstractmethod
    def initialize(self, indicator_keys):
        """Computes the initial objective function :math:`\sigma_0` for a QSHeur-run. 

        :param indicator_keys: Set of indices :math:`v_1,\dots,v_m \in \{ 1,\dots,M \}` where :math:`M \in \{N,C\}`
            is the number of states or state-action-pairs (dependent on mode) and each index :math:`v` corresponds to 
            such a state or state-action-pair. Only states that occur in this index set should be considered in the objective function.
        :type indicator_keys: Iterable[int]
        :return: A list of state-index/coefficient pairings. Each entry is a tuple :math:`(v, \sigma_{i+1}^{(v)})` 
            where :math:`v \in \{ v_1,\dots,v_m\}`.
        :rtype: List[Tuple[int,float]]
        """
        pass

    def __repr__(self):
        return type(self).__name__

class Updater(ABC):
    """Abstract base class for QSHeur-updaters. An updater 
    computes the new objective function :math:`\sigma_{i+1}` after each QSHeur-iteration. Computation may
    or may not be dependent on the last result vector :math:`QS_i`.
    """    

    def __init__(self, reachability_form, mode):
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

    @abstractmethod
    def update(self, last_result, indicator_keys):
        """Computes the updated objective function :math:`\sigma_{i+1}(QS_i)` for a QSHeur-run from 
        the past result-vector `last_result` :math:`= QS_i`.

        :param last_result: The past result vector :math:`QS_i`.
        :type last_result: :math:`N` or :math:`C`-dimensional vector containing values for each state
        :param indicator_keys: Set of indices :math:`v_1,\dots,v_m \in \{ 1,\dots,M \}` where :math:`M \in \{N,C\}`
            is the number of states or state-action-pairs (dependent on mode) and each index :math:`v` corresponds to 
            such a state or state-action-pair. Only states that occur in this index set should be considered in the objective function.
        :type indicator_keys: Iterable[int]
        :return: A list of state-index/coefficient pairings. Each entry is a tuple :math:`(v, \sigma_{i+1}^{(v)})` 
            where :math:`v \in \{ v_1,\dots,v_m\}`.
        :rtype: List[Tuple[int,float]]
        """        
        pass

    def __repr__(self):
        return type(self).__name__

class AllOnesInitializer(Initializer):
    """Gives each state the same weight, i.e.
    
    .. math::
    
        \sigma_0^{(v)} = 1, \quad \\forall v \in \{ v_1, \dots v_m \} 
    
    """

    def initialize(self, indicator_keys):
        return [(i,1) for i in indicator_keys]

class InverseResultUpdater(Updater):
    """Gives most weights to states that were removed in the last iteration (i.e. :math:`QS_i^{(v)} = 0`)
    and increases weight of states that are already close to beeing removed (i.e. small :math:`QS_i^{(v)}`):

    .. math::

        \sigma_{i+1}^{(v)} = \\begin{cases} 
            1/QS_i^{(v)} & QS_i^{(v)} > 0, \\\ 
            C & QS_i^{(v)} = 0 
        \end{cases}, \quad \\forall v \in \{v_1,\dots,v_m\}.

    where :math:`C \gg 0`.

    """    
    def update(self, last_result, indicator_keys):
        C = np.max(last_result) + 1e8
        objective = []
        for i in indicator_keys:
            objective.append((i, 1/last_result[i] if last_result[i] > 0 else C))
        return objective

class InverseReachabilityInitializer(Initializer):
    """Gives states the most weight that have a low probability of reaching the goal state.
    Currently only works for min- and max-form if model is a DTMC.

    .. math::

        \sigma_0^{(v)} = 1/(Pr_{v}(\diamond goal)+c), \quad \\forall v \in \{ v_1, \dots v_m \} 

    where :math:`c` is positive but close to zero. 
    """    
    def initialize(self, indicator_keys):
        I = np.identity(self.reachability_form.N)
        P = self.reachability_form.P
        to_target = self.reachability_form.to_target
        reachability = (I-P)**-1 * np.matrix(to_target).T
        return [(i,1/(reachability[i]+1e-12)) for i in indicator_keys]
