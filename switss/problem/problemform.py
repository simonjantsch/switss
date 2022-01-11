from abc import ABC, abstractmethod, abstractproperty
from bidict import bidict
from collections import deque
import numpy as np

from ..model import RewardReachabilityForm
from ..solver import MILP,LP
from ..utils import InvertibleDict

class ProblemFormulation:
    """A ProblemFormulation is an abstract base class for
    problems that are aimed at finding minimal witnesses
    for DTMCs or MDPs.
    """    
    def __init__(self):
        pass

    def solve(self, 
              reachability_form, 
              threshold, 
              mode, 
              labels=None, 
              timeout=None):
        """Searches for small subsystems for a given reachability form
        such that the probability of reaching the target state is above
        a given threshold: 

        .. math::

            \mathbf{Pr}_{\mathbf{x}}^{*}(\diamond \\text{goal}) \geq \lambda

        where :math:`\lambda` is the given threshold and :math:`* \in \{\\text{min},\\text{max}\}`.

        `.solve` returns the final result if multiple solutions are found.

        :param reachability_form: The system that should be minimized.
        :type reachability_form: model.ReachabilityForm
        :param threshold: The given threshold.
        :type threshold: float
        :param mode: The polytope that should be selected for optimization, either "min" or "max"
        :type mode: str
        :param labels: A list of labels. 
        :type labels: List[str]
        :param fixed_values: A dictionary mapping states to fixed values.
        :type fixed_values: Dict[int, int]
        :return: The resulting subsystem.
        :rtype: problem.Subsystem
        """
        return deque(self.solveiter(    reachability_form, 
                                        threshold, 
                                        mode,
                                        labels=labels,
                                        timeout=timeout), maxlen=1).pop()

    def solveiter(self, 
                  reachability_form, 
                  threshold, 
                  mode, 
                  labels=None, 
                  timeout=None):
        """Searches for small subsystems for a given reachability form
        such that the probability of reaching the target state is above
        a given threshold: 

        .. math::

            \mathbf{Pr}_{\mathbf{x}}^{*}(\diamond \\text{goal}) \geq \lambda

        where :math:`\lambda` is the given threshold and :math:`* \in \{\\text{min},\\text{max}\}`.

        `.solveiter` returns an iterator over all systems that are found. 
        
        :param reachability_form: The system that should be minimized.
        :type reachability_form: model.ReachabilityForm
        :param threshold: The given threshold.
        :type threshold: float
        :param mode: The polytope that should be selected for optimization, either "min" or "max"
        :type mode: str
        :param labels: A list of labels. 
        :type labels: List[str]
        :param fixed_values: A dictionary mapping states or state-action-pairs 
            or labels to fixed values.
        :type fixed_values: Dict[int, int] or Dict[str, int]
        :return: The resulting subsystem.
        :rtype: problem.Subsystem
        """
        assert (threshold >= 0)
        if not isinstance(reachability_form,RewardReachabilityForm):
             assert (threshold <= 1)
        assert mode in ["min","max"]
        return self._solveiter(reachability_form, 
                               threshold,
                               mode, 
                               labels=labels, 
                               timeout=timeout)

    @abstractmethod
    def _solveiter(self, 
                   reachability, 
                   threshold, 
                   mode, 
                   labels, 
                   timeout=None):
        pass

    def __repr__(self):
        params = ["%s=%s" % (k,v) for k,v in self.details.items() if k != "type"]
        return "%s(%s)" % (self.details["type"], ",".join(params))

    @abstractproperty
    def details(self):
        """A dictionary that contains information about this instance. Content
        is dependent on respective class and instance.
        """        
        pass

