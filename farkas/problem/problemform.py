from abc import ABC, abstractmethod, abstractproperty

class ProblemFormulation:
    """A ProblemFormulation is an abstract base class for
    problems that are aimed at finding minimal witnesses
    for DTMCs or MDPs.
    """    
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, reachability_form, threshold):
        """Finds a minimal witnessing subsystem for a given reachability form
        such that the probability of reaching the target state is above
        a given threshold: 

        .. math::

            Pr_{\mathbf{x}}(\diamond goal) \geq \lambda

        where :math:`\lambda` is the given threshold.

        :param reachability_form: The system that should be minimized.
        :type reachability_form: model.ReachabilityForm
        :param threshold: The given threshold.
        :type threshold: float
        :return: The resulting subsystem (minimal witness).
        :rtype: problem.Subsystem
        """
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