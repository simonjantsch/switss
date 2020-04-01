import numpy as np

class SolverResult:
    def __init__(self, problem, variables):
        """Initializes a `SolverResult` from a pulp-model and the corresponding list of variables.
        
        :param problem: The (solved) pulp-model.
        :type problem: pulp.LpProblem
        :param variables: The list of corresponding variables.
        :type variables: List[pulp.LpVariable]
        """        
        assert problem.status in [1,-1,-2,-3]
        self.__status = {  1:"optimal", 
                        -1:"infeasible", 
                        -2:"unbounded", 
                        -3:"undefined"}[problem.status]
        self.__result = list(map(lambda var: var.value(), variables))

    @property
    def status(self):
        """The status of a solved LP/MILP instance. May be optimal, infeasible, unbounded or undefined.
        
        :return: The status.
        :rtype: str
        """        
        return self.__status

    @property
    def result(self):
        """The result of a solved LP/MILP instance. 
        
        :return: A list containing assignments to variables.
        :rtype: List[float]
        """        
        return self.__result

    def __str__(self):
        return "SolverResult(status=%s, result=%s)" % (self.status, self.result)