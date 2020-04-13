import numpy as np

class SolverResult:
    def __init__(self, status, result):
        """Result of a solved MILP or LP instance.
        
        :param status: Status of the solved instance, e.g. optimal, infeasible, unbounded or undefined.
        :type status: str
        :param result: Resulting variable assignments.
        :type result: List[float]
        """        
        assert status in ["optimal", "infeasible", "unbounded", "undefined"]
        self.status = status
        self.result = result

    def __str__(self):
        return "SolverResult(status=%s, result=%s)" % (self.status, self.result)