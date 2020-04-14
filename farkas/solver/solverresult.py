import numpy as np

class SolverResult:
    def __init__(self, status, result_vector, result_value):
        """Result of a solved MILP or LP instance.
        
        :param status: Status of the solved instance, e.g. optimal, infeasible, unbounded or undefined.
        :type status: str
        :param result: Resulting variable assignments.
        :type result: List[float]
        """
        assert status in ["optimal", "infeasible", "unbounded", "undefined"]
        self.status = status
        self.result_vector = result_vector
        self.result_value = result_value

    def __repr__(self):
        return "SolverResult(status=%s, result_vector=%s, result_value=%s)" % (self.status, self.result_vector, self.result_value)
