from . import MILP, LP, SolverResult
import pulp

class Wrapper:
    def __init__(self, solver="cbc"):
        """Initializes a wrapper for a MILP/LP solver. Currently supported is only the CBC-solver.
        
        :param solver: The solver that should be used, defaults to "cbc"
        :type solver: str, optional
        """        
        assert solver in ["gurobi", "cbc"]
        self.solver = { "gurobi" : pulp.GUROBI_CMD, 
                        "cbc" : pulp.PULP_CBC_CMD }[solver]()

    def solve(self, problem):
        """Solves a MILP/LP problem instance by constructing a pulp-model and then solving it.
        
        :param problem: MILP/LP problem instance.
        :type problem: solver.MILP
        :return: Result of the solver.
        :rtype: solver.SolverResult
        """        
        model, variables = problem.pulpmodel()
        model.solve(self.solver)
        return SolverResult(model, variables)