from . import ProblemFormulation, ProblemResult

class MILPExact(ProblemFormulation):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def solve(self, reachability_form):
        # model.ReachabilityForm -> problem.ProblemResult
        pass

    def __repr__(self):
        return "MILPExact(solver=%s)" % (self.solver)