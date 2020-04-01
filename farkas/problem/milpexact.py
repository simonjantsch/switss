from . import ProblemFormulation, ProblemResult

class MILPExact(ProblemFormulation):
    def __init__(self):
        super().__init__()

    def solve(self, reachability_form):
        # model.ReachabilityForm -> problem.ProblemResult
        pass