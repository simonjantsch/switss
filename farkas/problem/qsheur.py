from . import ProblemFormulation, ProblemResult

class QSHeur(ProblemFormulation):
    def __init__(self):
        super().__init__()
        # initialize heuristics here,
        # number of iterations,
        # etc.

    def solve(self, reachability_form):
        # model.ReachabilityForm -> problem.ProblemResult
        pass