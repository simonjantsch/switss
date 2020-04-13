from ..model import ReachabilityForm

class ProblemResult:
    def __init__(self, status,reachability_form=None,mapping=None):
        self.status = status
        self.reachability_form = reachability_form
        self.mapping = mapping

    def __repr__(self):
        return "ProblemResult(status=%s, result=%s)" % (self.status, self.reachability_form)