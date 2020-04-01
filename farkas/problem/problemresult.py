from ..model import ReachabilityForm

class ProblemResult:
    def __init__(self, reachability_form, mapping):
        self.reachability_form = reachability_form
        self.mapping = mapping