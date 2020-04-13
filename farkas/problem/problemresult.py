from ..model import ReachabilityForm
from . import MinimalWitness

class ProblemResult:
    def __init__(self, status, witness):
        self.status = status
        self.witness = witness

    def __repr__(self):
        return "ProblemResult(status=%s, witness=%s)" % (self.status, self.witness)