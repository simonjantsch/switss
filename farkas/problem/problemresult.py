from ..model import ReachabilityForm
from . import Subsystem

class ProblemResult:
    def __init__(self, status, subsystem):
        self.status = status
        self.subsystem = subsystem

    def __repr__(self):
        return "ProblemResult(status=%s, subsystem=%s)" % (self.status, self.subsystem)