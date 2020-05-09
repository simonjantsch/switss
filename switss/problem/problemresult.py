from ..model import ReachabilityForm
from . import Subsystem

class ProblemResult:
    def __init__(self, status, subsystem, value, farkas_cert):
        self.status = status
        self.subsystem = subsystem
        self.value = value
        self.farkas_cert = farkas_cert

    def __repr__(self):
        return "ProblemResult(status=%s, subsystem=%s, value=%s)" % (self.status, self.subsystem, self.value)
