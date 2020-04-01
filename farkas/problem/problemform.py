from abc import ABC, abstractmethod

class ProblemFormulation:
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, reachability_form):
        pass