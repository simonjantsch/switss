from . import ProblemFormulation, ProblemResult, Subsystem, QSHeur
from switss.model.dtmc import DTMC
from switss.utils.tree_decomp import min_witnesses_from_tree_decomp

class TreeAlgo(ProblemFormulation):
    def __init__(self, partition, known_upper_bound = None, tol=1e-12):
        self.partition = partition
        self.known_upper_bound = known_upper_bound
        self.tol = tol

    @property
    def details(self):
        return {
            "type" : "TreeAlgo",
            "known_upper_bound" : str(self.known_upper_bound),
            "tol" : str(self.tol)
        }

    #min_witnesses_from_tree_decomp(rf,partition,thr,known_upper_bound)
    def _solveiter(self, reach_form, threshold, labels, timeout=None):
        assert type(reach_form.system) == DTMC

        upper_bound = self.known_upper_bound
        if upper_bound == None:
            qsheur_max = QSHeur(mode="max",solver="cbc",iterations=5)
            qsheur_min = QSHeur(mode="min",solver="cbc",iterations=5)
            res_max = qsheur_max.solve(reach_form,threshold)
            res_min = qsheur_min.solve(reach_form,threshold)
            print("res_max: " + str(res_max.value))
            print("res_min: " + str(res_min.value))
            if res_min.status != "success" or res_max.status != "success":
                yield ProblemResult("infeasible",None,None,None)
            upper_bound = min(res_max.value, res_min.value)
            # this is just to circumvent an apparent error in qsheur
            # TODO fix this
            if upper_bound == 0:
                upper_bound = max(res_max.value, res_min.value)
            if upper_bound == 0:
                upper_bound = None

        res = min_witnesses_from_tree_decomp(reach_form,
                                             self.partition,
                                             threshold,
                                             known_upper_bound = upper_bound,
                                             timeout=timeout)
        if res == -1:
            yield ProblemResult("notsolved",None,None,None)

        yield ProblemResult("success", None, res, None)
