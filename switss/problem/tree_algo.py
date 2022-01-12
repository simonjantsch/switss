from . import ProblemFormulation, ProblemResult, Subsystem, QSHeur
from switss.model.dtmc import DTMC

try:
    from switss.utils.tree_decomp import min_witnesses_from_tree_decomp
except:
    print("note: switss is installed without tree-algo support.")

class TreeAlgo(ProblemFormulation):
    def __init__(self, partition, known_upper_bound = None, tol=1e-12, solver="cbc"):
        self.partition = partition
        self.known_upper_bound = known_upper_bound
        self.tol = tol
        self.solver = solver

    @property
    def details(self):
        return {
            "type" : "TreeAlgo",
            "known_upper_bound" : str(self.known_upper_bound),
            "tol" : str(self.tol)
        }

    #min_witnesses_from_tree_decomp(rf,partition,thr,known_upper_bound)
    def _solveiter(self, reach_form, threshold, mode, labels, timeout=None):
        assert type(reach_form.system) == DTMC

        upper_bound = self.known_upper_bound
        if upper_bound == None:
            qsheur = QSHeur(solver=self.solver,iterations=5)
            res_max = qsheur.solve(reach_form,threshold,"max")
            res_min = qsheur.solve(reach_form,threshold,"min")
            print("res_max: " + str(res_max.value))
            print("res_min: " + str(res_min.value))
            if res_min.status != "success" or res_max.status != "success":
                yield ProblemResult("infeasible",None,None,None)

            upper_bound = min(res_max.value, res_min.value)

            assert upper_bound != 0

            # # this is just to circumvent an apparent error in qsheur
            # # TODO fix this
            # if upper_bound == 0:
            #     upper_bound = max(res_max.value, res_min.value)
            # if upper_bound == 0:
            #     upper_bound = None

        res = min_witnesses_from_tree_decomp(reach_form,
                                             self.partition,
                                             threshold,
                                             known_upper_bound = upper_bound,
                                             timeout=timeout)
        if res == -1:
            yield ProblemResult("notsolved",None,None,None)

        yield ProblemResult("success", None, res, None)
