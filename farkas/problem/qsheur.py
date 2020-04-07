from . import ProblemFormulation, ProblemResult
from farkas.solver import LP, Wrapper
from farkas.model.reachability_form import induced_subsystem

import numpy as np

class QSHeur(ProblemFormulation):
    def __init__(self,
                 threshold,
                 min_or_max,
                 inital_weights = None,
                 iterations = 3,
                 upd_fct = lambda x: 1e7 if x == 0 else 1 / x,
                 solver_name="cbc"):
        super().__init__()
        assert min_or_max in ["min","max"]
        assert solver_name in ["gurobi","cbc"]

        self.min_or_max = min_or_max
        self.threshold = threshold
        self.iterations = iterations
        self.solver = Wrapper(solver_name)
        self.initial_weights = None
        self.upd_fct = upd_fct

    def solve(self, reach_form):
        if self.min_or_max == "min":
            return self.solve_min(reach_form)
        else:
            return self.solve_max(reach_form)

    def solve_min(self, reach_form):
        problem_results = dict()
        _,N = reach_form.P.shape

        if self.initial_weights == None:
            current_weights = np.ones(N)
        else:
            assert self.initial_weights.size == N
            current_weights = self.initial_weights

        fark_matr,fark_rhs = reach_form.fark_min_constraints(self.threshold)

        for i in range(0,self.iterations):

            heur_i_lp = LP(fark_matr,fark_rhs,current_weights,lowBound=0,upBound=1)
            heur_i_result = self.solver.solve(heur_i_lp)

            if heur_i_result.status == "optimal":
                res_vector = heur_i_result.result
                to_one_if_positive = np.vectorize(lambda x: 1 if x > 0 else 0)
                induced_states = to_one_if_positive(res_vector[:N])
                subsys,mapping = induced_subsystem(reach_form,induced_states)
                problem_results[i] = ProblemResult("success",subsys,mapping)

                for x in range(0,N):
                    current_weights[x] = self.upd_fct(heur_i_result.result[x])

            else:
                # failed to optimize LP
                problem_results[i] = ProblemResult(heur_i_result.status)

        return problem_results

    def solve_max(self, reach_form):
        problem_results = dict()
        C,N = reach_form.P.shape

        if self.initial_weights == None:
            current_weights = np.ones(C)
        else:
            assert self.initial_weights.size == C
            current_weights = self.initial_weights

        fark_matr,fark_rhs = reach_form.fark_max_constraints(self.threshold)

        for i in range(0,self.iterations):
            heur_i_lp = LP(fark_matr,fark_rhs,current_weights,lowBound=0)
            heur_i_result = self.solver.solve(heur_i_lp)

            if heur_i_result.status == "optimal":
                res_vector = heur_i_result.result
                induced_states = np.zeros(N)
                for index in range(0,C):
                    if res_vector[index] > 0:
                        (st,act) = reach_form.index_by_state_action.inv[index]
                        induced_states[st] = 1

                subsys,mapping = induced_subsystem(reach_form,induced_states)
                problem_results[i] = ProblemResult("success",subsys,mapping)

                for x in range(0,C):
                    current_weights[x] = self.upd_fct(heur_i_result.result[x])

            else:
                # failed to optimize LP
                problem_results[i] = ProblemResult(heur_i_result.status)

        return problem_results
