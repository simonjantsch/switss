from . import ProblemFormulation, \
              ProblemResult, \
              Subsystem, \
              construct_MILP, \
              certificate_size, \
              construct_indicator_graph, \
              construct_RMP
from switss.solver import SolverResult
from switss.model import ReachabilityForm
from typing import Set, Dict, List

from bidict import bidict
import pybnb as bnb
import numpy as np


class BnBProblem(bnb.Problem):
    def __init__(self, 
                 rf : ReachabilityForm, 
                 labels : List[str],
                 thr : float, 
                 mode : str, 
                 solver="cbc"):
        
        assert mode in ["min", "max"]

        self._added_constraints = dict()
        self._solved_result = None
        self._incumbent = SolverResult("undefined", None, None, float("inf"))

        self._mode = mode
        self._solver = solver
        self._thr = thr
        self._rf = rf
        self._model, self._indicator_to_groups = construct_MILP(rf, thr, mode, 
                                                                labels=labels, relaxed=True, 
                                                                upper_bound_solver=solver,
                                                                modeltype="gurobi" if solver=="gurobi" else "pulp")

        self._indicator_var_to_idx = bidict({ idx: i for i, idx in enumerate(self._indicator_to_groups.keys()) })

        self._indicator_graph = construct_indicator_graph(
            rf, mode, self._indicator_to_groups, 
            self._indicator_var_to_idx)
        self._indicatorstate = -np.ones(len(self._indicator_var_to_idx.keys()))

        for indicatorvar in self._indicator_to_groups.inv[self._rf.initial]:
            indicatoridx = self._indicator_var_to_idx[indicatorvar]
            self._indicatorstate[indicatoridx] = 1
            self.__add_constraint(indicatorvar, 1)
        
        self._candidates = set()
        for indicatorvar in self._indicator_to_groups.inv[self._rf.initial]:
            indicatoridx = self._indicator_var_to_idx[indicatorvar]
            self.__add_successor_candidates(indicatoridx)

        self._reaching_target = set()
        for pred, _, _ in self._rf.system.predecessors( self._rf.target_state_idx ):
            if pred not in [ self._rf.target_state_idx, self._rf.fail_state_idx ]:
                for indvar in self._indicator_to_groups.inv[pred]:
                    indidx = self._indicator_var_to_idx[indvar]
                    self._reaching_target.add(indidx)

    def __add_successor_candidates(self, indicatoridx):
        for succ,_,_ in self._indicator_graph.successors(indicatoridx):
            if self._indicatorstate[succ] == -1:
                self._candidates.add(succ)

    def __remove_constraint(self, var):
        # print("removing constraint %d"  % indicatoridx)
        constr = self._added_constraints[var]
        self._model.remove_constraint(constr)
        del self._added_constraints[var]

    def __add_constraint(self, var, val):
        # print("adding constraint %d:=%g"  % (indicatoridx,val))
        constr = self._model.add_constraint([(var, 1)], "=", val) 
        self._added_constraints[var] = constr

    def sense(self):
        return bnb.minimize

    def objective(self):
        # print("(objective) indicatorstate", self._indicatorstate)
        if self._solved_result is None:
            result = self._model.solve(solver=self._solver)
            self._solved_result = result

        # construct solution from LP relaxation
        indices = list(self._indicator_to_groups.keys())
        solution = self._solved_result.result_vector.copy()
        solution[indices] = solution[indices] > 0
        val = solution[indices].sum()

        # check if solution is the new incumbent
        if val < self._incumbent.value:
            self._incumbent = SolverResult( "optimal", solution, None, val )
        return val

    def bound(self):
        # print("(bound) indicatorstate", self._indicatorstate)
        if self._solved_result is None:
            result = self._model.solve(solver=self._solver)
            self._solved_result = result
        
        if self._solved_result.status == "optimal":
            return self._solved_result.value
        else:
            return self.infeasible_objective()

    def save_state(self, node):
        node.state = self._indicatorstate.copy(), self._candidates.copy()
        
    def load_state(self, node):
        # check where node's state and current's state deviate
        diffindices, = (node.state[0] != self._indicatorstate).nonzero()
        for diffidx in diffindices:
            valnode = node.state[0][diffidx]
            valcur  = self._indicatorstate[diffidx]
            vardiffidx = self._indicator_var_to_idx.inv[diffidx]
            if valcur != -1 and valnode != -1:
                self.__remove_constraint(vardiffidx)
                self.__add_constraint(vardiffidx, valnode)
            elif valcur != -1 and valnode == -1:
                self.__remove_constraint(vardiffidx)
            elif valcur == -1 and valnode != -1:
                self.__add_constraint(vardiffidx, valnode)
        
        self._indicatorstate = node.state[0].copy()
        self._candidates = node.state[1].copy()
        self._solved_result = None

    def __choose_candidate_indicator(self):
        # otherwise, select indicator that is most likely to be 1
        # (according to the relaxation)
        # also don't branch on variables that are set to 0 or 1 in the relaxed solution
        bestindicatoridx, bestindicatorval = None, None
        for indicatoridx in self._candidates:
            val = self._solved_result.result_vector[self._indicator_var_to_idx.inv[indicatoridx]]
            if val == 0 or val == 1:
                continue
            if bestindicatoridx is None or abs(bestindicatorval - 1) > abs(val - 1):
                bestindicatoridx = indicatoridx
                bestindicatorval = val

        return bestindicatoridx

    def branch(self):
        chosen_indicator = self.__choose_candidate_indicator()
        if chosen_indicator is None:
            return

        self._candidates.remove(chosen_indicator)

        # case 1: set candidate to 0 (if possible)
        disabled_indicators = self._indicatorstate == 0
        disabled_indicators[chosen_indicator] = True 
        blocklist = set( disabled_indicators.nonzero()[0] )
        # this mask contains the set of states that reach target without using disabled states
        rt = { indidx for indidx in self._reaching_target if self._indicatorstate[indidx] != 0 }
        mask = self._indicator_graph.reachable(rt, "backward", blocklist=blocklist)
        # if a state is "enabled" then there must be a path from goal to this state
        if ((self._indicatorstate!=1) | mask).all():
            # branching
            child0 = bnb.Node()
            self._indicatorstate[chosen_indicator] = 0
            self.save_state(child0)
            yield child0

        # case 2: chosen set to 1
        # first: restore parent state
        # self.load_state(parent_node)
        child1 = bnb.Node()
        self._indicatorstate[chosen_indicator] = 1
        self.__add_successor_candidates(chosen_indicator)
        self.save_state(child1)
        yield child1

        # this becomes important in "load_state"
        self._indicatorstate[chosen_indicator] = -1

class BnPProblem(BnBProblem):
    def __init__(self, 
                 rf : ReachabilityForm, 
                 labels : List[str],
                 thr : float, 
                 mode : str, 
                 solver="cbc"):
        super().__init__(rf, labels, thr, mode, solver=solver)
        # model -> cg form
        # initialize P = { p } and add variable
        p = np.ones(self._indicatorstate.shape)
        lambda_p = self._model.add_variables("real")
        # self._P = { self.__p_to_id(p): lambda_p } # dictionary mapping lambda-variables to p-vectors

        # add non-negativity-constraint
        self._model.add_constraint([(lambda_p, 1)], ">=", 0)
        # add convexity-constraint 
        # sum_{p in P} lambda_p = 1
        self._convexity_constraint = self._model.add_constraint([(lambda_p, 1)], "=", 1)
        # add linear-combination-constraint 
        # sigma(l) = sum_{p in P} lambda_p p(l) 
        # <=> sigma(l) - sum_{p in P} lambda_p p(l) = 0
        self._lincomb_constraints = []
        for indicatoridx in range(self._indicatorstate.shape[0]):
            indicatorvar = self._indicator_var_to_idx.inv[indicatoridx]
            constr = self._model.add_constraint([(indicatorvar, 1), (lambda_p, -p[indicatoridx])], "=", 0)
            self._lincomb_constraints.append( constr )

    
    def __p_to_id(self, p):
        return "".join(map(str, p))


    def __id_to_p(self, id):
        return np.array([int(c) for c in id])


    def add_column(self, p):
        lambda_p = self._model.add_variables("real")
        # add non-zero-constraint
        self._model.add_constraint([(lambda_p, 1)], ">=", 0)
        # add to convexity-constraint
        self._model.add_to_constraint(self._convexity_constraint, 1, lambda_p)
        # add to linear-combination-constraint
        for idx, constr in enumerate(self._lincomb_constraints):
            self._model.add_to_constraint(constr, -p[idx], lambda_p)
        # add variable to variable set
        return lambda_p


    def solve_subproblem(self, result):
        delta = result.dual_result_vector[self._convexity_constraint]
        gamma = result.dual_result_vector[self._lincomb_constraints]
        # print(delta, gamma, len(gamma))
        # return a system that fulfils the definition of P!
        p = 1*(gamma < 0)
        # print(" ".join(map(str,p)))
        diff = delta - gamma.dot(p)
        print(diff)
        if diff > 0:
            return p
        else:
            return None 

    
    def column_generation(self):
        while True:
            result = self._model.solve(solver=self._solver)
            if result.status == "optimal":
                subproblem_result = self.solve_subproblem(result)
                if subproblem_result is None:
                    return result
                else:
                    var = self.add_column(subproblem_result)
            else:
                return None


    def bound(self):
        if self._solved_result is None:
            self._solved_result = self.column_generation()
            
        return super().bound()


class BnBFormulation(ProblemFormulation):
    def __init__(self, solver="cbc", branch_and_price=False):
        super().__init__()
        self.solver = solver
        self.bnp = branch_and_price

    @property
    def details(self):
        """Returns a dictionary with method details. Keys are "type", "mode" and "solver"."""
        return {
            "type" : "Column Generation",
            "solver" : self.solver
        }

    def _solveiter(self, reach_form, threshold, mode, labels, timeout=None):
        problemcls = BnPProblem if self.bnp else BnBProblem  
        prob = problemcls(reach_form, labels, threshold, mode, self.solver)
        bnb.solve(prob, queue_strategy="bound")
        
        incumbent = prob._incumbent 
        if incumbent.status != "optimal":
            yield ProblemResult(incumbent.status, None, None, None)
        else:
            certsize = certificate_size(reach_form, mode)
            certificate = incumbent.result_vector[:certsize]
            witness = Subsystem(reach_form, certificate, mode)
            yield ProblemResult("success", witness, incumbent.value, certificate)