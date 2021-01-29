from . import ProblemFormulation, ProblemResult, Subsystem, construct_MILP, certificate_size, construct_indicator_graph
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
        self._incumbent = SolverResult("undefined", None, float("inf"))

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
            self.__add_constraint(indicatoridx, 1)
        
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

    def __remove_constraint(self, indicatoridx):
        # print("removing constraint %d"  % indicatoridx)
        constr = self._added_constraints[indicatoridx]
        self._model.remove_constraint(constr)
        del self._added_constraints[indicatoridx]

    def __add_constraint(self, indicatoridx, val):
        # print("adding constraint %d:=%g"  % (indicatoridx,val))
        indicatorvar = self._indicator_var_to_idx.inv[indicatoridx]
        constr = self._model.add_constraint([(indicatorvar, 1)], "=", val) 
        self._added_constraints[indicatoridx] = constr

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
            self._incumbent = SolverResult( "optimal", solution, val )
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
            if valcur != -1 and valnode != -1:
                self.__remove_constraint(diffidx)
                self.__add_constraint(diffidx, valnode)
            elif valcur != -1 and valnode == -1:
                self.__remove_constraint(diffidx)
            elif valcur == -1 and valnode != -1:
                self.__add_constraint(diffidx, valnode)
        
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

class ColumnGeneration(ProblemFormulation):
    def __init__(self, solver="cbc"):
        super().__init__()
        self.solver = solver

    @property
    def details(self):
        """Returns a dictionary with method details. Keys are "type", "mode" and "solver"."""
        return {
            "type" : "Column Generation",
            "solver" : self.solver
        }

    def _solveiter(self, reach_form, threshold, mode, labels, timeout=None):
        prob = BnBProblem(reach_form, labels, threshold, mode, self.solver)
        bnb.solve(prob, queue_strategy="bound")
        
        incumbent = prob._incumbent 
        if incumbent.status != "optimal":
            yield ProblemResult(incumbent.status, None, None, None)
        else:
            certsize = certificate_size(reach_form, mode)
            certificate = incumbent.result_vector[:certsize]
            witness = Subsystem(reach_form, certificate, mode)
            yield ProblemResult("success", witness, incumbent.value, certificate)