from . import ProblemFormulation, construct_MILP, certificate_size, construct_indicator_graph
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
        
        assert mode == "min"

        self._mode = mode
        self._solver = solver
        self._thr = thr
        self._rf = rf
        self._model, self._indicator_to_groups = construct_MILP(rf, thr, mode, 
                                                                labels=labels, relaxed=True, 
                                                                upper_bound_solver=solver)

        self._indicator_var_to_idx = bidict({ idx: i for i, idx in enumerate(self._indicator_to_groups.keys()) })

        self._indicator_graph = construct_indicator_graph(
            rf, mode, self._indicator_to_groups, 
            self._indicator_var_to_idx)
        self._indicatorstate = -np.ones(len(self._indicator_var_to_idx.keys()))

        for indicatorvar in self._indicator_to_groups.inv[self._rf.initial]:
            indicatoridx = self._indicator_var_to_idx[indicatorvar]
            self._indicatorstate[indicatoridx] = 1
        
        self._candidates = set()
        for indicatorvar in self._indicator_to_groups.inv[self._rf.initial]:
            indicatoridx = self._indicator_var_to_idx[indicatorvar]
            self.__add_successor_candidates(indicatoridx)

        self._added_constraints = set()

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

    def __add_constraint(self, indicatoridx, val):
        indicatorvar = self._indicator_var_to_idx.inv[indicatoridx]
        return self._model.add_constraint([(indicatorvar, 1)], "=", val) 

    def sense(self):
        return bnb.minimize

    def objective(self):
        result = self._model.solve(self._solver)
        indices = list(self._indicator_to_groups.keys())
        return (result.result_vector[indices] > 0).sum()

    def bound(self):
        result = self._model.solve(self._solver)
        return result.value
    
    def save_state(self, node):
        node.state = self._indicatorstate.copy(), self._candidates.copy()
        
    def load_state(self, node):
        # remove old constraints from model
        for constr in self._added_constraints:
            self._model.remove_constraint(constr)

        self._indicatorstate = node.state[0].copy()
        self._candidates = node.state[1].copy() 

        # add new constraints
        self._added_constraints = set()
        for ind in (self._indicatorstate != -1).nonzero()[0]:
            val = self._indicatorstate[ind]
            constr = self.__add_constraint(ind, val)
            self._added_constraints.add(constr)

    def branch(self):
        if len( self._candidates ) == 0:
            return

        # chose a random canidate
        chosen_indicator = self._candidates.pop()

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
        results = bnb.solve(prob)
        print( results.objective, results.bound )
        yield results