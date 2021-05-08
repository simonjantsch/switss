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
        """implements the branch-and-bound problem-class from pybnb.

        :param rf: the reachability form that should be minimized.
        :type rf: ReachabilityForm
        :param labels: list of strings or None; the state/state-action-labeling that should be minimized instead 
        :type labels: List[str]
        :param thr: the threshold
        :type thr: float
        :param mode: either "min" or "max"
        :type mode: str
        :param solver: the solver that should be used, defaults to "cbc"
        :type solver: str, optional
        """
        assert mode in ["min", "max"]

        # dictionary that contains the current set of added constraints when 
        # descending down the tree of subproblems
        self._added_constraints = dict()
        self._solved_result = None
        # contains the best result known so far
        self._incumbent = SolverResult("undefined", None, None, float("inf")) 
        self._mode = mode
        self._solver = solver
        self._thr = thr
        self._rf = rf
        # construct problem without any further constraints, which is why "relaxed" is set to True
        self._model, self._indicator_to_groups = construct_MILP(rf, thr, mode, 
                                                                labels=labels, relaxed=True, 
                                                                upper_bound_solver=solver,
                                                                modeltype="gurobi" if solver=="gurobi" else "pulp") 
        # create mapping from indicator variables to state/state-action-variables
        self._indicator_var_to_idx = bidict({ 
          idx: i for i, idx in enumerate(self._indicator_to_groups.keys()) })

        # construct indicator graph, i.e. which group of states/state-action-pairs is connect
        # to which group of (other) states/state-action-pairs?
        self._indicator_graph = construct_indicator_graph(
            rf, mode, self._indicator_to_groups, 
            self._indicator_var_to_idx)
        # this basically contains the indicator variable assignments; i.e.: -1 for indicators 
        # that are unassigned, 1 for active indicators and 0 for inactive indicators 
        self._indicatorstate = -np.ones(len(self._indicator_var_to_idx.keys()))

        # set all indicator variables to 1 that group the initial state
        for indicatorvar in self._indicator_to_groups.inv[self._rf.initial]:
            indicatoridx = self._indicator_var_to_idx[indicatorvar]
            self._indicatorstate[indicatoridx] = 1
            self.__add_constraint(indicatorvar, 1)
        # initialize set of indicator variable that can be reached from the initial
        # indicator variables (those which were set to 1 in the lines above)
        self._candidates = set()
        for indicatorvar in self._indicator_to_groups.inv[self._rf.initial]:
            indicatoridx = self._indicator_var_to_idx[indicatorvar]
            self.__add_successor_candidates(indicatoridx)
        # initialize set of indicator variables that group states/state-action-pairs
        # which reach the target state
        self._reaching_target = set()
        for pred, _, _ in self._rf.system.predecessors( self._rf.target_state_idx ):
            if pred not in [ self._rf.target_state_idx, self._rf.fail_state_idx ]:
                for indvar in self._indicator_to_groups.inv[pred]:
                    indidx = self._indicator_var_to_idx[indvar]
                    self._reaching_target.add(indidx)

    def __add_successor_candidates(self, indicatoridx):
        """looks at the indicator graph and sets all successor-indicator entries to 1

        :param indicatoridx: the entry that should be looked at
        :type indicatoridx: int
        """
        for succ,_,_ in self._indicator_graph.successors(indicatoridx):
            if self._indicatorstate[succ] == -1:
                self._candidates.add(succ)

    def __remove_constraint(self, var):
        """removes the constraint of a varible, i.e. remove var=val constraints,
        where val is some arbitrary value

        :param var: the variable
        :type var: int
        """
        # print("removing constraint %d"  % indicatoridx)
        constr = self._added_constraints[var]
        self._model.remove_constraint(constr)
        del self._added_constraints[var]

    def __add_constraint(self, var, val):
        """adds a constraint to a variable, i.e. var=val

        :param var: the variable that should be constrained to a value
        :type var: int
        :param val: the value it should be assigned to
        :type val: float
        """
        # print("adding constraint %d:=%g"  % (indicatoridx,val))
        constr = self._model.add_constraint([(var, 1)], "=", val) 
        self._added_constraints[var] = constr

    def sense(self):
        return bnb.minimize

    def objective(self):
        """returns an upper bound for the problem. we do this by using the qs-heuristic

        :return: the upper bound
        :rtype: float
        """
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
        """compute lower bound of the problem by solving the lp-relaxation

        :return: the lower bound
        :rtype: float
        """
        # print("(bound) indicatorstate", self._indicatorstate)
        if self._solved_result is None:
            # remind that the model is a lp, not a milp
            result = self._model.solve(solver=self._solver)
            self._solved_result = result
        
        if self._solved_result.status == "optimal":
            return self._solved_result.value
        else:
            return self.infeasible_objective()

    def save_state(self, node):
        """standard pybnb-function: we save the state of our current node (i.e. subproblem). this means saving the current state of the indicator values (unconstrained or constrained to specific values) and the current set of candidate variables which can be branched on
        """
        node.state = self._indicatorstate.copy(), self._candidates.copy()
        
    def load_state(self, node):
        """standard pybnb-function: we load the state of a node (i.e. subproblem) into this object. this means removing the current constraints and/or the set of candidates and replacing both with the parameters of the loaded node
        """
        # check where node state and current state deviate
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
        """selects an indicator variable that should be branched on. current solution is by selecting the indicator variable that is closest to 1, according to its relaxed value"""
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
        """select indicator variable and add left and right subproblem by addign constraints "var=0" and "var=1".
        """
        # select variable
        chosen_indicator = self.__choose_candidate_indicator()
        if chosen_indicator is None:
            return
        # remove from set of candidates, since we constrain it to a specific value
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
        """implementation of a simple branch and price-procedure using the convexification approach. basically overrides only the bounding-method of the BnBProblem-class with the column-generation-approach, and also extends the formulation by new lambda-variables and further reformulation-specific constraints.

        :param rf: the reachability that should be minimized
        :type rf: ReachabilityForm
        :param labels: the set of labels if label-based-minimization is choosen
        :type labels: List[str] or None
        :param thr: the threshold
        :type thr: float
        :param mode: either "min" or "max"
        :type mode: str
        :param solver: the solver that should be used, defaults to "cbc"
        :type solver: str, optional
        """
        super().__init__(rf, labels, thr, mode, solver=solver)
        # model -> cg form
        # initialize P = { p } and add variable
        # setting p = (1,...1) will definetly enable a feasible solution for the primal problem
        p = np.ones(self._indicatorstate.shape)
        lambda_p = self._model.add_variables("real") # add a single variable lambda_p

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

    def add_column(self, p):
        """adds a new column to the primal problem, corresponding to a p in P.

        :param p: the vector that should be added
        :type p: np.ndarray[#indicator-variables]
        :return: the new lambda_p-variable index
        :rtype: int
        """
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
        """solves the subproblem as given by the convexification. current implementation
        does not strengthen the relaxation, since we allow P = { 0,1 }^#indicator-vars and therefore any possible value in [0,1]^#indicator-vars for sigma.

        :param result: result of primal and dual problem that yield dual values for gamma and delta
        :type result: SolverResult
        :param return: the "best" element p in P
        :type rtype: np.ndarray[#indicator-vars]
        """
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
        """the column-generation-algorithm used to solve the linear relaxation of the model

        :return: the resulting values
        :rtype: SolverResult
        """
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
        """implements the ProblemFormulation-class by use of a custom branch and bound/price-solver.

        :param solver: the solver that should be used (when solving linear relaxations), defaults to "cbc"
        :type solver: str, optional
        :param branch_and_price: if normal branch and bound or -price should be used, defaults to False
        :type branch_and_price: bool, optional
        """
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