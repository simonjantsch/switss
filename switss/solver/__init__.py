"""This module is a wrapper for the PuLP library, which is capable of 
solving LP/MILP instances by using different kinds of solvers (like Gurobi or CBC).
The wrapper defines custom MILP and LP classes in order to simplify the instantiation of 
problems from coefficient vectors and matrices."""
from .solverresult import SolverResult
from .milp import MILP, LP, GurobiMILP