"""This module is a wrapper for the PuLP library, which is capable of 
solving LP/MILP instances by using different kinds of solvers (like Gurobi or CBC).
The wrapper defines custom MILP and LP classes in order to simplify the instantiation of 
problems from coefficient vectors and matrices.

.. code-block::

    from farkas.solver import MILP, LP, Wrapper
    wrapper = Wrapper("cbc")
    # example for a LP instance. optimal result should be x_opt=[1.5,7.0].
    A = np.matrix([[2,1],[4,-1],[-8,2],[-1,0],[0,-1]])
    b = np.array([10,8,2,0,0])
    opt = np.array([-1,-1])
    lp = LP(A,b,opt)
    result = wrapper.solve(lp)
    print(result)

"""
from .milp import MILP, LP
from .solverresult import SolverResult
from .wrapper import Wrapper