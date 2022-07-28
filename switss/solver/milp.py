from ..utils import cast_dok_matrix
from . import SolverResult
from scipy.sparse import dok_matrix
import pulp
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except:
    print("milp.py: gurobipy is not installed. Please install it if you intend to use it.")

class MILP:
    """
    A MILP can either be initialized through a specification of coefficient matrices and -vectors 
    or manually, i.e. by adding variables, constraints and the objective function by hand.

    .. code-block::

        # example for a MILP instance. optimal result should be x_opt=[2,6].
        # this is the same as the LP instance but with an added integer constraint for the first variable.
        A = np.matrix([[2,1],[4,-1],[-8,2],[-1,0],[0,-1]])
        b = np.array([10,8,2,0,0])
        opt = np.array([1,1])
        domains = ["integer", "real"]
        milp = MILP.from_coefficients(A,b,opt,domains,objective="max")

        result = milp.solve(solver="cbc")
        print(result)

    .. code-block::

        # the same as the last MILP, but constraints and objective function are added manually.
        milp = MILP(objective="max")
        var1, var2 = milp.add_variables("integer", "real")
        milp.add_constraint([(var1, 2), (var2, 1)], "<=", 10)
        milp.add_constraint([(var1, 4), (var2, -1)], "<=", 8)
        milp.add_constraint([(var1, -8), (var2, 2)], "<=", 2)
        milp.add_constraint([(var1, 1)], ">=", 0)
        milp.add_constraint([(var2, 1)], ">=", 0)
        milp.set_objective_function([(var1, 1), (var2, 1)])

        result = milp.solve(solver="cbc")
        print(result)
    """
    def __init__(self, objective="min"):
        """Initializes an empty MILP.
        
        :param objective: Whether the problem should minimize or maximize ("min" or "max"), defaults to "min"
        :type objective: str, optional
        """        
        assert objective in ["min", "max"], "objective must be either 'min' or 'max'"
        objective = { "min" : pulp.LpMinimize, "max" : pulp.LpMaximize }[objective]
        self.__pulpmodel = pulp.LpProblem("",objective)
        self.__variables = [] 
        self.__constraints = []
        self.__set_objective_function = False

    def solve(self, solver="cbc",timeout=None,print_output=False):
        """Solves this problem and returns the problem result.
        
        :param solver: The solver that should be used. Currently supported are "cbc", "gurobi", "glpk" and "cplex", defaults to "cbc"
        :type solver: str, optional
        :return: Result.
        :rtype: solver.SolverResult
        """        
        assert solver in ["gurobi","cbc","glpk","cplex"], "solver must be in ['gurobi','cbc','glpk','cplex']"
        if timeout != None:
            assert isinstance(timeout,int), "timeout must be specified in seconds as integer value"

        if solver == "gurobi":
            gurobi_options = [
                ("MIPGap",0), ("MIPGapAbs",0), ("FeasibilityTol",1e-9),\
                ("IntFeasTol",1e-9),("NumericFocus",3)]
            if timeout != None:
                gurobi_options.append(("TimeLimit",str(timeout)))
            self.__pulpmodel.setSolver(pulp.GUROBI_CMD(msg=print_output,options=gurobi_options))
        elif solver == "cbc":
            cbc_options = ["--integerT","0"]
            self.__pulpmodel.setSolver(
                pulp.PULP_CBC_CMD(gapRel=1e-9,timeLimit=timeout,msg=print_output,options=cbc_options))
        elif solver == "glpk":
            glpk_options = ["--tmlim",str(timeout)] if timeout != None else []
            self.__pulpmodel.setSolver(pulp.GLPK_CMD(msg=print_output,options=glpk_options))
        elif solver == "cplex":
            self.__pulpmodel.setSolver(pulp.CPLEX_PY(msg=print_output,timeLimit=timeout))

        self.__pulpmodel.solve()

        status = {   1:"optimal",
                     0:"notsolved",
                    -1:"infeasible", 
                    -2:"unbounded", 
                    -3:"undefined"}[self.__pulpmodel.status]
        result_vector = np.array([var.value() for var in self.__variables])
        value = self.__pulpmodel.objective.value()

        return SolverResult(status, result_vector, value)

    def _assert_expression(self, expression):
        for idx,(var,coeff) in enumerate(expression):
            assert var >= 0 and var < len(self.__variables), "Variable %s does not exist (@index=%d)." % (var, idx)
            assert coeff == float(coeff), "Coefficient coeff=%s is not a number (@index=%d)." % (coeff, idx)

    def _expr_to_pulp(self, expression):
        for var, coeff in expression:
            yield self.__variables[var], coeff

    def set_objective_function(self, expression):
        """Sets the objective function of the form

        .. math::
            
            \sum_j \sigma_j x_j

        where :math:`\sigma_j` indicates a coefficient and :math:`x_j` a variable.
        
        :param expression: Sum is given as a list of variable/coefficient pairs. Each pair has the coefficient on the
            right and the variable on the left.
        :type expression: List[Tuple[int,float]]
        """        
        self._assert_expression(expression)
        if not self.__set_objective_function:
            self.__set_objective_function = True
            self.__pulpmodel += pulp.LpAffineExpression(self._expr_to_pulp(expression))
        else:
            for var,coeff in expression:
                self.__pulpmodel.objective[self.__variables[var]] = coeff
        
    def add_constraint(self, lhs, sense, rhs):
        """Adds a constraint of the form

        .. math::

            \sum_{j} a_j x_j \circ b
        
        where :math:`\circ \in \{ \leq, =, \geq \}`, :math:`a_j` indicates a coefficient and :math:`x_j` a variable.
        
        :param lhs: Left side of the equation, given as a list of variable/coefficient pairs. Each pair has the coefficient on the
            right and the variable on the left.
        :type lhs: List[Tuple[int,float]]
        :param sense: Type of equation, i.e. "<=", ">=" or "=".
        :type sense: str
        :param rhs: Right side of the equation, i.e. a number.
        :type rhs: float
        :return: name of the added constraint
        :rtype: str
        """        
        assert sense in ["<=", "=", ">="]
        assert rhs == float(rhs), "Right hand side is not a number: rhs=%s" % rhs 
        self._assert_expression(lhs)

        lhs = pulp.LpAffineExpression(self._expr_to_pulp(lhs))
        sense = { "<=" : pulp.LpConstraintLE, 
                  "=" : pulp.LpConstraintEQ, 
                  ">=" : pulp.LpConstraintGE }[sense]

        constridx = len( self.__constraints )
        name = "c%d" % constridx
        constraint = pulp.LpConstraint(name=name, e=lhs, sense=sense, rhs=rhs)
        self.__pulpmodel += constraint
        self.__constraints.append(constraint)
        
        return constridx

    def remove_constraint(self, constridx):
        """removes a given constraint from the model.

        :param constridx: the name of the constraint
        :type constridx: str
        """        
        self.__pulpmodel.constraints.pop(constridx)

    def add_variables(self, domains):
        """Adds a list of variables to this MILP. Each element in `domains` must be either `integer`, `binary` or `real`.
        
        :return: Index or indices of new variables.
        :rtype: either List[int] or int.
        """        
        var_descr = [ (dom,None,None) for dom in domains ]
        return self.add_variables_w_bounds(var_descr)

    def add_variables_w_bounds(self, var_descr):
        """Adds a list of variables to this MILP. Each element in `var_descr` must be a triple (dom,lb,ub) where dom is either `integer`, `binary` or `real` and lb,ub are floats, or None.

        :return: Index or indices of new variables.
        :rtype: either List[int] or int.
        """
        l = []
        for (domain,lb,ub) in var_descr:
            assert domain in ["integer", "real", "binary"]

            cat = { "real" : pulp.LpContinuous, "integer" : pulp.LpInteger, "binary" : pulp.LpBinary }[domain]
            varidx = len(self.__variables)
            var = pulp.LpVariable("x%d" % varidx, lowBound=lb, upBound=ub, cat=cat)
            self.__variables.append(var)

            if len(var_descr) == 1:
                return varidx
            else:
                l.append(varidx)
        return l

    @classmethod
    def from_coefficients(cls, A, b, opt, domains, sense="<=", objective="min", bounds=None):
        """Returns a Mixed Integer Linear Programming (MILP) formulation of a problem
        
        .. math::

            \min_x/\max_x\ \sigma^T x \quad \\text{ s.t. } \quad Ax \circ b, \ x_i \in \mathbb{D}_i,\ \\forall i=1,\dots,N

        where :math:`\circ \in \{ \leq, \geq \}`, :math:`N` is the number of variables and :math:`M`
        the number of linear constraints. :math:`\mathbb{D}_i` indicates
        the domain of each variable. If `A`, `b` and `opt` are not given as a `scipy.sparse.dok_matrix`, 
        they are transformed into that form automatically.
        
        :param A: Matrix for inequality conditions  (:math:`A`).
        :type A: :math:`M \\times N`-Matrix
        :param b: Vector for inequality conditions  (:math:`b`).
        :type b: :math:`M \\times 1`-Matrix
        :param opt: Weights for individual variables in x (:math:`\sigma`). If None, no objective function will be set.
        :type opt: :math:`N \\times 1`-Matrix
        :param domains: Array of strings, e.g. ["real", "integer", "integer", "binary", ...] which indicates the domain for each variable.
        :type domains: List[str]
        :param sense: "<=" or ">=", defaults to "<="
        :type sense: str, optional
        :param objective: "min" or "max", defaults to "min"
        :type objective: str, optional
        :param bounds: a vector of lower/upper bounds for all variables, optional
        :type bounds: [(float,float)], optional
        :return: The resulting MILP.
        :rtype: solver.MILP
        """

        A = cast_dok_matrix(A).tocsr()
        b = cast_dok_matrix(b)


        opt = cast_dok_matrix(opt)

        model = MILP(objective=objective)

        # initialize problem
        # this adds the variables and the objective function (which is opt^T*x, i.e. sum_{i=1}^N opt[i]*x[i])
        if bounds is not None:
            model.add_variables_w_bounds([(domains[idx],bounds[idx][0],bounds[idx][1]) for idx in range(A.shape[1])])
        else:
            model.add_variables([domains[idx] for idx in range(A.shape[1])])

        model.set_objective_function([(idx, opt[idx,0]) for idx in range(A.shape[1])])

        # this takes quite a lot of time since accessing the rows is inefficient, even for csr-formats.
        # maybe find a way to compute Ax <= b faster.
        # now: add linear constraints: Ax <= b.

        for constridx in range(A.shape[0]):
            # calculates A[constridx,:]^T * x
            row = A.getrow(constridx)
            lhs = [0]*len(row.indices)
            # print(row)
            for i,j,d in zip(range(len(row.indices)), row.indices, row.data):
                lhs[i] = (j, float(d))
            # adds constraint: A[constridx,:]^T * x <= b[constridx]
            model.add_constraint(lhs, sense, b[constridx,0])

        return model

    def __repr__(self):
        return str(self.__pulpmodel)


class LP(MILP):
    """
    An LP can either be initialized through a specification of coefficient matrices and -vectors 
    or manually, i.e. by adding variables, constraints and the objective function by hand.

    .. code-block::

        # example for a LP instance. optimal result should be x_opt=[1.5,7.0].
        A = np.matrix([[2,1],[4,-1],[-8,2],[-1,0],[0,-1]])
        b = np.array([10,8,2,0,0])
        opt = np.array([1,1])
        lp = LP.from_coefficients(A,b,opt,objective="max")

        result = lp.solve(solver="cbc")
        print(result)
    
    .. code-block::

        # the same as the last LP, but constraints and objective function are added manually.
        lp = LP(objective="max")
        var1, var2 = lp.add_variables(2)
        lp.add_constraint([(var1, 2), (var2, 1)], "<=", 10)
        lp.add_constraint([(var1, 4), (var2, -1)], "<=", 8)
        lp.add_constraint([(var1, -8), (var2, 2)], "<=", 2)
        lp.add_constraint([(var1, 1)], ">=", 0)
        lp.add_constraint([(var2, 1)], ">=", 0)
        lp.set_objective_function([(var1, 1), (var2, 1)])

        result = lp.solve(solver="cbc")
        print(result)
    """

    @classmethod
    def from_coefficients(cls, A, b, opt, sense="<=",objective="min"):
        """Returns a Linear Programming (LP) formulation of a problem

        .. math::

            \min_x/\max_x\ \sigma^T x \quad \\text{s.t.}\quad Ax \circ b

        where :math:`\circ \in \{\leq,\geq\}` :math:`N` is the number of
        variables and :math:`M` the number of linear constraints.
        If `A`, `b` and `opt` are not given as a `scipy.sparse.dok_matrix`,
        they are transformed into that form automatically.

        :param A: Matrix for inequality conditions  (:math:`A`).
        :type A: :math:`M \\times N`-Matrix
        :param b: Vector for inequality conditions  (:math:`b`).
        :type b: :math:`M \\times 1`-Matrix
        :param opt: Weights for individual variables in x (:math:`\sigma`).
        :type opt: :math:`N \\times 1`-Matrix
        :param sense: "<=" or ">=", defaults to "<="
        :type sense: str, optional
        :param objective: "min" or "max", defaults to "min"
        :type objective: str, optional
        :return: The resulting LP.
        :rtype: solver.LP
        """
        return MILP.from_coefficients(A,b,opt,["real"]*A.shape[1],sense=sense,objective=objective)

    def add_variables(self, count):
        """Adds a number of variables to the LP.
        
        :param count: The amount of new variables.
        :type count: int
        :return: Index or indices of new variables.
        :rtype: either List[int] or int.
        """        
        return MILP.add_variables(self, *["real"]*count)


class GurobiMILP(MILP):
    def __init__(self, objective="min"):
        """Initializes an empty MILP.
        
        :param objective: Whether the problem should minimize or maximize ("min" or "max"), defaults to "min"
        :type objective: str, optional
        """        

        assert objective in ["min", "max"], "objective must be either 'min' or 'max'"
        self.__model = gp.Model()
        self.__objective = { "min": GRB.MINIMIZE, "max": GRB.MAXIMIZE }[objective]
        self.__variables = [] 
        self.__constraints = [] # collection of (LinExpr, float, Constr) pairs

        self.__model.setParam("OutputFlag", 0)
        self.__model.setParam("MIPGap", 0)
        self.__model.setParam("MIPGapAbs", 0)
        self.__model.setParam("FeasibilityTol", 1e-9)
        self.__model.setParam("IntFeasTol", 1e-9)
        self.__model.setParam("NumericFocus", 3)
        self.__model.setParam('Threads', 4)


    def solve(self, solver, timeout=None,print_output=False):
        if timeout is not None:
            self.__model.setParam('TimeLimit',timeout)
        self.__model.optimize()

        if print_output:
            self.__model.setParam('OutputFlag',1)

        status_dict = { GRB.OPTIMAL: "optimal",
                        GRB.LOADED: "notsolved",
                        GRB.INFEASIBLE: "infeasible", 
                        GRB.UNBOUNDED: "unbounded" }
                        
        status = "undefined"
        if self.__model.status in status_dict:
            status = status_dict[self.__model.status]
        
        if status == "optimal":    
            result_vector = np.array([var.x for var in self.__variables])
            value = self.__model.objVal
            return SolverResult(status, result_vector, value)
        else:
            return SolverResult(status, None, None)

    def _assert_expression(self, expression):
        for idx,(var,coeff) in enumerate(expression):
            assert var >= 0 and var < len(self.__variables), "Variable %s does not exist (@index=%d)." % (var, idx)
            assert coeff == float(coeff), "Coefficient coeff=%s is not a number (@index=%d)." % (coeff, idx)

    def _eval_pulp_expr(self, expression):
        return sum([ self.__variables[var]*coeff for var, coeff in expression ])

    def set_objective_function(self, expression):
        """Sets the objective function of the form

        .. math::
            
            \sum_j \sigma_j x_j

        where :math:`\sigma_j` indicates a coefficient and :math:`x_j` a variable.
        
        :param expression: Sum is given as a list of variable/coefficient pairs. Each pair has the coefficient on the
            right and the variable on the left.
        :type expression: List[Tuple[int,float]]
        """        
        self._assert_expression(expression)
        self.__model.setObjective( 
            self._eval_pulp_expr( expression ), 
            self.__objective 
        )
        
    def add_constraint(self, lhs, sense, rhs):
        """Adds a constraint of the form

        .. math::

            \sum_{j} a_j x_j \circ b
        
        where :math:`\circ \in \{ \leq, =, \geq \}`, :math:`a_j` indicates a coefficient and :math:`x_j` a variable.
        
        :param lhs: Left side of the equation, given as a list of variable/coefficient pairs. Each pair has the coefficient on the
            right and the variable on the left.
        :type lhs: List[Tuple[int,float]]
        :param sense: Type of equation, i.e. "<=", ">=" or "=".
        :type sense: str
        :param rhs: Right side of the equation, i.e. a number.
        :type rhs: float
        :return: index of the added constraint
        :rtype: int
        """        
        assert sense in ["<=", "=", ">="]
        assert rhs == float(rhs), "Right hand side is not a number: rhs=%s" % rhs 
        self._assert_expression(lhs)

        name = "c%d" % len( self.__constraints )

        newconstr = None
        linexpr = self._eval_pulp_expr( lhs )
        if sense == "<=":
            newconstr = self.__model.addConstr(linexpr <= rhs, name)
        elif sense == "=":
            newconstr = self.__model.addConstr(linexpr == rhs, name)
        else:
            newconstr = self.__model.addConstr(linexpr >= rhs, name)
        
        constridx = len(self.__constraints)
        self.__constraints.append(newconstr) 
        return constridx


    def add_to_constraint(self, constridx, coeff, varidx):
        constr = self.__constraints[constridx]
        self.__model.chgCoeff(constr, self.__variables[varidx], coeff)

    def add_indicator_constraint(self, ind_varidx, rhs_varidx):
        """Adds a constraint of the form:
        .. math:: \sigma = 0 \Implies x = 0
        :param ind_varidx: index of the indicator variable
        :type ind_varidx: int
        :param rhs_varidx: index of rhs variable
        :type rhs_varidx: int
        :return: index of the added constraint
        :rtype: int
        """
        name = "c%d" % len( self.__constraints )

        new_constr = self.__model.addConstr((self.__variables[ind_varidx] == 0) << (self.__variables[rhs_varidx] == 0))

        constridx = len(self.__constraints)
        self.__constraints.append(newconstr)
        return constridx

    def remove_constraint(self, constridx):
        """removes a given constraint from the model.

        :param constridx: index of the constraint
        :type constridx: int
        """
        self.__model.remove(self.__constraints[constridx])
        self.__constraints[constridx] = None


    def add_variables_w_bounds(self, var_descr):
        """Adds a list of variables to this MILP. Each element in `var_descr` must be a triple (dom,lb,ub) where dom is either `integer`, `binary` or `real` and lb,ub are floats, or None.

        :return: Index or indices of new variables.
        :rtype: either List[int] or int.
        """
        l = []
        for (domain,lb,ub) in var_descr:
            assert domain in ["integer", "real", "binary"]
            cat = { "binary": GRB.BINARY,
                    "real": GRB.CONTINUOUS,
                    "integer": GRB.INTEGER }[domain]

            varidx = len(self.__variables)
            varname = "x%d" % varidx
            if lb is not None:
                if ub is not None:
                    var = self.__model.addVar(lb=lb, ub=ub, vtype=cat, name=varname)
                else:
                    var = self.__model.addVar(lb=lb, vtype=cat, name=varname)
            elif ub is not None:
                var = self.__model.addVar(ub=ub, vtype=cat, name=varname)
            else:
                var = self.__model.addVar(vtype=cat, name=varname)

            self.__variables.append(var)

            if len(var_descr) == 1:
                return varidx
            else:
                l.append(varidx)
        return l

    def add_variables(self, domains):
        """Adds a list of variables to this MILP. Each element in `domains` must be either `integer`, `binary` or `real`.
        
        :return: Index or indices of new variables.
        :rtype: either List[int] or int.
        """        
        var_descr = [(dom,None,None) for dom in domains]
        return self.add_variables_w_bounds(var_descr)


    @classmethod
    def from_coefficients(cls, A, b, opt, domains, sense="<=", objective="min", bounds=None):
        """Returns a Mixed Integer Linear Programming (MILP) formulation of a problem
        
        .. math::

            \min_x/\max_x\ \sigma^T x \quad \\text{ s.t. } \quad Ax \circ b, \ x_i \in \mathbb{D}_i,\ \\forall i=1,\dots,N

        where :math:`\circ \in \{ \leq, \geq \}`, :math:`N` is the number of variables and :math:`M`
        the number of linear constraints. :math:`\mathbb{D}_i` indicates
        the domain of each variable. If `A`, `b` and `opt` are not given as a `scipy.sparse.dok_matrix`, 
        they are transformed into that form automatically.
        
        :param A: Matrix for inequality conditions  (:math:`A`).
        :type A: :math:`M \\times N`-Matrix
        :param b: Vector for inequality conditions  (:math:`b`).
        :type b: :math:`M \\times 1`-Matrix
        :param opt: Weights for individual variables in x (:math:`\sigma`). If None, no objective function will be set.
        :type opt: :math:`N \\times 1`-Matrix
        :param domains: Array of strings, e.g. ["real", "integer", "integer", "binary", ...] which indicates the domain for each variable.
        :type domains: List[str]
        :param sense: "<=" or ">=", defaults to "<="
        :type sense: str, optional
        :param objective: "min" or "max", defaults to "min"
        :type objective: str, optional
        :param bounds: a vector of lower/upper bounds for all variables, optional
        :type bounds: [(float,float)], optional
        :return: The resulting MILP.
        :rtype: solver.MILP
        """

        A = cast_dok_matrix(A).tocsr()
        b = cast_dok_matrix(b)

        opt = cast_dok_matrix(opt)

        model = GurobiMILP(objective=objective)

        # initialize problem
        # this adds the variables and the objective function (which is opt^T*x, i.e. sum_{i=1}^N opt[i]*x[i])
        if bounds is not None:
            model.add_variables_w_bounds([(domains[idx], bounds[idx][0], bounds[idx][1]) for idx in range(A.shape[1])])
        else:
            model.add_variables([domains[idx] for idx in range(A.shape[1])])

        model.set_objective_function([(idx, opt[idx,0]) for idx in range(A.shape[1])])

        # this takes quite a lot of time since accessing the rows is inefficient, even for csr-formats.
        # maybe find a way to compute Ax <= b faster.
        # now: add linear constraints: Ax <= b.

        for constridx in range(A.shape[0]):
            # calculates A[constridx,:]^T * x
            row = A.getrow(constridx)
            lhs = [0]*len(row.indices)
            # print(row)
            for i,j,d in zip(range(len(row.indices)), row.indices, row.data):
                lhs[i] = (j, float(d))
            # adds constraint: A[constridx,:]^T * x <= b[constridx]
            model.add_constraint(lhs, sense, b[constridx,0])

        return model

    def __repr__(self):
        return str(self.__model)
