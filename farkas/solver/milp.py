from ..utils import array_to_dok_matrix
from scipy.sparse import dok_matrix
import pulp

class MILP:
    def __init__(self, A, b, opt, integer_constraints, lowBound = None, upBound = None):
        """Returns a Mixed Integer Linear Programming (MILP) formulation of a problem
        
        .. math::

            \min_{x} \sigma^T x \quad \\text{ s.t. } \quad Ax \leq b, \ x_i \in \mathbb{Z},\ \\forall i\in I

        where :math:`N` is the number of variables and :math:`M` the number of linear constraints.
        If `A`, `b` and `opt` are not given as a `scipy.sparse.dok_matrix`, they are transformed into that form automatically.
        The optional `lowBound` and `upBound` arguments are global lower/upper bounds on each variable.

        :param A: Matrix for inequality conditions  (:math:`A`).
        :type A: :math:`M \\times N`-Matrix
        :param b: Vector for inequality conditions  (:math:`b`).
        :type b: :math:`M \\times 1`-Matrix
        :param opt: Weights for individual variables in x (:math:`\sigma`).
        :type opt: :math:`N \\times 1`-Matrix
        :param integer_constraints: Set of indices :math:`I \subseteq \{1,\dots,N-1\}` which 
            indicates integer constraints on variables.
        :type integer_constraints: Set[int]
        :param lowBound: global lower bound for all variables
        :type lowBound: Float
        :param upBound: global upper bound for all variables
        :type upBound: Float
        """
        self.A = array_to_dok_matrix(A) if not isinstance(A,dok_matrix) else A
        self.b = array_to_dok_matrix(b) if not isinstance(b,dok_matrix) else b
        self.opt = array_to_dok_matrix(opt) if not isinstance(opt,dok_matrix) else opt
        self.lowBound = lowBound
        self.upBound = upBound
        self.integer_constraints = integer_constraints

    @property
    def shape(self):
        """Returns the shape of this problem instance. 
        
        :return: :math:`(M,N)`
        :rtype: Tuple[int,int]
        """        
        return self.A.shape

    def pulpmodel(self):
        """Constructs a new pulp-model (`pulp.LpProblem`) from this instance.
        
        :return: A tuple containing the model and variables. 
        :rtype: Tuple[pulp.LpProblem, List[pulp.LpVariable]]
        """        
        # initialize problem
        model = pulp.LpProblem("", pulp.LpMinimize)
        # initialize variables w/ respective category: integer or continous.
        # this also initializes all integer constraints on the variables.
        objf, variables = [], []
        for varidx in range(self.shape[1]):
            vartype = pulp.LpInteger if varidx in self.integer_constraints else pulp.LpContinuous
            variables.append(pulp.LpVariable("x%d" % varidx, lowBound=self.lowBound, upBound=self.upBound, cat=vartype))
            objf.append((variables[varidx], self.opt[varidx,0]))
        # this adds the objective function (which is opt^T*x, i.e. sum_{i=1}^N opt[i]*x[i])
        model += pulp.LpAffineExpression(objf)
        
        # now: add linear constraints: Ax <= b.
        for constridx in range(self.shape[0]):
            # calculates A[constridx,:]^T * x
            lhs = []
            row = self.A[constridx,:]
            for (_,j), a in row.items():
                lhs.append((variables[j], a))
            lhs = pulp.LpAffineExpression(lhs)
            # adds constraint: A[constridx,:]^T * x <= b[constridx]
            model += pulp.LpConstraint(name="c%d"%constridx, e=lhs, sense=pulp.LpConstraintLE, rhs=self.b[constridx,0])

        return model, variables


    def __str__(self):
        return str(self.pulpmodel())


class LP(MILP):
    def __init__(self, A, b, opt,lowBound = None, upBound = None):
        """Returns a Linear Programming (LP) formulation of a problem

        .. math::
        
            \min_x \sigma^T x \quad \\text{s.t.}\quad Ax \leq b
        
        where :math:`N` is the number of variables and :math:`M` the number of linear constraints.
        If `A`, `b` and `opt` are not given as a `scipy.sparse.dok_matrix`, they are transformed into that form automatically.

        :param A: Matrix for inequality conditions  (:math:`A`).
        :type A: :math:`M \\times N`-Matrix
        :param b: Vector for inequality conditions  (:math:`b`).
        :type b: :math:`M \\times 1`-Matrix
        :param opt: Weights for individual variables in x (:math:`\sigma`).
        :type opt: :math:`N \\times 1`-Matrix
        :param lowBound: global lower bound for all variables
        :type lowBound: Float
        :param upBound: global upper bound for all variables
        :type upBound: Float
        """
        super().__init__(A,b,opt,[],lowBound,upBound)
