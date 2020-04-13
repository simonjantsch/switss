from farkas.utils import array_to_dok_matrix

from bidict import bidict
import numpy as np
from scipy.sparse import dok_matrix,hstack,vstack

class ReachabilityForm:
    """ A reachability form is an MDP with a dedicated initial and target state. 
    It is represented by its transition matrix, an initial state and a vector that holds 
    the probability to move to the target state in one step for each state. 
    The target state is not included in the transition matrix. """
    
    def __init__(self, P, initial, to_target, index_by_state_action):
        self.P = P
        self.initial = initial
        self.to_target = to_target
        self.index_by_state_action = index_by_state_action

    def __repr__(self):
        return "ReachabilityForm(C=%s, N=%s, initial=%s)" % (self.P.shape[0], self.P.shape[1], self.initial)

    def fark_min_constraints(self, threshold):
        """ Returns the constraints of the Farkas-min polytope:

        .. math::

            (I-P) \, \mathbf{z} \leq \mathbf{b} \land \mathbf{z}(\\texttt{init}) \leq \lambda

        where :math:`\mathbf{b}`` is the vector "to_target", :math:`\lambda` is the threshold and :math:`I` is the matrix that for every row (state,action) has a 1 in the column "state".

        :param threshold: The threshold :math:`\lambda` for which the Farkas-min polytope should be constructed
        :type threshold: Float
        :return: :math:`C+1 \\times N`-matrix :math:`M`, and :math:`N`-vector :math:`rhs` such that :math:`M \mathbf{z} \leq rhs` yields the Farkas-min polytope
        """
        C,N = self.P.shape
        I = self._reach_form_id_matrix()

        rhs = self.to_target.copy()
        rhs.resize(C+1)
        rhs[C] = -threshold

        delta = np.zeros(N)
        delta[self.initial] = 1

        # TODO: check whether the casting to dok_matrix is necessary
        fark_min_matr = dok_matrix(vstack(((I-self.P),-delta)))
        return fark_min_matr, rhs

    def fark_max_constraints(self, threshold):
        """ Returns the constraints of the Farkas-max polytope:

        .. math::

            \mathbf{y} \, (I-P) \leq \delta_{\\texttt{init}} \land \mathbf{b} \, \mathbf{y} \leq \lambda

        where :math:`\mathbf{b}`` is the vector "to_target", :math:`\lambda` is the threshold and :math:`I` is the matrix that for every row (state,action) has a 1 in the column "state". The vector :math:`\delta_{\\texttt{init}}` is 1 for the initial state, and otherwise 0.

        :param threshold: The threshold :math:`\lambda` for which the Farkas-max polytope should be constructed
        :type threshold: Float
        :return: :math:`N+1 \\times C`-matrix :math:`M`, and :math:`C`-vector :math:`rhs` such that :math:`M \mathbf{y} \leq rhs` yields the Farkas-max polytope
        """
        C,N = self.P.shape
        I = self._reach_form_id_matrix()

        b = array_to_dok_matrix(self.to_target)

        rhs = np.zeros(N+1)
        rhs[self.initial] = 1
        rhs[N] = -threshold

        # TODO: check whether the casting to dok_matrix is necessary
        fark_max_matr = dok_matrix(hstack(((I-self.P),-b)).T)

        return fark_max_matr, rhs

    def _reach_form_id_matrix(self):
        """Computes the matrix :math:`I` for a given reachability form that for every row (st,act) has an entry 1 at the column corresponding to st."""
        C,N = self.P.shape
        I = dok_matrix((C,N))

        for i in range(0,C):
            (state, action) = self.index_by_state_action.inv[i]
            I[i,state] = 1

        return I

def induced_subsystem(reach_form, state_vector):
    """ Given a reachability form with :math:`N` states and a vector in :math:`\{0,1\}^N` computes the induced subsystem of that vector.

    :param reach_form: A reachability form
    :type reach_form: model.ReachabilityForm
    :param state_vector: a vector indicating which states to keep
    :type state_vector: :math:`N`-vector over :math:`\{0,1\}`
    :return: The induced subsystem as a rechability form, and a bidirectional mapping from states in the subsystem to states in the original system.
    :rtype: Tuple[model.ReachabilityForm, bidict]
    """
    C,N = reach_form.P.shape

    assert state_vector.size == N
    assert state_vector[reach_form.initial] == 1

    new_to_old_states = bidict()
    new_index_by_state_action = bidict()
    new_N = 0
    new_C = 0

    # Map the new states to new indices and compute a map to the old states
    for i in range(0,N):
        assert state_vector[i] in [0,1]
        if state_vector[i] == 1:
            new_to_old_states[new_N] = i
            new_N += 1

    # Compute the new number of choices (= rows)
    for rowidx in range(0,C):
        (source,action) = reach_form.index_by_state_action.inv[rowidx]
        if state_vector[source] == 1:
            new_source = new_to_old_states.inv[source]
            new_index_by_state_action[(new_source,action)] = new_C
            new_C += 1

    new_P = dok_matrix((new_C,new_N))

    # Populate the new transition matrix
    for (rowidx,target) in reach_form.P.keys():
        (source,action) = reach_form.index_by_state_action.inv[rowidx]
        if state_vector[source] == 1 and state_vector[target] == 1:
            new_source = new_to_old_states.inv[source]
            new_target = new_to_old_states.inv[target]
            new_row_idx = new_index_by_state_action[(new_source,action)]
            new_P[new_row_idx,new_target] = reach_form.P[rowidx,target]


    new_to_target = np.zeros(new_N)

    for new_state in range(0,new_N):
        old_to_target = reach_form.to_target
        new_to_target[new_state] = old_to_target[new_to_old_states[new_state]]

    new_initial = new_to_old_states.inv[reach_form.initial]

    subsys_reach_form = ReachabilityForm(new_P,new_initial,new_to_target,new_index_by_state_action)

    return subsys_reach_form,new_to_old_states
