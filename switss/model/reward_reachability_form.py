from . import AbstractMDP,MDP,ReachabilityForm

class RewardReachabilityForm:


    def __init__(self, reachability_form, reward_vector):

        self.target_label = reachability_form.target_label
        self.__reach_form = reachability_form
        self.reward_vector = reward_vector

    @classmethod
    def fromsystem(cls, system, initial_label, target_label="rrf_target",ignore_consistency_checks=False):
        if not ignore_consistency_checks:
            RewardReachabilityForm.assert_consistency(system, initial_label, target_label)

        # add a dummy fail state, which is required for reachability form
        N = system.P.N
        C = system.P.C

        system.P.resize(C + 1, N + 1)
        system.P[C + 1, N + 1] = 1

        system.add_label(N, "rf_fail")

        # initialize underlying reachability form
        return cls(ReachabilityForm(system, initial_label, target_label,"rf_fail",ignore_consistency_checks), system.reward_vector)


    @staticmethod
    def reduce(system, initial_label, target_label, new_target_label="rrf_target", debug=False):

        reach_form, to_rf_cols, to_rf_rows = ReachabilityForm.reduce(system, initial_label, target_label)
        return RewardReachabilityForm.fromsystem(reach_form), to_rf_cols, to_rf_rows

    @staticmethod
    def assert_consistency(system, initial_label, target_label="rf_target"):
        """Checks whether a system fulfills the reward reachability form properties.

        :param system: The system
        :type system: model.AbstractMDP
        :param initial_label: Label of initial state
        :type initial_label: str
        :param target_label: Label of target state, defaults to "rf_target"
        :type target_label: str, optional
        """
        assert isinstance(system, AbstractMDP)
        assert (system.reward_vector != None)
        assert len({initial_label, target_label}) == 2, "Initial label, target label and fail label must be pairwise distinct"

        # check that there is exactly 1 target and initial state resp.
        states = []
        for statelabel, name in [(initial_label, "initial"), (target_label, "target")]:
            labeledstates = system.states_by_label[statelabel].copy()
            count = len(labeledstates)  
            assert count == 1, \
                "There must be exactly 1 %s state. There are %s states in system %s with label %s" % (name, count, system, statelabel)
            states.append(labeledstates.pop())
        init,target = states
        
        # check that fail and target state only map to themselves
        for state,name,rowidx,colidx in [(target,"target",system.C-1,system.N-1)]:
            successors = list(system.successors(state))
            assert len(successors) == 1 and successors[0][0] == state, "State %s must only one action and successor; itself" % name
            succst,succact,p = successors[0]
            saindex = system.index_by_state_action[(succst,succact)]
            assert saindex == rowidx, "State-action of %s must be at index %s but is at %s" % (name, rowidx, saindex)
            assert state == colidx, "State %s must be at index %s but is at %s" % (name, colidx, state)

        # check that every state is reachable
        # the fail state doesn't need to be reachable
        fwd_mask = system.reachable_mask({init},"forward")
        assert (fwd_mask).all(), "Not every state is reachable from %s in system %s" % (initial_label, system)

        # check that every state except fail reaches goal
        bwd_mask = system.reachable_mask({target},"backward")
        assert (bwd_mask).all(), "Not every state reaches %s in system %s" % (target_label, system)

        # check that there is no proper end component apart from the target state
        # target_mask has a 1 only at the fail state and zeros otherwise
        target_mask = np.zeros(system.N,dtype=np.bool)
        target_mask[target] = True
        mec_mask = system.maximal_end_components()
        assert (target_mask = mec_mask), "there is some proper end component apart from the target state"

        def fark_constraints(self, threshold, mode):
        """returns the right constraint set dependent on the given mode.

        :param threshold: the threshold
        :type threshold: float
        :param mode: either 'min' or 'max'
        :type mode: str
        :param rewards: if true, returns Farkas constraints for reward thresholds. Defaults to False
        :type rewards: Optional bool
        :return: either :math:`(C+1) \\times N`-matrix :math:`M_z`, and vector of length :math:`C+1` :math:`rhs_z` or :math:`(N+1) \\times C`-matrix :math:`M_y`, and :math:`N+1`-vector :math:`rhs_y`.
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """ 
        assert mode in ["min", "max"]

        if mode == "min":
            return self.fark_z_constraints(threshold)
        else:
            return self.fark_y_constraints(threshold)

    def fark_z_constraints(self, threshold):
        """
        Returns a matrix :math:`M_z` and a vector :math:`rhs_z` such that for a :math:`N` vector :math:`\mathbf{z}`

        .. math::

            M_z\, \mathbf{z} \leq rhs_z \quad \\text{  iff  } \quad 
            \\mathbf{A} \, \mathbf{z} \leq \mathbf{b} \land \mathbf{z}(\\texttt{init}) \geq \lambda
            \quad \\text{  iff  } \quad
            \mathbf{z} \in \mathcal{P}^{\\text{min}}(\lambda)
                
        :param threshold: The threshold :math:`\lambda` for which the Farkas z-constraints should be constructed
        :type threshold: Float
        :return: :math:`(C+1) \\times N`-matrix :math:`M_z`, and vector of length :math:`C+1` :math:`rhs_z`
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """
        C,N = self.__P.shape

        rhs = self.reward_vector.A1.copy()
        rhs.resize(C+1)
        rhs[C] = -threshold

        delta = np.zeros(N)
        delta[self.initial] = 1

        fark_z_matr = vstack((self.A,-delta))
        return fark_z_matr, rhs

    def fark_y_constraints(self, threshold):
        """ 
        Returns a matrix :math:`M_y` and a vector :math:`rhs_y` such that for a :math:`C` vector :math:`\mathbf{y}`

        .. math::

            M_y\, \mathbf{y} \leq rhs_y \quad \\text{  iff  } \quad
            \mathbf{y} \, \mathbf{A} \leq \delta_{\\texttt{init}} \land \mathbf{b} \, \mathbf{y} \geq \lambda
            \quad \\text{  iff  } \quad
            \mathbf{y} \in \mathcal{P}^{\\text{max}}(\lambda)

        where :math:`\lambda` is the threshold, :math:'\mathbf{A}' is the system matrix and :math:`\mathbf{b}` is the reward vector. The vector :math:`\delta_{\\texttt{init}}` is 1 for the initial state, and otherwise 0.

        :param threshold: The threshold :math:`\lambda` for which the Farkas y-constraints should be constructed
        :type threshold: Float
        :return: :math:`(N+1) \\times C`-matrix :math:`M_y`, and :math:`N+1`-vector :math:`rhs_y` 
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """
        C,N = self.__P.shape

        b = cast_dok_matrix(self.reward_vector)

        rhs = np.zeros(N+1)
        rhs[self.initial] = 1
        rhs[N] = -threshold

        fark_y_matr = hstack((self.A,-b)).T
        return fark_y_matr, rhs

