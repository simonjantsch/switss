from . import AbstractMDP,MDP
from ..utils import InvertibleDict, cast_dok_matrix, DTMCVisualizationConfig, VisualizationConfig
from ..solver.milp import LP

from collections import defaultdict
from bidict import bidict
import copy as copy
import numpy as np
from scipy.sparse import dok_matrix,hstack,vstack

class ReachabilityForm:
    """ 
    A reachability form is a wrapper for special DTMCs/MDPs with dedicated initial and target states. 
    In particular, the following properties are fulfilled:

    - exactly one fail, goal and initial state,
    - fail and goal have exactly one action, which maps only to themselves,
    - the fail state (goal state) has index :math:`N_{S_{\\text{all}}}-1` (:math:`N_{S_{\\text{all}}}-2`) and the corresponding state-action-pair index :math:`C_{S_{\\text{all}}}-1` (:math:`C_{S_{\\text{all}}}-2`),
    - every state is reachable from the initial state (fail doesn't need to be reachable) and
    - every state reaches the goal state (except the fail state)

    """
    def __init__(self, system, initial_label, target_label="rf_target", fail_label="rf_fail", ignore_consistency_checks=False):
        """Instantiates a RF.

        :param system: The MDP/DTMC that fulfills the specified properties.
        :type system: model.AbstractMDP
        :param initial_label: Label of initial state - there must be exactly one
        :type initial_label: str
        :param target_label: Label of target state - there must be exactly one, defaults to "rf_target"
        :type target_label: str, optional
        :param fail_label: Label of fail state - there must be exactly one, defaults to "rf_fail"
        :type fail_label: str, optional
        :param ignore_consistency_checks: If set to False, checks consistency of given model (i.e. if the properties are satisfied),
            defaults to False
        :type ignore_consistency_checks: bool, optional
        :type ignore_consistency_checks: bool, optional
        """        
        if not ignore_consistency_checks:
            ReachabilityForm.assert_consistency(system, initial_label, target_label, fail_label)
        
        self.__P = system.P[:system.C-2, :system.N-2]
        self.__system = system
        self.initial = next(iter(system.states_by_label[initial_label]))
        self.target_label = target_label
        self.fail_label = fail_label
        self.initial_label = initial_label
        self.__index_by_state_action = system.index_by_state_action.copy()
        del self.__index_by_state_action.inv[system.C-2]
        del self.__index_by_state_action.inv[system.C-1]

        self.__A = self._reach_form_id_matrix() - self.__P
        self.__to_target = system.P.getcol(system.N-2).todense()[:system.C-2]

        system_mecs, proper_mecs, nr_of_system_mecs = system.maximal_end_components()
        self.__state_to_mec = system_mecs[:system.N-2]
        self.__nr_of_mecs = int(nr_of_system_mecs - 2)
        self.__proper_mecs = proper_mecs

        self.__target_visualization_style = None
        self.__fail_visualization_style = None
        self.set_target_visualization_style()
        self.set_fail_visualization_style()

        self.__fark_y_matr = None
        self.__fark_y_rhs = None
        self.__fark_z_matr = None
        self.__fark_z_rhs = None

    @property
    def target_sap_idx(self):
        return self.system.C-2

    @property
    def target_state_idx(self):
        return self.system.N-2

    @property
    def fail_sap_idx(self):
        return self.system.C-1

    @property
    def fail_state_idx(self):
        return self.system.N-1

    def set_target_visualization_style(self,style=None):
        assert style is None or type(style) == type(self.system.visualization)

        def _state_map(sourceidx,labels):
            return { "color" : "green", "style": "filled", "label" : self.target_label }
        def _action_map(sourceidx,action,labels):
            return { "node" : { "shape" : "point" }, "edge" : { "dir" : "none" } }
        
        if style is None:
            if type(self.system) is MDP: style = VisualizationConfig(state_map=_state_map,action_map=_action_map)
            else: style = DTMCVisualizationConfig(state_map=_state_map)
        
        self.__target_visualization_style = style

    def set_fail_visualization_style(self,style=None):
        assert style is None or type(style) == type(self.system.visualization)
        
        def _state_map(sourceidx,labels):
            return { "color" : "red", "style": "filled", "label" : self.fail_label }
        def _action_map(sourceidx,action,labels):
            return { "node" : { "shape" : "point" }, "edge" : { "dir" : "none" } }

        if style is None:
            if type(self.system) is MDP: style = VisualizationConfig(state_map=_state_map,action_map=_action_map)
            else: style = DTMCVisualizationConfig(state_map=_state_map)
        
        self.__fail_visualization_style = style

    @property
    def system(self):
        """The underlying system (instance of model.AbstractMDP)"""
        return self.__system

    @property
    def A(self):
        """
        Returns a :math:`C \\times N` matrix :math:`\mathbf{A}` where 

        .. math::
        
            \\textbf{A}((s,a), d) = \\begin{cases} 1 - \\textbf{P}((s,a), d) &\\text{ if } d = s \\\\
            - \\textbf{P}((s,a), d) &\\text{ if } d \\neq s \end{cases}, 

        for all :math:`(s,a),d \in \mathcal{M} \\times S`."""
        return self.__A

    @property
    def P(self):
        """
        Returns the :math:`C \\times N` transition probability matrix :math:`\mathbf{P}`
        """
        return self.__P

    @property
    def index_by_state_action(self):
        """
        Returns a bidict which maps state-action pairs to indices (i.e. rows) of the matrix P, and vice versa
        """
        return self.__index_by_state_action

    @property
    def to_target(self):
        """
        Returns a vector of length :math:`C` :math:`\\textbf{b}` where

        .. math::

            \\textbf{b}((s,a)) = \\text{P}((s,a),\\text{goal}),

        for all :math:`(s,a) \in \mathcal{M}`. 
        """
        return self.__to_target

    @property
    def proper_mecs(self):
        """
        Returns a vector which contains a boolean value for each maximal end component and indicates whether it is proper or not.
        """
        return self.__proper_mecs

    @property
    def nr_of_mecs(self):
        """
        Returns the number of end components (excluding goal and fail).
        """
        return self.__nr_of_mecs

    @property
    def nr_of_proper_mecs(self):
        """
        Returns the number of proper end components (excluding goal and fail).
        """
        return sum(self.__proper_mecs) - 2

    @property
    def is_ec_free(self):
        """
        Returns yes if the RF is EC-free (its only proper end components are induced by goal and fail).
        """
        return self.nr_of_proper_mecs == 0

    @property
    def state_to_mec(self):
        """
        Returns a vector of length :math:`N` which contains the index of the corresponding MEC for each state.
        """
        return self.__state_to_mec

    def in_proper_ec(self, state):
        """
        Returns yes if "state" is included in a proper end component.
        """
        return self.proper_mecs[self.__state_to_mec[state]]

    @staticmethod
    def assert_consistency(system, initial_label, target_label="rf_target", fail_label="rf_fail"):
        """Checks whether a system fulfills the reachability form properties.

        :param system: The system
        :type system: model.AbstractMDP
        :param initial_label: Label of initial state
        :type initial_label: str
        :param target_label: Label of target state, defaults to "rf_target"
        :type target_label: str, optional
        :param fail_label: Label of fail state, defaults to "rf_fail"
        :type fail_label: str, optional
        """        
        assert isinstance(system, AbstractMDP)
        assert len({initial_label, target_label, fail_label}) == 3, "Initial label, target label and fail label must be pairwise distinct"

        # check that there is exactly 1 target,fail and initial state resp.
        states = []
        for statelabel, name in [(initial_label, "initial"), (target_label, "target"), (fail_label, "fail")]:
            labeledstates = system.states_by_label[statelabel].copy()
            count = len(labeledstates)  
            assert count == 1, \
                "There must be exactly 1 %s state. There are %s states in system %s with label %s" % (name, count, system, statelabel)
            states.append(labeledstates.pop())
        init,target,fail = states
        
        # check that fail and target state only map to themselves
        for state,name,rowidx,colidx in [(target,"target",system.C-2,system.N-2),(fail,"fail",system.C-1,system.N-1)]:
            successors = list(system.successors(state))
            assert len(successors) == 1 and successors[0][0] == state, "State %s must only one action and successor; itself" % name
            succst,succact,p = successors[0]
            saindex = system.index_by_state_action[(succst,succact)]
            assert saindex == rowidx, "State-action of %s must be at index %s but is at %s" % (name, rowidx, saindex)
            assert state == colidx, "State %s must be at index %s but is at %s" % (name, colidx, state)

        # fail_mask has a 1 only at the fail state and zeros otherwise
        fail_mask = np.zeros(system.N,dtype=bool)
        fail_mask[fail] = True 

        # check that every state is reachable
        # the fail state doesn't need to be reachable
        fwd_mask = system.reachable_mask({init},"forward")
        assert (fwd_mask | fail_mask).all(), "Not every state is reachable from %s in system %s" % (initial_label, system)

        # check that every state except fail reaches goal
        # if bwd_mask[fail] == 1 then bwd_mask[fail] ^ fail_mask[fail] == 0
        bwd_mask = system.reachable_mask({target},"backward")
        assert (bwd_mask ^ fail_mask).all(), "Not every state reaches %s in system %s" % (target_label, system)

    @staticmethod
    def reduce(system, initial_label, target_label, new_target_label="rf_target", new_fail_label="rf_fail", debug=False):
        """Reduces a system to a system in reachability form. 
        The transformation does a forward search starting at the initial state, then a 
        backwards search starting from the targets states and then removes all states 
        that happen to be not reachable in at least one of these searches. Transitions that 
        lead to removed states are mapped to a dedicated new "fail"-state (default label is "rf_fail"). 
        All old target states are remapped to a dedicated new "target"-state (default label is "rf_target"). 
        
        :param system: The system that should be reduced.
        :type system: model.AbstractMDP
        :param initial_label: Label of initial state (there must be exactly one)
        :type initial_label: str
        :param target_label: Label of target state (there must be at least one)
        :type target_label: str
        :param new_target_label: Label of dedicated new target state, defaults to "rf_target"
        :type new_target_label: str, optional
        :param new_fail_label: Label of dedicated new fail state, defaults to "rf_fail"
        :type new_fail_label: str, optional
        :param debug: If True, additional diagnostic information is printed, defaults to False
        :type debug: bool, optional
        :return: A triple (RF, state_map, state_action_map) where state_map (state_action_map) is a mapping from system states
            (state-actions pairs) to states (state-action pairs) of the reduced system. If a state (state-action pair) is not a 
            key in the dictionary, it was removed.
        :rtype: Tuple[model.ReachabilityForm, Dict[int,int], Dict[Tuple[int,int],Tuple[int,int]]]
        """        
        assert isinstance(system, AbstractMDP)
        assert new_target_label not in system.states_by_label.keys(), "Label '%s' for target state already exists in system %s" % (new_target_label, system)
        assert new_fail_label not in system.states_by_label.keys(), "Label '%s' for fail state already exists in system %s" % (new_fail_label, system)
        target_states = system.states_by_label[target_label]
        assert len(target_states) > 0, "There needs to be at least one target state."
        initial_state_count = len(system.states_by_label[initial_label])
        assert initial_state_count == 1, "There are %d states with label '%s'. Must be 1." % (initial_state_count, initial_label)
        initial = list(system.states_by_label[initial_label])[0]

        if debug:
            print("calculating reachable mask (backward)...")
        backward_reachable = system.reachable_mask(target_states, "backward")
        if debug:
            print("calculating reachable mask (forward)...")
        forward_reachable = system.reachable_mask(set([initial]), "forward", blocklist=target_states)
        # states which are reachable from the initial state AND are able to reach target states
        reachable_mask = backward_reachable & forward_reachable

        if debug:
            print("tested backward & forward reachability test")

        # reduce states + new target and new fail state 
        new_state_count = np.count_nonzero(reachable_mask) + 2
        target_idx, fail_idx = new_state_count - 2, new_state_count - 1

        if debug:
            print("new states: %s, target index: %s, fail index: %s" % (new_state_count, target_idx, fail_idx))

        # create a mapping from system to reachability form
        to_rf_cols, to_rf_rows = bidict(), bidict()

        reachable_index_mapping = dict()
        nextidx = 0
        for sapidx in range(system.C):
            stateidx, actionidx = system.index_by_state_action.inv[sapidx]
            if reachable_mask[stateidx]:
                if stateidx not in reachable_index_mapping:
                    reachable_index_mapping[stateidx] = nextidx
                    nextidx += 1
                idx = reachable_index_mapping[stateidx]
                to_rf_rows[(stateidx,actionidx)] = (idx,actionidx)
                to_rf_cols[stateidx] = idx

        if debug:
            print("computed state-action mapping")

        # compute reduced transition matrix (without non-reachable states)
        # compute probability of reaching the target state in one step 
        new_N = len(set(to_rf_cols.values()))
        new_C = len(set(to_rf_rows.values()))
        new_P = dok_matrix((new_C, new_N))
        new_index_by_state_action = bidict()
        
        if debug:
            print("shape of new_P (%s,%s)" % (new_C,new_N))

        to_target = np.zeros(new_C)

        # mask for faster access
        target_mask = np.zeros(system.N, dtype=bool)
        for t in target_states:
            target_mask[t] = 1

        for newidx, ((source,action),(newsource,newaction)) in enumerate(to_rf_rows.items()):
            new_index_by_state_action[(newsource,newaction)] = newidx
            if target_mask[source]: # in target_states:
                to_target[newidx] = 1
            else:
                idx = system.index_by_state_action[(source,action)]
                for dest in [s for s,a,p in system.successors(source) if a == action]:
                    if dest in to_rf_cols:
                        newdest = to_rf_cols[dest]
                        new_P[newidx, newdest] = system.P[idx, dest]
        if debug:
            print("computed transition matrix & to_target")

        rf_system = ReachabilityForm.__initialize_system(
            new_P, 
            new_index_by_state_action,
            to_target, 
            to_rf_rows, 
            system, 
            new_target_label, 
            new_fail_label)

        rf = ReachabilityForm(
            rf_system, 
            initial_label, 
            target_label=new_target_label, 
            fail_label=new_fail_label,
            ignore_consistency_checks=True)

        rf.__adapt_style(to_rf_cols.inv, to_rf_rows.inv, system.visualization)

        return rf, to_rf_cols, to_rf_rows

    @staticmethod
    def __initialize_system(P, index_by_state_action, to_target, mapping, configuration, target_label, fail_label):
        C,N = P.shape
        P_compl = dok_matrix((C+2, N+2))
        target_state, fail_state = N, N+1

        index_by_state_action_compl = index_by_state_action.copy()
        index_by_state_action_compl[(target_state,0)] = C
        index_by_state_action_compl[(fail_state,0)] = C+1

        if configuration.reward_vector is not None:
            reward_vector = np.zeros(C+2)
            reward_vector[C] = 0
            reward_vector[C+1] = 0
        else:
            reward_vector = None

        
        # copy labels from configuration (i.e. a system)
        # mapping defines which state-action pairs in the system map to which state-action pairs in the r.f.
        label_to_actions = defaultdict(set)
        label_to_states = defaultdict(set)
        for idx in range(C):
            stateidx, actionidx = index_by_state_action.inv[idx]
            sys_stateidx, sys_actionidx = mapping.inv[(stateidx,actionidx)]
            labels = configuration.labels_by_state[sys_stateidx]
            for l in labels:
                label_to_states[l].add(stateidx)
            actionlabels = configuration.labels_by_action[(sys_stateidx,sys_actionidx)]
            for l in actionlabels:
                label_to_actions[l].add((stateidx,actionidx))

            if configuration.reward_vector is not None:
                # initialize new reward vector
                sys_idx = configuration.index_by_state_action[(sys_stateidx,sys_actionidx)]
                reward_vector[idx] = configuration.reward_vector[sys_idx]

        label_to_states[fail_label].add(fail_state)
        label_to_states[target_label].add(target_state)

        not_to_fail = np.zeros(C)
        for (idx, dest), p in P.items():
            not_to_fail[idx] += p
            P_compl[idx, dest] = p

        for idx, p_target in enumerate(to_target):
            if p_target > 0:
                P_compl[idx, target_state] = p_target
            p_fail = 1 - (p_target + not_to_fail[idx])
            if p_fail > 0:
                P_compl[idx, fail_state] = p_fail

        P_compl[C, target_state] = 1
        P_compl[C+1, fail_state] = 1

        return type(configuration)( 
                    P=P_compl, 
                    index_by_state_action=index_by_state_action_compl, 
                    label_to_states=label_to_states,
                    label_to_actions=label_to_actions,
                    reward_vector=reward_vector)
    
    def __adapt_style(self, state_map, state_action_map, viz_cfg):
        C,N = self.system.C,self.system.N

        def _state_style(sourceidx, labels):
            if sourceidx == N-2: # target state
                return self.__target_visualization_style.state_map(sourceidx,labels)
            elif sourceidx == N-1: 
                return self.__fail_visualization_style.state_map(sourceidx,labels)
            else:
                return viz_cfg.state_map(state_map[sourceidx],labels)
        
        def _action_style(sourceidx,action,labels):
            if sourceidx == N-2:
                return self.__target_visualization_style.action_map(sourceidx,action,labels)
            elif sourceidx == N-1:
                return self.__fail_visualization_style.action_map(sourceidx,action,labels)
            else:
                _sourceidx,_action = state_action_map[(sourceidx,action)]
                return viz_cfg.action_map(_sourceidx,_action,labels)
        
        def _trans_style_dtmc(sourceidx,destidx,p):
            if destidx == N-2:
                return self.__target_visualization_style.trans_map(sourceidx,destidx,p)
            elif destidx == N-1:
                return self.__fail_visualization_style.trans_map(sourceidx,destidx,p)
            else:
                return viz_cfg.trans_map(state_map[sourceidx],state_map[destidx],p)
        
        def _trans_style_mdp(sourceidx,action,destidx,p):
            if destidx == N-2:
                return self.__target_visualization_style.trans_map(sourceidx,action,destidx,p)
            elif destidx == N-1:
                return self.__fail_visualization_style.trans_map(sourceidx,action,destidx,p)
            else:
                _sourceidx,_action = state_action_map[(sourceidx,action)]
                return viz_cfg.trans_map(_sourceidx,_action,state_map[destidx],p)
        
        if type(viz_cfg) == DTMCVisualizationConfig:
            self.system.visualization = DTMCVisualizationConfig(state_map=_state_style,trans_map=_trans_style_dtmc)
        else:
            self.system.visualization = VisualizationConfig(state_map=_state_style,trans_map=_trans_style_mdp,action_map=_action_style)

    def __repr__(self):
        return "ReachabilityForm(initial=%s, target=%s, fail=%s, system=%s)" % (self.initial_label, self.target_label, self.fail_label, self.system)

    def fark_constraints(self, threshold, mode):
        """returns the right constraint set dependent on the given mode.

        :param threshold: the threshold
        :type threshold: float
        :param mode: either 'min' or 'max'
        :type mode: str
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

        if (self.__fark_z_matr is not None) and (self.__fark_z_rhs is not None):
            self.__fark_z_rhs[C] = -threshold
            return self.__fark_z_matr, self.__fark_z_rhs

        self.__fark_z_rhs = self.to_target.A1.copy()
        self.__fark_z_rhs.resize(C+1)
        self.__fark_z_rhs[C] = -threshold

        delta = np.zeros(N)
        delta[self.initial] = 1

        self.__fark_z_matr = vstack((self.A,-delta))

        return self.__fark_z_matr, self.__fark_z_rhs

    def fark_y_constraints(self, threshold):
        """ 
        Returns a matrix :math:`M_y` and a vector :math:`rhs_y` such that for a :math:`C` vector :math:`\mathbf{y}`

p        .. math::

            M_y\, \mathbf{y} \leq rhs_y \quad \\text{  iff  } \quad
            \mathbf{y} \, \mathbf{A} \leq \delta_{\\texttt{init}} \land \mathbf{b} \, \mathbf{y} \geq \lambda
            \quad \\text{  iff  } \quad
            \mathbf{y} \in \mathcal{P}^{\\text{max}}(\lambda)

        where :math:`\lambda` is the threshold, :math:'\mathbf{A}' is the system matrix and :math:`\mathbf{b}` is to_target. The vector :math:`\delta_{\\texttt{init}}` is 1 for the initial state, and otherwise 0.

        :param threshold: The threshold :math:`\lambda` for which the Farkas y-constraints should be constructed
        :type threshold: Float
        :return: :math:`(N+1) \\times C`-matrix :math:`M_y`, and :math:`N+1`-vector :math:`rhs_y` 
        :rtype: Tuple[scipy.sparse.dok_matrix, np.ndarray[float]]
        """
        C,N = self.__P.shape

        if (self.__fark_y_matr is not None) and (self.__fark_y_rhs is not None):
            self.__fark_y_rhs[N] = -threshold
            return self.__fark_y_matr, self.__fark_y_rhs

        b = cast_dok_matrix(self.to_target)

        self.__fark_y_rhs = np.zeros(N+1)
        self.__fark_y_rhs[self.initial] = 1
        self.__fark_y_rhs[N] = -threshold

        self.__fark_y_matr = hstack((self.A,-b)).T

        return self.__fark_y_matr, self.__fark_y_rhs

    def _reach_form_id_matrix(self):
        """Computes the matrix :math:`I` for a given reachability form that for every row (st,act) has an entry 1 at the column corresponding to st."""
        C,N = self.__P.shape
        I = dok_matrix((C,N))

        for i in range(0,C):
            (state, _) = self.__index_by_state_action.inv[i]
            I[i,state] = 1

        return I

    def max_z_state(self,solver="cbc"):
        """
        Returns a solution to the LP        

        .. math::

            \max \, \sum_{s} \mathbf{x}(s) \quad \\text{ subject to } \quad \mathbf{x} \in \mathcal{P}^{\\text{min}}(0)
            
        The solution vector corresponds to the minimal reachabiliy probability, i.e. 
        :math:`\mathbf{x}^*(s) = \mathbf{Pr}^{\\text{min}}_s(\diamond \\text{goal})` for all :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        
        C,N = self.__P.shape
        matr, rhs = self.fark_z_constraints(0)
        opt = np.ones(N)
        max_z_lp = LP.from_coefficients(
            matr,rhs,opt,sense="<=",objective="max")

        for st_idx in range(N):
            if self.in_proper_ec(st_idx):
                max_z_lp.add_constraint([(st_idx,1)],"=",0)
            else:
                max_z_lp.add_constraint([(st_idx,1)],">=",0)
                max_z_lp.add_constraint([(st_idx,1)],"<=",1)

        result = max_z_lp.solve(solver=solver)
        return result.result_vector

    def max_z_state_action(self,solver="cbc"):
        """
        Let :math:`\mathbf{x}` be a solution vector to `max_z_state`. This function then returns a 
        :math:`C` vector :math:`\mathbf{v}` such that

        .. math::

            \mathbf{v}((s,a)) = \mathbf{P}((s,a),\\text{goal}) + \sum_{d \in S } \mathbf{P}((s,a),d) \mathbf{x}(d)

        for all :math:`(s,a) \in \mathcal{M}`.

        :param solver: [description], defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        

        max_z_vec = self.max_z_state(solver=solver)
        return self.__P.dot(max_z_vec) + self.to_target.A1

    def max_y_state_action(self,solver="cbc"):
        """
        Returns a solution to the LP        

        .. math::

            \max \, \mathbf{b} \, \mathbf{x} \quad \\text{ subject to } \quad \mathbf{x} \in \mathcal{P}^{\\text{max}}(0)
            
        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """
        C,N = self.__P.shape

        matr, rhs = self.fark_y_constraints(0)
        max_y_lp = LP.from_coefficients(
            matr,rhs,self.to_target,sense="<=",objective="max")

        for sap_idx in range(C):
            max_y_lp.add_constraint([(sap_idx,1)],">=",0)

        result = max_y_lp.solve(solver=solver)
        return result.result_vector

    def max_y_state(self,solver="cbc"):
        """
        Let :math:`\mathbf{x}` be a solution vector to `max_y_state_action`. This function then returns a 
        :math:`N` vector :math:`\mathbf{v}` such that

        .. math::

            \mathbf{v}(s) = \sum_{a \in \\text{Act}(s)} \mathbf{x}((s,a))

        for all :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        
        C,N = self.__P.shape
        max_y_vec = self.max_y_state_action(solver=solver)
        max_y_states = np.zeros(N)
        max_y_states[self.initial] = 1
        for sap_idx in range(C):
            (st,act) = self.__index_by_state_action.inv[sap_idx]
            max_y_states[st] = max_y_states[st] + max_y_vec[sap_idx]
        return max_y_states

    def pr_min(self,solver="cbc"):
        """Computes an :math:`N` vector :math:`\mathbf{x}` such that 
        :math:`\mathbf{x}(s) = \mathbf{Pr}^{\\text{min}}_s(\diamond \\text{goal})` for :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """        
        return self.max_z_state(solver=solver)

    def pr_max(self,solver="cbc"):
        """Computes an :math:`N` vector :math:`\mathbf{x}` such that :math:`\mathbf{x}(s) = 
        \mathbf{Pr}^{\\text{max}}_s(\diamond \\text{goal})` for :math:`s \in S`.

        :param solver: Solver that should be used, defaults to "cbc"
        :type solver: str, optional
        :return: Result vector
        :rtype: np.ndarray[float]
        """
        C,N = self.__P.shape

        matr, rhs = self.fark_z_constraints(1)
        opt = np.ones(N)
        pr_max_z_lp = LP.from_coefficients(
            matr,rhs,opt,sense=">=",objective="min")

        for st_idx in range(N):
            pr_max_z_lp.add_constraint([(st_idx,1)],">=",0)
            pr_max_z_lp.add_constraint([(st_idx,1)],"<=",1)

        result = pr_max_z_lp.solve(solver=solver)
        return result.result_vector


    # check that the only proper mecs are the ones induced by target and fail
    def _check_mec_freeness(self):
        assert sum(self.__proper_mecs) == 2, sum(self.__proper_mecs)
