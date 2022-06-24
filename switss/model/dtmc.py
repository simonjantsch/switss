from bidict import bidict
from graphviz import Digraph
from scipy.sparse import dok_matrix
from numpy import zeros
import os.path

from . import AbstractMDP
from ..utils import color_from_hash, cast_dok_matrix, DTMCVisualizationConfig
from ..prism import prism

class DTMC(AbstractMDP):
    def __init__(self, P, label_to_states={}, index_by_state_action=None, vis_config=None,reward_vector=None, **kwargs):
        """Instantiates a DTMC from a transition matrix and labelings for states.

        :param P: :math:`N_{S_{\\text{all}}} \\times N_{S_{\\text{all}}}` transition matrix.
        :type P: Either 2d-list, numpy.matrix, numpy.array or scipy.sparse.spmatrix
        :param reward_vector: A vector containing a nonnegative reward per state
        :type reward_vector: Dict[int,int]
        :param label_to_states: Mapping from labels to sets of states.
        :type label_to_states: Dict[str,Set[int]]
        :param index_by_state_action: Mapping from states to their corresponding row-entries. Every
            key must have 0 for its action value. If None, then every row-index corresponds to the
            same column-index.
        :type index_by_state_action: Dict[Tuple[int,int],int]
        :param vis_config: Used to configure how model is visualized.
        :type vis_config: VisualizationConfig
        """
        # transform P into dok_matrix if neccessary
        P =  cast_dok_matrix(P)  
        assert P.shape[0] == P.shape[1], "P must be a (NxN)-matrix but has shape %s" % P.shape
        if index_by_state_action is None:
            index_by_state_action = bidict()
            for i in range(P.shape[0]):
                index_by_state_action[(i,0)] = i
        else:
            for s,a in index_by_state_action.keys():
                assert a == 0, "If state-actions are specified, DTMCs must have all 0-entries for actions: (%s,%s)" % (s,a)

        if vis_config is None:
            vis_config = DTMCVisualizationConfig()

        #TODO below might not be needed
        if reward_vector:
            correct_length_reward_vector = zeros(len(index_by_state_action), dtype= int)
            for (state, action), index in index_by_state_action.items():
                correct_length_reward_vector[index] = reward_vector[state]
            reward_vector = correct_length_reward_vector
        #===================
        
        super().__init__(P, index_by_state_action, {}, label_to_states, vis_config, reward_vector)

    def digraph(self, state_map = None, trans_map = None, **kwargs):
        """Creates a `graphviz.Digraph` object from this instance. When a digraph object is created, 
        new nodes are added for states plus additional edges for transitions between states. 
        `state_map` and `trans_map` are functions that, on some input, compute keyword arguments for
        the digraph instance. If any one of these is None, the default visualization config will be used. `action_map`
        is ignored.
        Any additional arguments will be passed to the Digraph(..) call of graphviz'.

        For example, these functions below are used as default parameters if no `state_map` or `trans_map` is specified.

        .. highlight:: python
        .. code-block:: python

            def standard_state_map(stateidx,labels):
                return { "color" : color_from_hash(tuple(sorted(labels))),
                         "label" : "State %d\\n%s" % (stateidx,",".join(labels)),
                         "style" : "filled" }

        .. highlight:: python
        .. code-block:: python
        
            def standard_trans_map(sourceidx, destidx, p):
                return { "color" : "black", 
                         "label" : str(round(p,10)) }

        where `color_from_hash` is imported from `switss.utils`. For further information on graphviz attributes, 
        see https://www.graphviz.org/doc/info/attrs.html. 

        :param state_map: A function that computes parameters for state-nodes, defaults to None.
        :type state_map: (stateidx : int, labels : Set[str]) -> Dict[str,str], optional
        :param trans_map: A function that computes parameters for edges between actions and nodes, defaults to None. 
        :type trans_map: (sourceidx : int, destidx : int, p : float) -> Dict[str,str], optional
        :return: The digraph instance.
        :rtype: graphviz.Digraph
        """ 

        state_map = self.visualization.state_map if state_map is None else state_map
        trans_map = self.visualization.trans_map if trans_map is None else trans_map

        dg = Digraph()

        # connect nodes between each other
        existing_nodes = set({})

        for (rowidx, dest), p in self.P.items():
            # transition from source to dest w/ probability p
            source,_ = self.index_by_state_action.inv[rowidx]
            for node in [source, dest]:
                if node not in existing_nodes:
                    state_setting = state_map(node, self.labels_by_state[node])
                    dg.node(str(node), **state_setting)
                    existing_nodes.add(node)

            params = (source, dest, p)
            trans_setting = trans_map(*params)
            dg.edge(str(source), str(dest), **trans_setting)

        return dg

    def save(self, filepath):   
        tra_path = filepath + ".tra"
        lab_path = filepath + ".lab"
        srew_path = filepath+".srew"

        with open(tra_path, "w") as tra_file:
            tra_file.write("%d %d\n" % (self.N, self.P.nnz))
            for (rowidx,dest), p in self.P.items():
                source,_ = self.index_by_state_action.inv[rowidx]
                tra_file.write("%d %d %f\n" % (source, dest, p))

        with open(lab_path, "w") as lab_file:
            unique_labels_list = list(self.states_by_label.keys())
            header = ["%d=\"%s\"" % (i, label) for i,label in enumerate(unique_labels_list)]
            lab_file.write("%s\n" % (" ".join(header)))
            for idx, labels in self.labels_by_state.items():
                if len(labels) == 0:
                    continue
                labels_str = " ".join(map(str, map(unique_labels_list.index, labels)))
                lab_file.write("%d: %s\n" % (idx, labels_str))

        #TODO add save functionality for rewards
        #TODO think about this loop
        #goal is to get nonzero rewards and not duplicates because of the way we implemented state rewards
        non_zero_amount = 0
        for (state, action), index in self.index_by_state_action.items():
            if(self.reward_vector[index] == 0):
                non_zero_amount +=1

        with open(srew_path, "w") as srew_file:
            srew_file.write("%d %d" % (self.N, non_zero_amount))
            for (state, action), index in self.index_by_state_action.items():
                current_reward = self.reward_vector[index]
                if( current_reward != 0):
                    srew_file.write("%d %d" % (state, current_reward))

        return tra_path, lab_path, srew_path

    @classmethod
    def _load_transition_matrix(cls, filepath):
        P = dok_matrix((1,1))
        N = 0

        with open(filepath) as tra_file:
            for line in tra_file:
                line_split = line.split()
                # check for first line, which has format "#states #transitions"
                if len(line_split) == 2:
                    N = int(line_split[0])
                    P.resize((N,N))
                # all other lines have format "from to prob"
                else:
                    source = int(line_split[0])
                    dest = int(line_split[1])
                    prob = float(line_split[2])
                    P[source,dest] = prob
        return { "P" :  P }

    @classmethod
    def from_stormpy_model(cls,stormpy_model):
        
        #assert stormpy_model.model_type == "ModelType.MDP"

        P = dok_matrix((1,1))
        index_by_state_action = bidict()
        N = stormpy_model.nr_states

        P.resize((N,N))

        for state in stormpy_model.states:
            sid = state.id
            for action in state.actions:
                for transition in action.transitions:
                    P[sid,transition.column] = transition.value()

        return { "P" : P }

    def insert_rewards(self, filepath):
        if(os.path.isfile(filepath)):
            if(filepath.endswith(".srew")):
                #in srew files the first line contains the number 
                #of states in the model and the amount of states with non-zero reward 
                with open(filepath,"r") as srew_file:
                    first_line_split = srew_file.readline().split()
                    N = int(first_line_split[0])
                    parsed_reward_vector = zeros(N, dtype=int)
                    for line in srew_file.readlines():
                        # of format "state reward"
                        line_split = line.split()
                        state = int(line_split[0])
                        state_reward = int(line_split[1])
                        #TODO Question: insert at 'next' index or according to state index in srew file. Currently using according to state index
                        parsed_reward_vector[state] = state_reward
                    self.reward_vector = parsed_reward_vector    
            else:
                print("error: file path in function _load_rewards doesnt end with or '.srew'")
        else:
            print("Given file/filepath does not exist")


    @classmethod
    def _load_rewards(cls, filepath):
        if(os.path.isfile(filepath)):
            if(filepath.endswith(".srew")):
                #in srew files the first line contains the number 
                #of states in the model and the amount of states with non-zero reward 
                with open(filepath,"r") as srew_file:
                    first_line_split = srew_file.readline().split()
                    N = int(first_line_split[0])
                    reward_vector = zeros(N, dtype=int)
                    for line in srew_file.readlines():
                        # of format "state reward"
                        line_split = line.split()
                        state = int(line_split[0])
                        state_reward = int(line_split[1])
                        reward_vector[state] = state_reward
                        
                    return reward_vector
            else:
                print("error: file path in function _load_rewards doesnt end with or '.srew'")
        else:
            print("Given file/filepath does not exist")


    def _load_rewards_instance(self,filepath):
        if(os.path.isfile(filepath)):
            if(filepath.endswith(".srew")):
                #in srew files the first line contains #States #states with non-zero reward
                with open(filepath,"r") as srew_file:

                    rows, cols = self.P.shape

                    first_line_split = srew_file.readline().split()
                    N = int(first_line_split[0])

                    if(N != cols):
                        print("Error! The .srew file gives rewards for a system of a different shape")

                    reward_vector = zeros(N, dtype=int)    
                    for line in srew_file.readlines():
                        # of format "state reward"
                        line_split = line.split()
                        state = int(line_split[0])
                        state_reward = int(line_split[1])
                        reward_vector[state] = state_reward
                    #LAST PROGRESS
                    correct_length_reward_vector = zeros(len(self.index_by_state_action), dtype= int)
                    for (state, action), index in self.index_by_state_action.items():
                        correct_length_reward_vector[index] = reward_vector[state]
                    self.reward_vector = correct_length_reward_vector

            
           
            else:
                print("error: file path in function _load_rewards doesnt end with '.srew'")
                print("Stopping parsing process")
        else:
            print("Given file/filepath does not exist")