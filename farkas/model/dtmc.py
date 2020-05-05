from bidict import bidict
from graphviz import Digraph
from scipy.sparse import dok_matrix

from . import AbstractMDP
from ..utils import color_from_hash, cast_dok_matrix
from ..prism import prism

class DTMC(AbstractMDP):
    def __init__(self, P, label_to_states, **kwargs):
        # transform P into dok_matrix if neccessary
        P =  cast_dok_matrix(P)  
        assert P.shape[0] == P.shape[1], "P must be a (NxN)-matrix but has shape %s" % P.shape
        index_by_state_action = bidict()
        for i in range(P.shape[0]):
            index_by_state_action[(i,0)] = i
        super().__init__(P, index_by_state_action, {}, label_to_states)

    def digraph(self, state_map = None, trans_map = None, action_map = None):
        """Creates a graphviz.Digraph object from this instance. When a digraph object is created, 
        new nodes are added for states plus additional transitions which are edges between nodes. 
        `state_map` and `trans_map` are functions that, on some input, compute keyword arguments for
        the digraph instance. If any one of these is None, the default mapping will be used.
        
        For example, these functions below are used as default parameters if no `state_map` or `trans_map` is specified.
        
        .. highlight:: python
        .. code-block:: python

            def standard_state_map(stateidx):
                labels = self.labels_by_state[stateidx]
                return { "color" : color_from_hash(tuple(sorted(labels))),
                        "label" : "State %d\\n%s" % (stateidx,",".join(labels)),
                        "style" : "filled" }

        .. highlight:: python
        .. code-block:: python
        
            def standard_trans_map(sourceidx, destidx, p):
                return { "color" : "black", 
                         "label" : str(round(p,10)) }

        For further information on graphviz attributes, see https://www.graphviz.org/doc/info/attrs.html. 


        :param state_map: A function that computes parameters for state-nodes, defaults to None. If the function returns None,
            no node for this state will be drawn.
        :type state_map: (stateidx : int, labels : Set[str]) -> Dict[str,str], optional
        :param trans_map: A function that computes parameters for edges between actions and nodes, defaults to None. 
            If the function returns None, no edge between the given action and destination will be drawn.
        :type trans_map: (sourceidx : int, destidx : int, sourcelabels : Set[str], destlabels : Set[str], p : float) -> Dict[str,str], optional
        :return: The digraph instance.
        :rtype: graphviz.Digraph
        """ 

        def standard_state_map(stateidx,labels):
            return { "color" : color_from_hash(tuple(sorted(labels))),
                     "label" : "State %d\n%s" % (stateidx,",".join(labels)),
                     "style" : "filled" }

        def standard_trans_map(sourceidx, destidx, p):
            return { "color" : "black", 
                     "label" : str(round(p,10)) }

        state_map = standard_state_map if state_map is None else state_map
        trans_map = standard_trans_map if trans_map is None else trans_map

        dg = Digraph()

        # connect nodes between each other
        existing_nodes = set({})

        for (source, dest), p in self.P.items():

            # transition from source to dest w/ probability p
            if p > 0:
                for node in [source, dest]:
                    if node not in existing_nodes:
                        # print(self.labels[node])
                        state_setting = state_map(
                            node, self.labels_by_state[node])
                        if state_setting is not None:
                            dg.node(str(node), **state_setting)
                        existing_nodes.add(node)

                params = (source, dest, p)
                trans_setting = trans_map(*params)
                if trans_setting is not None:
                    dg.edge(str(source), str(dest), **trans_setting)

        return dg

    def save(self, filepath):   
        tra_path = filepath + ".tra"
        lab_path = filepath + ".lab"

        with open(tra_path, "w") as tra_file:
            tra_file.write("%d %d\n" % (self.N, self.P.nnz))
            for (source,dest), p in self.P.items():
                if p > 0:
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

        return tra_path, lab_path

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
        return P,
