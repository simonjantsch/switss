import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix
from scipy.spatial import ConvexHull
from scipy import linalg
from collections import namedtuple
import itertools as it
import functools as ft
from bidict import bidict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from .graph_utils import *
from .casting import cast_dok_matrix
from ..solver.milp import LP

import graph_tool.all as gt

from dd import autoref as _bdd
import time as time

def tree_decomp_from_partition_networkx(P,partition):
    G = underlying_graph(P)
    #nx.nx_pydot.write_dot(G,"orig.dot")
    decomp = nx.quotient_graph(G,partition)

    # check if the decomposition forms a tree
    if not nx.is_arborescence(decomp):
        print("partition does not induce a tree!")
        assert False

    # compute some stats:
    max_nodes = 0
    for n in decomp.nodes:
        max_nodes = max(max_nodes,decomp.nodes[n]['nnodes'])

    print(max_nodes)
    #nx.nx_pydot.write_dot(decomp,"tree_decomp.dot")


def fully_directed_tree_decomp(P):
    G = underlying_graph(P)
    scc_graph_G = scc_graph(G)
    tree_decomp = scc_graph_G.copy()
    for n in tree_decomp.nodes:
        # initialize labeling as singleton
        tree_decomp.nodes[n]['label'] = set([n])

    node_index = tree_decomp.number_of_nodes()
    count = 0
    while True:
        count += 1
        violating_node = next(( n for n in tree_decomp.nodes if tree_decomp.in_degree(n) > 1 ),None)
        if violating_node == None:
            break
        preds = set(tree_decomp.predecessors(violating_node))
        preds_labels_union = set().union( *(tree_decomp.nodes[p]['label'] for p in preds) )
        new_node = node_index
        node_index += 1
        tree_decomp.add_node(new_node)
        tree_decomp.nodes[new_node]['label'] = preds_labels_union

        # add predecessors of new_node
        for p in preds:
            for p_pr in tree_decomp.predecessors(p):
                tree_decomp.add_edge(p_pr,new_node)

        # add successors of new_node
        for p in preds:
            for p_suc in tree_decomp.successors(p):
                tree_decomp.add_edge(new_node,p_suc)

        #remove old nodes
        tree_decomp.remove_nodes_from(preds)

    print(count)
    #nx.nx_pydot.write_dot(tree_decomp,"tree_decomp.dot")
    #nx.nx_pydot.write_dot(SCC_graph,"scc_graph.dot")
    #nx.nx_pydot.write_dot(G,"G.dot")

    #compute some stats
    max_interface = 0
    max_node_size = 0
    for n in tree_decomp.nodes:
        # compute ns interface to its predecessor

        n_sccs = tree_decomp.nodes[n]['label']
        n_nodes = set().union( *(scc_graph_G.nodes[scc]['members'] for scc in n_sccs) )

        n_pred = next(tree_decomp.predecessors(n),None)
        if n_pred is not None:
            n_pred_sccs = tree_decomp.nodes[n_pred]['label']
            n_pred_nodes = set().union( *(scc_graph_G.nodes[scc]['members'] for scc in n_pred_sccs) )
            n_interface_size = len(set(nx.node_boundary(G,n_pred_nodes,n_nodes)))

            max_interface = max(max_interface,n_interface_size)
        max_node_size = max(max_node_size,len(n_nodes))

    print("max_interface: " + str(max_interface))
    print("max_node_size: " + str(max_node_size))


    return tree_decomp
