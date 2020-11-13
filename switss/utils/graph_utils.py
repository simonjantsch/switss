import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix

import graph_tool.all as gt
from bidict import bidict

def underlying_graph(P):
    rows,cols = np.shape(P)
    G = nx.DiGraph()
    G.add_nodes_from(range(0,cols))
    for (source,dest) in P.keys():
        G.add_edge(source,dest)
    return G

def underlying_graph_graphtool(P):
    rows,cols = np.shape(P)
    G = gt.Graph()
    G.add_vertex(cols)
    for (source,dest) in P.keys():
        G.add_edge(G.vertex(source),G.vertex(dest))
    return G

def quotient(G,partition):
    quotientGraph = gt.Graph()
    quotientGraph.add_vertex(len(partition))
    labeling = G.new_vertex_property("int32_t")
    interface = G.new_vertex_property("bool")
    for p in range(0,len(partition)):
        for v in partition[p]:
            labeling[G.vertex(v)] = p

    for e in G.edges():
        l_source = labeling[G.vertex(e.source())]
        l_target = labeling[G.vertex(e.target())]
        if l_source != l_target:
            interface[e.target()] = True
            quotientGraph.edge(quotientGraph.vertex(l_source),
                               quotientGraph.vertex(l_target),
                               add_missing = True)

    return quotientGraph,labeling,interface

def scc_graph(G):
    components = list(nx.strongly_connected_components(G))
    return nx.condensation(G,components)

# -> to cython (called very often)
def sub_matrix(node_set,P):
    N = len(node_set)
    P_sub = dok_matrix((N,N))
    node_map = bidict({})
    i = 0

    for n in node_set:
        node_map[n] = i
        i += 1

    for (n1,n2) in P.keys():
        if (n1 in node_set) and (n2 in node_set):
            P_sub[node_map[n1],node_map[n2]] = P[n1,n2]

    return P_sub, node_map

def interface(node_set1,node_set2,P):
    M = len(node_set1)
    N = len(node_set2)
    P_int = dok_matrix((M,N))

    node_map1 = dict([])
    node_map2 = dict([])

    i = 0
    for n in node_set1:
        node_map1[n] = i
        i += 1

    i = 0
    for n in node_set2:
        node_map2[n] = i
        i += 1

    for (n1,n2) in P.keys():
        if (n1 in node_set1) and (n2 in node_set2):
            P_int[node_map[n1],node_map[n2]] = P[n1,n2]

    return P_int,node_map1,node_map2
