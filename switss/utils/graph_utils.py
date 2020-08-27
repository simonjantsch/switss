import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix

def underlying_graph(P):
    rows,cols = np.shape(P)
    G = nx.DiGraph()
    G.add_nodes_from(range(0,cols))
    for (source,dest) in P.keys():
        G.add_edge(source,dest)
    return G

def scc_graph(G):
    components = list(nx.strongly_connected_components(G))
    return nx.condensation(G,components)

def sub_matrix(node_set,P):
    N = len(node_set)
    P_sub = dok_matrix((N,N))
    node_map = dict([])
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
