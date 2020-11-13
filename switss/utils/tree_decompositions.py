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


def is_tree(G):
    max_in_degree = max(G.get_in_degrees(G.get_vertices()))
    u = gt.extract_largest_component(G, directed = True)
    return ((len(u.get_vertices()) == 1) and (max_in_degree == 1))

def tree_decomp_from_partition_graphtool(P,partition):
    G = underlying_graph_graphtool(P)
    #nx.nx_pydot.write_dot(G,"orig.dot")
    decomp,labeling,interface = quotient(G,partition)

    # check if the decomposition forms a tree
    assert is_tree(decomp)

    #gt.graph_draw(G, output="input.pdf",output_size=(1500,1500),vertex_text=G.vertex_index,vertex_fill_color=interface,vertex_size=3)
    #gt.graph_draw(decomp, output="quotient.pdf",output_size=(300,300),vertex_text=decomp.vertex_index,vertex_size=5)


def min_witnesses_from_tree_decomp(rf,partition,thr):
    G = underlying_graph_graphtool(rf.P)

    edge_probs = G.new_ep("double")
    for e in G.edges():
        edge_probs[e] = rf.P[e.source(),e.target()]

    decomp, labeling, interface = quotient(G,partition)

    ## hack to get differen partition

    new_partition = []
    for p in decomp.get_vertices():
        p_sucs = decomp.get_out_neighbors(decomp.vertex(p))
        new_p_states = [q for q in G.get_vertices()
                        if (labeling[q] == p) and (not interface[q]) or (labeling[q] in p_sucs and interface[p])]
        new_partition.append(new_p_states)

    decomp, labeling, interface = quotient(G,new_partition)
    ##


    # compute an order by which to process the quotient

    rev_top_order = np.flipud(gt.topological_sort(decomp))
    gt.graph_draw(decomp, output="quotient.pdf",output_size=(300,300),vertex_text=decomp.vertex_index,vertex_size=5)
    gt.graph_draw(G, output="input.pdf",output_size=(1000,1000),vertex_text=G.vertex_index,vertex_fill_color=interface,vertex_size=10,edge_text=edge_probs)
    #print(rev_top_order)

    # go bottom up and on each level compute function
    # post-points x level-subgraph -> level-points
    # where *-points is a set of points in N times Q^{interface}

    # the "solved" partitions have an entry in the following dictionary, which includes a set of relevant points
    partition_points = dict()

    for part_id in rev_top_order:
        print("\n\n**in partition " + str(part_id) + "**\n\n")

        suc_ps = decomp.get_out_neighbors(decomp.vertex(part_id))

        part_view = gt.GraphView(G,
                                 vfilt = lambda v: (labeling[v] == part_id) or ((labeling[v] in suc_ps) and interface[v]))

        gt.graph_draw(part_view, output="part " + str(part_id) + ".pdf",vertex_fill_color=interface,vertex_size=3)

        # maybe changes these from lambdas to (np.)arrays?
        is_in = lambda v: (interface[v] and (labeling[v] == part_id)) or (rf.initial == v)
        is_out = lambda v: interface[v] and (labeling[v] != part_id)
        is_target = lambda v: (rf.to_target[v] > 0) or len(np.intersect1d(part_view.get_out_neighbours(v), [i for i in part_view.get_vertices() if is_out(i)])) > 0

        # collect the points from successors in quotient graph
        suc_points = []
        for suc_p in suc_ps:
            suc_points.append(partition_points[suc_p])

        # if there is no such successor, add the singleton list containing the 'zero point'
        if len(suc_ps) == 0:
            suc_points.append([dict({ 'states' : 0, 'in_probs' : dict() })])

        #list containing the points of p
        partition_points[part_id] = []

        # convex hulls used to remove unnecessary points
        p_hulls = dict([])

        part_to_input_mapping = bidict()
        inp_dim = 0
        for i in part_view.get_vertices():
            if is_in(i):
                part_to_input_mapping[i] = inp_dim
                inp_dim += 1

        k_points = dict()

        # enumerate rel. subsystems of part_view

        bdd, p_expr = compute_subsys_bdd(part_view,is_in,is_out,is_target)
        for model in bdd.pick_iter(p_expr):
            states = [ i for i in part_view.get_vertices() if model['x{i}'.format(i=i)] == True ]
            k_points = handle_subsys(states,
                                     k_points,
                                     rf,
                                     suc_points,
                                     part_to_input_mapping,
                                     is_out,
                                     is_in,
                                     part_view,
                                     inp_dim)

        k_vertices = dict()
        conv_hull = None
        for k in sorted(k_points.keys()):
            #is this slow?
            points_to_add = np.concatenate((*map(lambda p: points_to_add_in_inp_space(p,part_to_input_mapping,inp_dim), k_points[k]),))

            if conv_hull == None:
                # todo: figure out how to handle degenerate cases!
                # todo: handle dim 1 case!
                conv_hull = ConvexHull(np.append(points_to_add,np.array([np.zeros(inp_dim)]),axis=0),
                                       incremental=True,qhull_options='QJ1e-12')
            else:
                conv_hull.add_points(points_to_add)

            # pp = PdfPages(str(k) + '_conv_hull.pdf')
            # plt.plot(conv_hull.points[:,0], conv_hull.points[:,1], 'o')

            # for simplex in conv_hull.simplices:

            #     plt.plot(conv_hull.points[simplex, 0], conv_hull.points[simplex, 1], 'k-')

            # pp.savefig()
            # pp.close()

            k_vertices[k] = conv_hull.vertices

        conv_hull.close()
        print("k_vertices:\n" + str(k_vertices))

        # for k in range(part_view.num_vertices()):
        #     if k in k_vertices:
        #         print("convex hull at most" + str(k) + ": " )
        #         for p in k_vertices[k]:
        #             print(conv_hull.points[p])

        tip = lambda p: to_inp_point(p,part_to_input_mapping,inp_dim)

        sofar = []
        for k in sorted(k_vertices.keys()):
            # cythonize this part:
            k_vertex_points = []
            for pnt in k_vertices[k]:
                k_vertex_points.append(conv_hull.points[pnt])

            k_points[k] = [p for p in k_points[k]
                           if (arreqclose_in_list(tip(p), k_vertex_points)
                               and not (arreqclose_in_list(tip(p), list(map(tip,sofar)))))]

            sofar.extend(k_points[k])

        partition_points[part_id] = sofar.copy()
        print("\npartition " + str(part_id) + " points: \n" + str(partition_points[part_id]))
        print("partition " + str(part_id) + " no-points: \n" + str(len(partition_points[part_id])) + "\n")

# from https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays
                 if elem.size == myarr.size and np.allclose(elem, myarr,atol=1e-10)), False)

def compute_subsys_bdd(part_view,is_in,is_out,is_target):
    bdd = _bdd.BDD()
    vrs = ['x{i}'.format(i=i) for i in part_view.get_vertices()]
    bdd.declare(*vrs)

    part_formula = []
    for i in part_view.get_vertices():
        if is_out(i):
            continue
        elif not is_in(i) and not is_target(i):
            i_out_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                               for j in part_view.get_out_neighbours(i)]) + ')'
            i_in_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                              for j in part_view.get_in_neighbours(i)]) + ')'
            part_formula.append(i_out_expr)
            part_formula.append(i_in_expr)
        elif not is_in(i) and is_target(i):
            i_in_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                              for j in part_view.get_in_neighbours(i)]) + ')'
            part_formula.append(i_in_expr)
        elif is_in(i) and not is_target(i):
            i_out_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                               for j in part_view.get_out_neighbours(i)]) + ')'
            part_formula.append(i_out_expr)

    some_in_node = ' | '.join(['x{j}'.format(j=j) for j in part_view.get_vertices() if is_in(j)])

    some_target_node = ' | '.join(['x{j}'.format(j=j) for j in part_view.get_vertices() if is_target(j)])

    all_out_nodes = ' & '.join(['x{j}'.format(j=j) for j in part_view.get_vertices() if is_out(j)])

    big_conjunction = ' & '.join(part_formula + ["(" + some_in_node + ")", "(" + some_target_node + ")"])
    if all_out_nodes != "":
        big_conjunction = ' & '.join([big_conjunction, "(" + all_out_nodes + ")"])

    p_expr = bdd.add_expr(big_conjunction)

    return bdd, p_expr


# rewrite this in Cython, don't pass so many arguments (doesn't need k_points)
def handle_subsys(states,k_points,rf,suc_points,part_to_input_mapping,is_out,is_in,part_view,inp_dim):
    # these two need to be given as input
    thr = 1e-6
    states_bound = 98

    no_states = len(states)
    P_sub, part_to_sub_mapping = sub_matrix(states,rf.P)

    for point in point_product(suc_points):

        total_no_states = len([q for q in states if not is_out(q)]) + point['states']

        if total_no_states > states_bound:
            continue

        targ_vect = np.zeros(no_states)
        for i in range(no_states):
            state_idx = part_to_sub_mapping.inverse[i]
            if is_out(state_idx):
                targ_vect[i] = point['in_probs'][state_idx]
            else:
                targ_vect[i] = rf.to_target[state_idx]

        reach_probs = compute_reach(P_sub,targ_vect)

        # this part needs to be changes into "rest-system" doesn't match threshold
        sum_inp_reach = 0
        for i in range(no_states):
            if is_in(part_to_sub_mapping.inverse[i]):
                sum_inp_reach += reach_probs[i]

        #fix the following ("if entire subsystem has less then thr...")
        if(sum_inp_reach < thr):
            continue

        in_probs_dict = dict([(i,reach_probs[part_to_sub_mapping[i]])
                              for i in states if is_in(i)])
        in_probs_not_in_subsys = dict([(i,0)
                                       for i in part_view.get_vertices()
                                       if not (i in states) and is_in(i)])

        new_point = dict({ 'states' : total_no_states,
                           'in_probs': { **in_probs_dict, **in_probs_not_in_subsys }})

        if total_no_states not in k_points:
            k_points[total_no_states] = [new_point]
        else:
            k_points[total_no_states].append(new_point)

        # for k in at_most_k_points.keys():
        #     if k > total_no_states:
        #         at_most_k_points[k].append(new_point)

    return k_points

def to_inp_point(point,part_to_input_mapping,inp_dim):
    inp_point = np.zeros(inp_dim)
    for i in range(inp_dim):
        inp_point[i] = point['in_probs'][part_to_input_mapping.inverse[i]]
    return inp_point

def points_to_add_in_inp_space(new_point,part_to_input_mapping,inp_dim):
    main_point = to_inp_point(new_point,part_to_input_mapping,inp_dim)

    ## todo: directly use np.array instead of python list for projections
    projections = []
    for i in range(inp_dim):
        i_proj = np.zeros(inp_dim)
        i_proj[i] = main_point[i]
        projections.append(i_proj)

    return np.array([main_point] + projections)

def compute_reach(P,target):
    N,N = P.shape
    I = np.identity(N)
    return linalg.solve(I-P.todense(),target)

## todo: use a c-struct instead of a python dict for points
def add_points(p1,p2):
    return dict({ 'states' : p1['states'] + p2['states'], 'in_probs' : {**p1['in_probs'], **p2['in_probs']}})

def point_product(point_collections):
    for points in it.product(*point_collections):
        yield ft.reduce(add_points,points)

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
