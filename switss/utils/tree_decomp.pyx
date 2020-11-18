# cython: language_level=3
# cython: profile=True

import itertools as it
import functools as ft
import numpy as np
from scipy import linalg
from scipy.spatial import ConvexHull

from bidict import bidict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .graph_utils import *
from .casting import cast_dok_matrix
from ..solver.milp import LP

import graph_tool.all as gt

from dd import cudd as _bdd
import time as time

cimport cython
from libc.stdlib cimport malloc, free

cdef struct StateProb:
    int state_idx
    float prob

cdef struct Point:
    int no_states
    StateProb* probs

cdef void free_point(Point* p):
    if p != NULL:
        if p.probs != NULL:
            free(p.probs)
        free(p)

cdef void points_test(int i):
    cdef Point empty_point = Point(0,NULL)
    cdef StateProb* some_state_probs = <StateProb *> malloc(i * sizeof(StateProb))
    for k in range(i):
        some_state_probs[k].state_idx = k
    empty_point.probs = some_state_probs

cdef StateProb* py_to_c_probs(state_probs_py):
    cdef int l = len(state_probs_py)
    if l == 0:
        return NULL
    cdef StateProb* state_probs = <StateProb* > malloc(l * sizeof(StateProb))
    for i in range(len(state_probs_py)):
        idx,p = list(state_probs_py)[i]
        state_probs[i].state_idx = idx
        state_probs[i].prob = p
    return state_probs

cdef Point* py_to_c_points(suc_points_py):
    cdef int l = len(suc_points_py)
    if l == 0:
        return NULL
    cdef Point* suc_points_c = <Point* > malloc(l * sizeof(Point))
    for i in range(len(suc_points_py)):
        suc_points_c[i].no_states = suc_points_py[i]['states']
        suc_points_c[i].probs = py_to_c_probs(suc_points_py[i]['in_probs'])
    return suc_points_c

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


def min_witnesses_from_tree_decomp(rf,partition,thr,known_upper_bound):
    cdef Point* suc_points_c = NULL

    G = underlying_graph_graphtool(rf.P)

    edge_probs = G.new_ep("double")
    for e in G.edges():
        edge_probs[e] = rf.P[e.source(),e.target()]
    G.edge_properties["edge_probs"] = edge_probs

    decomp, labeling, interface = quotient(G,partition)

    ## hack to get better partition

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
    #gt.graph_draw(decomp, output="quotient.pdf",output_size=(300,300),vertex_text=decomp.vertex_index,vertex_size=5)
    #gt.graph_draw(G, output="input.pdf",output_size=(1000,1000),vertex_text=G.vertex_index,vertex_fill_color=interface,vertex_size=10,edge_text=edge_probs)
    #print(rev_top_order)

    # the "solved" partitions have an entry in the following dictionary, which includes a set of relevant points
    partition_points = dict()

    for part_id in rev_top_order:
        print("\n\n**in partition " + str(part_id) + "**\n\n")

        suc_ps = decomp.get_out_neighbors(decomp.vertex(part_id))

        part_view = gt.GraphView(G,
                                 vfilt = lambda v: (labeling[v] == part_id) or ((labeling[v] in suc_ps) and interface[v]))

        #gt.graph_draw(part_view, output="part " + str(part_id) + ".pdf",vertex_fill_color=interface,vertex_size=3)

        is_in = part_view.new_vp("bool")
        is_out = part_view.new_vp("bool")
        is_target = part_view.new_vp("bool")

        for v in part_view.get_vertices():
            if (interface[v] and (labeling[v] == part_id)) or (rf.initial == v):
                is_in[v] = True
            if interface[v] and (labeling[v] != part_id):
                is_out[v] = True
            if (rf.to_target[v] > 0) or len([i for i in part_view.get_out_neighbours(v) if (interface[i] and (labeling[i] != part_id))]) > 0:
                is_target[v] = True
        part_view.vertex_properties["is_in"] = is_in
        part_view.vertex_properties["is_out"] = is_out
        part_view.vertex_properties["is_target"] = is_target

        # collect the points from successors in quotient graph
        suc_points = []
        points_per_suc = np.zeros(len(suc_ps))
        for i in range(len(suc_ps)):
            suc_points.append(partition_points[suc_ps[i]])
            points_per_suc[i] = len(partition_points[partition_points[suc_ps[i]]])
        # if there is no such successor, add the singleton list containing the 'zero point'
        if len(suc_ps) == 0:
            # figure out encoding of points...
            #suc_points.append([np.zeros(1)])
            suc_points.append([dict({ 'states' : 0, 'in_probs' : dict() })])

        suc_points_c = py_to_c_points(suc_points)

        #list containing the points of p
        partition_points[part_id] = []

        # convex hulls used to remove unnecessary points
        p_hulls = dict([])

        part_to_input_mapping = bidict()
        inp_dim = 0
        for i in part_view.get_vertices():
            if is_in[i]:
                part_to_input_mapping[i] = inp_dim
                inp_dim += 1

        # properly handle lower-dim cases!
        if inp_dim < 2:
            break

        k_points = dict()

        # enumerate rel. subsystems of part_view

        bdd, p_expr = compute_subsys_bdd(part_view)
        for model in bdd.pick_iter(p_expr):
            states = bidict(zip(range(part_view.num_vertices()),
                                [ i for i in part_view.get_vertices() if model['x{i}'.format(i=i)] == True ]))
            subsys_points = handle_subsys(states,
                                          suc_points,
                                          part_to_input_mapping,
                                          rf,
                                          part_view,
                                          inp_dim,
                                          known_upper_bound,
                                          thr,
                                          points_per_suc)

            for k in subsys_points.keys():
                if k not in k_points.keys():
                    k_points[k] = subsys_points[k]
                else:
                    k_points[k].extend(subsys_points[k])

        k_vertices = dict()
        conv_hull = None
        for k in sorted(k_points.keys()):
            #is this slow?
            points_to_add = np.concatenate((*map(lambda p: points_to_add_in_inp_space_v2(p,part_to_input_mapping,inp_dim), k_points[k]),))

            if conv_hull == None:
                # todo: figure out how to handle degenerate cases!
                # todo: handle dim 1 case!
                start = time.perf_counter()
                conv_hull = ConvexHull(np.append(points_to_add,np.array([np.zeros(inp_dim)]),axis=0),
                                       incremental=True,qhull_options='QJ1e-12')
                print("\nhull-time: " + str(time.perf_counter() - start) + "\n")
            else:
                start = time.perf_counter()
                conv_hull.add_points(points_to_add)
                print("\nhull-time: " + str(time.perf_counter() - start) + "\n")

            # pp = PdfPages(str(k) + '_conv_hull.pdf')
            # plt.plot(conv_hull.points[:,0], conv_hull.points[:,1], 'o')

            # for simplex in conv_hull.simplices:

            #     plt.plot(conv_hull.points[simplex, 0], conv_hull.points[simplex, 1], 'k-')

            # pp.savefig()
            # pp.close()

            k_vertices[k] = conv_hull.vertices

        conv_hull.close()

        #tip = lambda p: to_inp_point(p,part_to_input_mapping,inp_dim)
        fip = lambda p,k: from_inp_point(p,part_to_input_mapping,inp_dim,k)

        sofar = []
        for k in sorted(k_vertices.keys()):
            # cythonize this part:
            k_vertex_points = []
            for pnt in k_vertices[k]:
                k_vertex_points.append(conv_hull.points[pnt])

            k_points[k] = [p for p in k_points[k]
                           if (arreqclose_in_list2(p, k_vertex_points,inp_dim)
                               and not (arreqclose_in_list2(p, list(sofar), inp_dim)))]

            sofar.extend(k_points[k])
            partition_points[part_id].extend([fip(p,k) for p in k_points[k]])
        # print("\npartition " + str(part_id) + " points: \n" + str(partition_points[part_id]))
        # print("partition " + str(part_id) + " no-points: \n" + str(len(partition_points[part_id])) + "\n")

# from https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays
                 if elem.size == myarr.size and np.allclose(elem, myarr,atol=1e-10)), False)

# reimplement this using boost is-close-check
def arreqclose_in_list2(myarr, list_arrays, dim):
    for elem in list_arrays:
        for i in range(dim):
            if not (((myarr[i] - elem[i]) < 1e-10) and ((elem[i] - myarr[i]) < 1e-10)):
                break
            return True
    return False

def compute_subsys_bdd(part_view):
    is_in = part_view.vertex_properties["is_in"]
    is_out = part_view.vertex_properties["is_out"]
    is_target = part_view.vertex_properties["is_target"]

    bdd = _bdd.BDD()
    vrs = ['x{i}'.format(i=i) for i in part_view.get_vertices()]
    bdd.declare(*vrs)

    part_formula = []
    for i in part_view.get_vertices():
        if is_out[i]:
            continue
        elif not is_in[i] and not is_target[i]:
            i_out_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                               for j in part_view.get_out_neighbours(i)]) + ')'
            i_in_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                              for j in part_view.get_in_neighbours(i)]) + ')'
            part_formula.append(i_out_expr)
            part_formula.append(i_in_expr)
        elif not is_in[i] and is_target[i]:
            i_in_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                              for j in part_view.get_in_neighbours(i)]) + ')'
            part_formula.append(i_in_expr)
        elif is_in[i] and not is_target[i]:
            i_out_expr = '(x{i} => '.format(i=i) + ' | '.join(['x{j}'.format(j=j)
                                                               for j in part_view.get_out_neighbours(i)]) + ')'
            part_formula.append(i_out_expr)

    some_in_node = ' | '.join(['x{j}'.format(j=j) for j in part_view.get_vertices() if is_in[j]])

    some_target_node = ' | '.join(['x{j}'.format(j=j) for j in part_view.get_vertices() if is_target[j]])

    all_out_nodes = ' & '.join(['x{j}'.format(j=j) for j in part_view.get_vertices() if is_out[j]])

    big_conjunction = ' & '.join(part_formula + ["(" + some_in_node + ")", "(" + some_target_node + ")"])
    if all_out_nodes != "":
        big_conjunction = ' & '.join([big_conjunction, "(" + all_out_nodes + ")"])

    p_expr = bdd.add_expr(big_conjunction)

    return bdd, p_expr

def handle_subsys(states,
                  suc_points_c,
                  part_to_input_mapping,
                  rf,
                  part_view,
                  inp_dim,
                  known_upper_bound,
                  thr,
                  points_per_suc):
    is_in = part_view.vertex_properties["is_in"]
    is_out = part_view.vertex_properties["is_out"]
    no_states = len(states.keys())
    no_not_out_states = len([(i,q) for (i,q) in states.items() if not is_out[q]])

    P_sub = sub_matrix(no_states,states,part_view)

    subsys_points = dict()

    cdef int no_sucs = len(points_per_suc)

    cdef int* points_iterator = <int* > malloc(no_sucs * sizeof(int))
    cdef int [:] points_per_suc_view = points_per_suc
    cdef int tot_points = 0

    cdef int i = 0

    for i in range(no_sucs):
        points_iterator[i] = 0
        tot_points = tot_points * points_per_suc_view[i]

    for point in point_product(suc_points):

        total_no_states = no_not_out_states + point['states']

        if total_no_states > known_upper_bound:
            continue

        targ_vect = np.zeros(no_states)
        for i in range(no_states):
            state_idx = states[i]
            if is_out[state_idx] and state_idx in point['in_probs']:
                targ_vect[i] = point['in_probs'][state_idx]
            else:
                targ_vect[i] = rf.to_target[state_idx]

        reach_probs = compute_reach(no_states,P_sub,targ_vect)

        # this part needs to be changed into "rest-system" doesn't match threshold
        sum_inp_reach = 0
        for i in range(no_states):
            if is_in[states[i]]:
                sum_inp_reach += reach_probs[i]

        #fix the following ("if entire subsystem has less then thr...")
        if(sum_inp_reach < thr):
            continue

        p_probs = np.zeros(inp_dim)
        for (i,q) in states.items():
            if is_in[q]:
                p_probs[part_to_input_mapping[q]] = reach_probs[i]

        # in_probs_dict = dict([(q,reach_probs[i])
        #                       for (i,q) in states.items() if is_in[q]])

        # new_point = dict({ 'states' : total_no_states,
        #                    'in_probs': { **in_probs_dict }})

        if total_no_states not in subsys_points:
            subsys_points[total_no_states] = [p_probs]
        else:
            subsys_points[total_no_states].append(p_probs)

    return subsys_points

def to_inp_point(point,part_to_input_mapping,inp_dim):
    inp_point = np.zeros(inp_dim)
    for i in range(inp_dim):
        if part_to_input_mapping.inverse[i] in point['in_probs']:
            inp_point[i] = point['in_probs'][part_to_input_mapping.inverse[i]]
        else:
            inp_point[i] = 0
    return inp_point

def from_inp_point(point,part_to_input_mapping,inp_dim,k):
    p = dict()
    p["states"] = k
    p["in_probs"] = dict()
    for i in range(inp_dim):
        p["in_probs"][part_to_input_mapping.inverse[i]] = point[i]
    return p

def points_to_add_in_inp_space(new_point,part_to_input_mapping,inp_dim):
    main_point = to_inp_point(new_point,part_to_input_mapping,inp_dim)

    projections = []
    for i in range(inp_dim):
        i_proj = np.zeros(inp_dim)
        i_proj[i] = main_point[i]
        projections.append(i_proj)

    return np.array([main_point] + projections)

def points_to_add_in_inp_space_v2(new_point,part_to_input_mapping,inp_dim):
    projections = []

    for i in range(inp_dim):
        i_proj = np.zeros(inp_dim)
        i_proj[i] = new_point[i]
        projections.append(i_proj)

    return np.array([new_point] + projections)


cdef inline compute_reach(no_states,P,target):
    return linalg.solve(np.identity(no_states)-P,target)

## todo: use a c-struct instead of a python dict for points
def add_points(p1,p2):
    return dict({ 'states' : p1['states'] + p2['states'], 'in_probs' : {**p1['in_probs'], **p2['in_probs']}})

def add_points_np_arr(p1,p2):
    p1[0] += p2[0]
    p1.extend(p2[1:])
    return p1

def point_product(point_collections):
    for points in it.product(*point_collections):
        yield ft.reduce(add_points,points)

def point_product_np_arr(point_collections):
    for points in it.product(*point_collections):
        yield ft.reduce(add_points_np_arr,points)

def sub_matrix(int no_states,node_dict,part_view):
    P_sub = np.zeros((no_states,no_states))
    for (s,t,p) in part_view.get_edges(eprops = [part_view.edge_properties["edge_probs"]]):
        if s in node_dict.values() and t in node_dict.values():
            i = node_dict.inverse[s]
            j = node_dict.inverse[t]
            P_sub[i,j] = p
    return P_sub
