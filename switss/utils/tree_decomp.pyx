# cython: language_level=3
# cython: profile=True
# distutils: include_dirs = /usr/local/Cellar/numpy/1.19.4/lib/python3.9/site-packages/numpy/core/include

import itertools as it
from more_itertools import powerset
import functools as ft
import numpy as np
cimport numpy as np
from scipy import linalg
from scipy.spatial import ConvexHull
from scipy.linalg.cython_lapack cimport dgesv

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

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy

ctypedef np.uint8_t uint8

cdef int INT_MAX = 2147483647

cdef struct StateProb:
    int state_idx
    double prob

cdef struct Point:
    int no_states
    int no_probs
    StateProb* probs

cdef struct SortedInputPoints:
    int size
    int no_points
    double** points
    SortedInputPoints* next_entry

cdef void print_point(Point p):
    print("no_states:" + str(p.no_states))
    print("no_probs:" + str(p.no_probs))
    print("**probs**")
    for i in range(p.no_probs):
        print("(" + str(p.probs[i].state_idx) + ", " + str(p.probs[i].prob) + ")")

cdef void free_mem(Point* p, int no_points):
    if p == NULL:
        return
    cdef int i = 0
    for i in range(no_points):
        if p[i].probs != NULL:
            #print("freeing probs in loc: {0:x}".format(<unsigned int> p[i].probs))
            free(p[i].probs)
    #print("freeing points in loc: {0:x}".format(<unsigned int> p))
    free(p)

cdef StateProb* py_to_c_probs(state_probs_py):
    cdef int l = len(state_probs_py)
    cdef int i = 0
    if l == 0:
        return NULL
    cdef StateProb* state_probs = <StateProb* > malloc(l * sizeof(StateProb))
    #print("allocated " + str(l) + "* stateprob to memory-loc: {0:x}".format(<unsigned int>state_probs))
    for i in range(l):
        idx,p = list(state_probs_py.items())[i]
        state_probs[i].state_idx = idx
        state_probs[i].prob = p
    return state_probs

cdef Point* py_to_c_points(suc_points_py):
    cdef int l = len(suc_points_py)
    cdef int i = 0
    if l == 0:
        return NULL
    cdef Point* suc_points_c = <Point* > malloc(l * sizeof(Point))
    #print("allocated " + str(l) + "* Point to memory-loc: {0:x}".format(<unsigned int>suc_points_c))
    for i in range(l):
        suc_points_c[i].no_states = suc_points_py[i]['states']
        suc_points_c[i].no_probs = len(suc_points_py[i]['in_probs'])
        suc_points_c[i].probs = py_to_c_probs(suc_points_py[i]['in_probs'])
    return suc_points_c

cdef Point get_point(Point** suc_points_c, int* point_it, int no_sucs):
    cdef Point new_point = Point(0,0,NULL)
    cdef int i = 0
    for i in range(no_sucs):
        new_point = add_points_c(new_point,suc_points_c[i][point_it[i]])
    return new_point

cdef Point add_points_c(Point p1, Point p2):
    cdef int tot_states = p1.no_states + p2.no_states
    cdef int tot_probs = p1.no_probs + p2.no_probs
    cdef Point new_point = Point(tot_states,
                                 tot_probs,
                                 NULL)
    cdef StateProb* new_probs = <StateProb*> malloc(tot_probs * sizeof(StateProb))
    #print("allocated " + str(tot_probs) + "* StateProb to memory-loc: {0:x}".format(<unsigned int>new_probs))

    memcpy(new_probs, p1.probs, p1.no_probs * sizeof(StateProb))
    memcpy(new_probs + p1.no_probs, p2.probs, p2.no_probs * sizeof(StateProb))
    new_point.probs = new_probs
    return new_point


cdef void increase_iterator(int* points_iterator, int [:] points_per_suc, int no_sucs):
    if no_sucs == 0:
        return
    cdef int i = no_sucs -1
    while True:
        if points_iterator[i] == points_per_suc[i]:
            points_iterator[i] = 0
            if i == 0:
                break
            i = i-1
        else:
            points_iterator[i] = points_iterator[i] + 1
            break

cdef double state_prob(Point p, int idx):
    cdef int i = 0
    for i in range(p.no_probs):
        if p.probs[i].state_idx == idx:
            return p.probs[i].prob
    return -1

cdef double[:,:] comp_subsys_edges(double[:,:] P,int [:] states, int no_states):
    cdef double[:,:] subsys_memview = np.zeros((no_states,no_states),dtype='d')
    cdef int i,j = 0
    for i in range(no_states):
        for j in range(no_states):
            subsys_memview[i,j] = P[states[i],states[j]]
    return subsys_memview

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


def partition_graph(G,part_id,suc_ps,labeling,interface,rf):
    in_partition = np.array([((labeling[v] == part_id)
                              or ((labeling[v] in suc_ps)
                                  and interface[v])) for v in G.get_vertices()])

    # in_partition = lambda v: ((labeling[v] == part_id)
    #                           or ((labeling[v] in suc_ps)
    #                               and interface[v]))

    part_view = gt.GraphView(G, vfilt = in_partition)

    is_in = part_view.new_vp("bool")
    is_out = part_view.new_vp("bool")
    is_target = part_view.new_vp("bool")

    has_out_suc = lambda v: len([i for i in part_view.get_out_neighbours(v)
                                 if (interface[i] and
                                     (labeling[i] != part_id))]) > 0

    for v in part_view.get_vertices():
        if (interface[v] and (labeling[v] == part_id)) or (rf.initial == v):
            is_in[v] = True
        if interface[v] and (labeling[v] != part_id):
            is_out[v] = True
        if (rf.to_target[v] > 0) or has_out_suc(v):
            is_target[v] = True
    part_view.vertex_properties["is_in"] = is_in
    part_view.vertex_properties["is_out"] = is_out
    part_view.vertex_properties["is_target"] = is_target

    return part_view


def min_witnesses_from_tree_decomp(rf,partition,thr,known_upper_bound=None,timeout=None):
    start_time = time.perf_counter()
    cdef Point** suc_points_c = NULL
    cdef int no_new_points = 0
    cdef int i,j = 0
    cdef int k = 0
    cdef StateProb* probs = NULL
    cdef SortedInputPoints* subsys_points = NULL
    cdef SortedInputPoints* subsys_points_it = NULL

    # TODO when does the overhead of keeping 'dense' P in memory become visible?
    cdef double[:,:] P_memview = rf.P.todense()

    if known_upper_bound == None:
        known_upper_bound = rf.P.shape[0]

    G = underlying_graph_graphtool(rf.P)

    #gt.graph_draw(G, output="input.pdf",output_size=(1500,1500),vertex_text=G.vertex_index,vertex_size=10)

    dist_from_init = gt.shortest_distance(G,source=G.vertex(rf.initial))
    print(dist_from_init.a)

    edge_probs = G.new_ep("double")
    for e in G.edges():
        edge_probs[e] = rf.P[e.source(),e.target()]
    G.edge_properties["edge_probs"] = edge_probs

    decomp, labeling, interface = quotient(G,partition)

    ## hack to get better partition for brp

    new_partition = []
    for p in decomp.get_vertices():
        p_sucs = decomp.get_out_neighbors(decomp.vertex(p))
        new_p_states = [q for q in G.get_vertices()
                        if (labeling[q] == p) and (not interface[q]) or (labeling[q] in p_sucs and interface[p])]
        new_partition.append(new_p_states)

    decomp, labeling, interface = quotient(G,new_partition)
    ##

    #gt.graph_draw(G, output="input.pdf",output_size=(1500,1500),vertex_text=G.vertex_index,vertex_fill_color=dist_from_init.a,edge_text=edge_probs,vertex_size=10)

    #assert(False)

    # compute an order by which to process the quotient
    rev_top_order = np.flipud(gt.topological_sort(decomp))

    # the "solved" partitions have an entry in the following dictionary, which includes a set of relevant points
    partition_points = dict()

    for part_id in rev_top_order:
        suc_ps = decomp.get_out_neighbors(decomp.vertex(part_id))

        part_view = partition_graph(G,part_id,suc_ps,labeling,interface,rf)

        is_in = part_view.vp["is_in"]
        is_out = part_view.vp["is_out"]
        is_target = part_view.vp["is_target"]

        # collect the points from successors in quotient graph
        suc_points = []
        no_sucs = len(suc_ps)
        points_per_suc = np.zeros(no_sucs,dtype=np.dtype("i"))
        tot_no_points = 1
        if no_sucs > 0:
            suc_points_c = <Point**> malloc(sizeof(Point*) * (no_sucs))
            for i in range(no_sucs):
                suc_points.append(partition_points[suc_ps[i]])
                suc_points_c[i] = py_to_c_points(partition_points[suc_ps[i]])
                points_per_suc[i] = len(partition_points[suc_ps[i]])
                tot_no_points = tot_no_points * points_per_suc[i]

        # if there is no such successor, add the singleton list containing the 'zero point'
        elif no_sucs == 0:
            suc_points_c = <Point**> malloc(sizeof(Point*))
            suc_points_c[0] = py_to_c_points([dict({ 'states' : 0, 'in_probs' : dict() })])

        #list containing the points of p
        partition_points[part_id] = []

        # convex hulls used to remove dominated points
        p_hulls = dict([])

        # map from partition state indices to [0,..,input_dimension-1]
        part_to_input_mapping = bidict()
        inp_dim = 0
        for i in part_view.get_vertices():
            if is_in[i]:
                part_to_input_mapping[i] = inp_dim
                inp_dim += 1

        k_points = dict()

        subsys_points = <SortedInputPoints*> malloc (sizeof(SortedInputPoints))
        subsys_points.size = 0
        subsys_points.no_points = 0
        subsys_points.points = NULL
        subsys_points.next_entry = NULL

        # enumerate relevant subsystems of part_view
        bdd, p_expr = compute_subsys_bdd(part_view)
        for model in bdd.pick_iter(p_expr):
            if timeout != None:
                if time.perf_counter() - start_time > timeout:
                    return -1
            # states in the subsystem
            states = np.array([ i for i in part_view.get_vertices()
                                if model['x{i}'.format(i=i)] == True ],
                              dtype="i")
            no_states = len(states)

            subsys_edges = comp_subsys_edges(P_memview,states,no_states)

            to_target = ((np.array(rf.to_target)).ravel())

            # compute all the new points generated by the subsystem (stored into subsys_points)
            no_new_points = handle_subsys(states,
                                          suc_points_c,
                                          to_target,
                                          is_in.a,
                                          is_out.a,
                                          is_target.a,
                                          subsys_edges,
                                          inp_dim,
                                          known_upper_bound,
                                          thr,
                                          points_per_suc,
                                          no_sucs,
                                          tot_no_points,
                                          subsys_points,
                                          dist_from_init.a,
                                          np.array(part_to_input_mapping,dtype='i'))

            #print("no_new_points :" + str(no_new_points))
            #subsys_points_it = subsys_points
            # print("iterating through subsystem points")
            # while subsys_points_it != NULL:
            #     print("size")
            #     print (subsys_points_it.size)
            #     print("no_points")
            #     print (subsys_points_it.no_points)
            #     for i in range(subsys_points_it.no_points):
            #         print("a point:")
            #         for j in range(inp_dim):
            #             print(subsys_points_it.points[i][j])
            #     subsys_points_it = subsys_points_it.next_entry

        if no_sucs == 0:
            free_mem(suc_points_c[0],1)
            free(suc_points_c)
        else:
            for i in range(no_sucs):
                free_mem(suc_points_c[i],points_per_suc[i])
                free(suc_points_c)

        # lambda to go from inp point to python Point dict
        fip = lambda p,k: from_inp_point(p,part_to_input_mapping,inp_dim,k)

        if inp_dim == 1:
            current_max = 0
            subsys_points_it = subsys_points
            while subsys_points_it != NULL:
                best_k = 0
                #k_points[subsys_points_it.size] = subsys_points_it.points
                for p_idx in range(subsys_points_it.no_points):
                    best_k = max(best_k,subsys_points_it.points[p_idx][0])
                    free(subsys_points_it.points[p_idx])
                partition_points[part_id].extend([fip(np.array([best_k]),subsys_points_it.size)])
                free(subsys_points_it.points)
                tmp = subsys_points_it
                subsys_points_it = subsys_points_it.next_entry
                free(tmp)

        else:
            # compute convex-hulls for all k (iteratively from lowest)
            # and keep points that are vertices in the respective hulls
            k_vertices = dict()
            conv_hull = None

            #print(k_points)

            subsys_points_it = subsys_points
            while subsys_points_it != NULL:
                k = subsys_points_it.size
                for p_idx in range(subsys_points_it.no_points):
                    if conv_hull == None:
                        # todo: figure out how to handle degenerate cases!
                        conv_hull = ConvexHull(np.append(points_to_add_in_inp_space(subsys_points_it.points[p_idx],inp_dim),
                                                         np.array([np.zeros(inp_dim)]),
                                                         axis=0),
                                               incremental=True,qhull_options='QJ1e-12 Q12 Pp')
                    else:
                        conv_hull.add_points(points_to_add_in_inp_space(subsys_points_it.points[p_idx],inp_dim))
                if(subsys_points_it.no_points > 0):
                    k_vertices[k] = np.reshape(conv_hull.vertices,(conv_hull.vertices.shape[0],1))
                    # print("convex-hull <=%d points:" % k)
                    # print(conv_hull.points)
                    # print("convex-hull <=%d vertices:" % k)
                    # print(np.take_along_axis(conv_hull.points,k_vertices[k],0))

                subsys_points_it = subsys_points_it.next_entry

            conv_hull.close()

            # keep k-points that are real points from the current partitions (not projections)
            # and vertices that are not vertices of convex hulls for any k' < k
            sofar = np.zeros(shape=(1,inp_dim))
            sofar_cnt = 0
            for k in sorted(k_vertices.keys()):

                vertex_cnt = 0
                k_vertex_points = np.take_along_axis(conv_hull.points,k_vertices[k],0)
                vertex_cnt = len(k_vertex_points)
                subsys_points_it = subsys_points
                while subsys_points_it != NULL:
                    if subsys_points_it.size == k:
                        break
                    subsys_points_it = subsys_points_it.next_entry

                if subsys_points_it == NULL:
                    assert(False)

                k_points_k_cnt = subsys_points_it.no_points
                tmp_list = []

                for i in range(vertex_cnt):
                    if (arreqclose_in_list2_carr(k_vertex_points[i],subsys_points_it.points,inp_dim,k_points_k_cnt,1e-12) and
                        not arreqclose_in_list2(k_vertex_points[i],sofar,inp_dim,sofar_cnt+1,1e-12)):
                        sofar = np.append(sofar,[k_vertex_points[i]],axis=0)
                        sofar_cnt += 1
                        tmp_list.append(k_vertex_points[i])

                # print("**k_vertex_points**")
                # print(k_vertex_points)
                # print("**k_points[k]**")
                # print(subsys_points_it.points)
                # print("k_points computed")
                # print(np.array(tmp_list))

                #k_points[k] = np.array(tmp_list)

                partition_points[part_id].extend([fip(p,k) for p in tmp_list])

            subsys_points_it = subsys_points
            while subsys_points_it != NULL:
                free(subsys_points_it.points)
                tmp = subsys_points_it
                subsys_points_it = subsys_points_it.next_entry
                free(tmp)

        print("\npartition " + str(part_id) + " points: \n" + str(partition_points[part_id]))
        print("partition " + str(part_id) + " no-points: \n" + str(len(partition_points[part_id])) + "\n")

    return min([p["states"] for p in partition_points[rev_top_order[-1]]])

# from https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays
                 if elem.size == myarr.size and np.allclose(elem, myarr,atol=1e-10)), False)

# reimplement this using some sound is-close-check
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint arreqclose_in_list2(double[:] myarr,
                              double[:,:] list_arrays,
                              int dim,
                              int list_size,
                              double tol):
    cdef bint is_eq = True
    cdef int i,j = 0
    for j in range(list_size):
        for i in range(dim):
            if not (((myarr[i] - list_arrays[j][i]) < tol) and ((list_arrays[j][i] - myarr[i]) < tol)):
                is_eq = False
                break
        if is_eq:
            return True
        else:
            is_eq = True
    return False

# reimplement this using some sound is-close-check
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint arreqclose_in_list2_carr(double[:] myarr,
                                   double** list_arrays,
                                   int dim,
                                   int list_size,
                                   double tol):
    cdef bint is_eq = True
    cdef int i,j = 0
    for j in range(list_size):
        for i in range(dim):
            if not (((myarr[i] - list_arrays[j][i]) < tol) and ((list_arrays[j][i] - myarr[i]) < tol)):
                is_eq = False
                break
        if is_eq:
            return True
        else:
            is_eq = True
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

# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef int handle_subsys(int[:] states,
                       Point** suc_points_c,
                       double[:] to_target,
                       uint8[:] is_in,
                       uint8[:] is_out,
                       uint8[:] is_target,
                       double[:,:] subsys_edges,
                       int inp_dim,
                       int known_upper_bound,
                       float thr,
                       int[:] points_per_suc,
                       int no_sucs,
                       int tot_no_points,
                       SortedInputPoints* subsys_points,
                       int[:] dist_from_init,
                       int[:] inp_to_part):
    cdef int found = 0
    cdef int no_states = states.shape[0]
    cdef int no_not_out_states = 0
    cdef int i = 0
    cdef int j = 0
    cdef int state_idx = 0

    for i in range(no_states):
        if not is_out[states[i]]:
            no_not_out_states = no_not_out_states +1

    cdef int * points_iterator = NULL
    if no_sucs > 0:
        points_iterator = <int* > malloc(no_sucs * sizeof(int))
    else:
        points_iterator = <int* > malloc(sizeof(int))
    cdef int [:] points_per_suc_view = points_per_suc
    cdef int no_new_points = 0
    cdef int tot_no_states = 0
    cdef int min_dist_from_init = 0
    cdef int excl_by_states = 0
    cdef int excl_by_thr = 0
    cdef double sum_inp_reach = 0
    cdef double p = 0

    cdef double* reach_probs = <double*> malloc(no_states * sizeof(double))
    cdef SortedInputPoints* subsys_points_it = NULL
    cdef SortedInputPoints* subsys_points_new = NULL

    ## stuff needed for compute_reach
    cdef double* P_arr = <double*> malloc(no_states * no_states * sizeof(double))
    cdef double* target_arr = <double*> malloc(no_states * sizeof(double))
    # this array is needed for LAPACK call
    cdef int* IPIV_arr = <int*> malloc(no_states * sizeof(int))

    cdef int ALLOC_BLOCK_SIZE = 10

    if no_sucs > 0:
        for i in range(no_sucs):
            points_iterator[i] = 0
    else:
        points_iterator[0] = 1

    cdef Point current_point = Point(0,0,NULL)

    for i in range(tot_no_points):
        current_point = get_point(suc_points_c,points_iterator,no_sucs)
        increase_iterator(points_iterator, points_per_suc_view, no_sucs)
        tot_no_states = no_not_out_states + current_point.no_states

        # exclude point if it (+ its minimal distance to the initial state) already has more states than known upper-bound
        # TODO sensible value for min_dist_from_init
        min_dist_from_init = INT_MAX
        for j in range(no_states):
            if is_in[states[j]]:
                min_dist_from_init = min(min_dist_from_init,dist_from_init[states[j]])

        # print("min_dist_from_init:" + str(min_dist_from_init))
        # print("tot_no_states:" + str(tot_no_states))
        # print("known_upper_bound:" + str(known_upper_bound))

        if (tot_no_states + min_dist_from_init) > known_upper_bound:
            excl_by_states = excl_by_states + 1
            continue

        # define target_arr (RHS of the linear equation system to solve) using out-probabilities
        # given by current_point
        for j in range(no_states):
            state_idx = states[j]
            if is_out[state_idx]:
                p = state_prob(current_point,state_idx)
                if p >= 0:
                    target_arr[j] = p
                else:
                    target_arr[j] = 0
            elif is_target[state_idx]:
                target_arr[j] = to_target[state_idx]
            else:
                target_arr[j] = 0

        if current_point.probs != NULL:
            free(current_point.probs)

        compute_reach(no_states,subsys_edges,target_arr,P_arr,reach_probs,IPIV_arr)

        # exclude point if the sum of probabilities of the inits is less than the threshold
        # TODO maybe replace by more elaborate checks
        # e.g. including *all* predecessor states the probability is too low
        sum_inp_reach = 0
        for j in range(no_states):
            if is_in[states[j]]:
                sum_inp_reach += reach_probs[j]

        if(sum_inp_reach < thr):
            excl_by_thr = excl_by_thr + 1
            continue

        subsys_points_it = subsys_points
        while True:
            if subsys_points_it.next_entry == NULL:
                subsys_points_new = <SortedInputPoints*> malloc(sizeof(SortedInputPoints))
                subsys_points_new.size = tot_no_states
                subsys_points_new.no_points = 0
                subsys_points_new.next_entry = NULL
                subsys_points_it.next_entry = subsys_points_new
            elif subsys_points_it.size > tot_no_states:
                subsys_points_new = <SortedInputPoints*> malloc(sizeof(SortedInputPoints))
                subsys_points_new.size = tot_no_states
                subsys_points_new.no_points = 0
                subsys_points_new.next_entry = subsys_points_it.next_entry
                subsys_points = subsys_points_new
            elif subsys_points_it.size == tot_no_states:
                subsys_points_new = subsys_points_it
            elif subsys_points_it.next_entry.size > tot_no_states:
                subsys_points_new = <SortedInputPoints*> malloc(sizeof(SortedInputPoints))
                subsys_points_new.size = tot_no_states
                subsys_points_new.no_points = 0
                subsys_points_new.next_entry = subsys_points_it.next_entry
                subsys_points_it.next_entry = subsys_points_new
            else:
                subsys_points_it = subsys_points_it.next_entry
                continue
            no_points = subsys_points_new.no_points + 1
            subsys_points_new.no_points = no_points
            if ((no_points - 1) % ALLOC_BLOCK_SIZE) == 0:
                if (no_points - 1) == 0:
                    subsys_points_new.points = <double**> malloc(sizeof(double*) * ((no_points-1) + ALLOC_BLOCK_SIZE))
                else:
                    subsys_points_new.points = <double**> realloc(subsys_points_new.points,sizeof(double*) * ((no_points-1) + ALLOC_BLOCK_SIZE))
            subsys_points_new.points[no_points-1] = <double*> malloc(sizeof(double) * inp_dim)
            for j in range(inp_dim):
                found = 0
                for i in range(no_states):
                    if inp_to_part[j] == states[i]:
                        found = 1
                        subsys_points_new.points[no_points-1][j] = reach_probs[i]
                if found == 0:
                    subsys_points_new.points[no_points-1][j] = 0
            no_new_points += 1
            break

    free(points_iterator)
    free(P_arr)
    free(IPIV_arr)
    free(target_arr)

    return no_new_points

# def to_inp_point(point,part_to_input_mapping,inp_dim):
#     inp_point = np.zeros(inp_dim)
#     for i in range(inp_dim):
#         if part_to_input_mapping.inverse[i] in point['in_probs']:
#             inp_point[i] = point['in_probs'][part_to_input_mapping.inverse[i]]
#         else:
#             inp_point[i] = 0
#     return inp_point

def from_inp_point(point,part_to_input_mapping,inp_dim,k):
    p = dict()
    p["states"] = k
    p["in_probs"] = dict()
    for i in range(inp_dim):
        p["in_probs"][part_to_input_mapping.inverse[i]] = point[i]
    return p

cdef points_to_add_in_inp_space(double* new_point,
                                int inp_dim):
    projections = []

    for sub_dims in powerset(range(inp_dim)):
        if len(sub_dims) == 0:
            continue
        proj = np.zeros(inp_dim)
        for i in sub_dims:
            proj[i] = new_point[i]
        projections.append(proj)

    return np.array(projections)


# computes the reachability probabilities of input states of a partition
# for given probabilties of the output states (in target_arr)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int compute_reach(int no_states,
                       double[:,:] P,
                       double* target_arr,
                       double* P_arr,
                       double* reach_probs,
                       int* IPIV_arr):
    cdef int i,j = 0
    # will contain the status of computation
    cdef int INFO = 0
    cdef int NO_COLS = 1

    for i in range(no_states):
        for j in range(no_states):
            if i == j:
                P_arr[no_states * i + i] = 1 - P[i][i]
            else:
                P_arr[no_states * i + j] = - P[j][i]

    # solve equation system by call to lapack routine dgesv
    dgesv(&no_states,&NO_COLS,P_arr,&no_states,IPIV_arr,target_arr,&no_states,&INFO)

    for i in range(no_states):
        target_arr[i]
        reach_probs[i] = target_arr[i]

    return INFO



def print_conv_hull(conv_hull,k):
    pp = PdfPages(str(k) + '_conv_hull.pdf')
    plt.plot(conv_hull.points[:,0], conv_hull.points[:,1], 'o')

    for simplex in conv_hull.simplices:
        plt.plot(conv_hull.points[simplex, 0], conv_hull.points[simplex, 1], 'k-')

    pp.savefig()
    pp.close()



####  ----  Below is old stuff  ----
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
