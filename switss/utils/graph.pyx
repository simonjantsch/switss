#cython: language_level=3

from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
import numpy as np
from scipy.sparse import dok_matrix
from bidict import bidict

from .treaps cimport TNode, in_treap, free_treap, treap_to_arr, treap_from_arr, add_to_treap
from .stack cimport IntStack, TreapStack, push, pop, freestack, ts_push, ts_pop, free_tstack

ctypedef (int,int,float) SAPPair

cdef struct Node:
    SAPPair *predecessors
    SAPPair *successors
    int predcount, succcount 

cdef struct TarjanNode:
    int index, lowlink
    int onstack

cdef struct SubMDP:
    int* E
    SAPPair* F
    int nodecount, sapcount

cdef class Graph:
    cdef Node *nodes
    cdef int nodecount

    def __cinit__(self, P=None, index_by_state_action=None):
        if P is not None and index_by_state_action is not None:
            self.nodecount = P.shape[1]
            self.nodes = <Node *> malloc(self.nodecount * sizeof(Node))
            for i in range(self.nodecount):
                self.nodes[i] = Node(NULL,NULL,0,0)
            for (i,d),p in P.items():
                s,a = index_by_state_action.inv[i]
                self.add_successor(s,a,p,d)

    def get_nodecount(self):
        return self.nodecount

    cdef void add_successor(self, int fromidx, int action, float prob, int toidx):
        cdef Node* fromnode = &self.nodes[fromidx]
        cdef Node* tonode = &self.nodes[toidx]

        fromnode[0].succcount += 1
        cdef SAPPair *newsuccs = <SAPPair *> malloc(fromnode[0].succcount * sizeof(SAPPair))
        for i in range(fromnode[0].succcount-1):
            newsuccs[i] = fromnode[0].successors[i]
        newsuccs[fromnode[0].succcount-1] = (toidx, action, prob)
        free(fromnode[0].successors)
        fromnode[0].successors = newsuccs

        tonode[0].predcount += 1
        cdef SAPPair *newpreds = <SAPPair *> malloc(tonode[0].predcount * sizeof(SAPPair))
        for i in range(tonode[0].predcount-1):
            newpreds[i] = tonode[0].predecessors[i]
        newpreds[tonode[0].predcount-1] = (fromidx, action, prob)
        free(tonode[0].predecessors)
        tonode[0].predecessors = newpreds

    def successors(self, nodeidx, actionidx=None):
        cdef Node *node = &self.nodes[nodeidx]
        for i in range(node[0].succcount):
            if actionidx is None or actionidx == node[0].successors[i][1]:
                yield node[0].successors[i]
    
    def predecessors(self, nodeidx, actionidx=None):
        cdef Node *node = &self.nodes[nodeidx]
        for i in range(node[0].predcount):
            if actionidx is None or actionidx == node[0].predecessors[i][1]:
                yield node[0].predecessors[i]

    def __str__(self):
        cdef Node *currentnode
        ret = ""
        for i in range(self.nodecount):
            currentnode = &self.nodes[i]
            ret += str(i) + " ->"
            for j in range(currentnode[0].succcount):
                ret += " " + str(currentnode[0].successors[j])
            ret += "\n"
        return ret

    cdef (int,int,IntStack*) strongconnect(self, int v, IntStack* stack, TarjanNode* tnodes,
                                           int i, int* sccs, int scc_counter, TNode* treap):

        tnodes[v] = TarjanNode(i, i, 1)
        i += 1
        stack = push(stack, v)

        cdef Node *node = &self.nodes[v]
        for succidx in range(node[0].succcount):
            w = node[0].successors[succidx][0] # -> index of successor state

            if not in_treap(treap,w):
                continue

            if tnodes[w].index == -1:
                i,scc_counter,stack = self.strongconnect(w,stack,tnodes,i,sccs,scc_counter,treap)
                tnodes[v].lowlink = min(tnodes[v].lowlink,tnodes[w].lowlink)
            elif tnodes[w].onstack:
                tnodes[v].lowlink = min(tnodes[v].lowlink,tnodes[w].index)


        if tnodes[v].lowlink == tnodes[v].index:
            w = -1
            while w != v:
                stack, w = pop(stack)
                tnodes[w].onstack = 0
                sccs[w] = scc_counter
            scc_counter += 1

        return i,scc_counter,stack


    cdef (int) strongly_connected_components(self, TNode* treap, int* sccs):

        cdef int subg_nodescount = treap.size
        cdef int* subg_arr = <int *> malloc(subg_nodescount * sizeof(int))
        treap_to_arr(treap,subg_arr)
        cdef int j = 0

        # Implementation of Tarjan's Algorithm (possibly on a subgraph defined by the mask 'nmask')
        # initialize vector containing strongly connected endcomponents
        j = 0
        while j < self.nodecount:
            sccs[j] = 0
            j += 1

        cdef int scc_counter = 1
        cdef int i = 0

        # initialize a vector containing meta-data for each node (including nodes not in subgraph, for simplicity)
        cdef TarjanNode* tnodes = <TarjanNode *> malloc(self.nodecount * sizeof(TarjanNode))

        j = 0
        while j < subg_nodescount:
            tnodes[subg_arr[j]].index = -1
            j += 1

        cdef IntStack *stack = NULL

        j = 0
        while j < subg_nodescount:
            if tnodes[subg_arr[j]].index == -1:
                i,scc_counter,stack = self.strongconnect(subg_arr[j], stack, tnodes, i, sccs, scc_counter, treap)
            j += 1

        
        # clear everything up
        free(tnodes)
        free(subg_arr)
        freestack(stack)

        return scc_counter-1

    def maximal_end_components(self):
        ret = np.zeros(self.nodecount)
        cdef int compcount = 0
        cdef int mec_counter = 1

        cdef Node *node = NULL
        cdef int* sccs = <int *> malloc(self.nodecount * sizeof(int))

        # initialize first subgraph by adding all nodes
        cdef TNode* subg_treap = NULL
        cdef int subg_nodecount = 0
        cdef int* subg_arr = NULL
        cdef int i = 0
        cdef int j = 0
        while i < self.nodecount:
            subg_treap = add_to_treap(subg_treap,i)
            i += 1

        # stack contains subgraphs which are represented by a set of nodes using treaps
        cdef TreapStack* treap_stack = NULL
        treap_stack = ts_push(treap_stack, subg_treap)

        cdef TNode** new_subg_treaps = NULL

        while treap_stack != NULL:
            treap_stack, subg_treap = ts_pop(treap_stack)
            j = 0
            subg_nodecount = subg_treap.size
            subg_arr = <int*> malloc(subg_nodecount * sizeof(int))
            treap_to_arr(subg_treap,subg_arr)
            
            compcount = self.strongly_connected_components(subg_treap,sccs)

            if compcount == 1:
                # make sure that every node has at least one outgoing edge (one action that can be enabled for states in the MDP)
                if subg_nodecount == 1:
                    ignore_this_graph = True
                    node = &self.nodes[subg_arr[0]]
                    for succidx in range(node[0].succcount):
                        w = node[0].successors[succidx][0] # -> index of successor state
                        if in_treap(subg_treap,w):
                            ignore_this_graph = False
                            break

                    if ignore_this_graph: 
                        free_treap(subg_treap)
                        continue

                j = 0
                while j < subg_nodecount:
                    ret[subg_arr[j]] = mec_counter
                    j += 1
                mec_counter += 1
                free_treap(subg_treap)
            else:
                new_subg_treaps = <TNode**> malloc(compcount * sizeof(TNode*))
                j = 0
                while j < compcount:
                    new_subg_treaps[j] = NULL
                    j += 1

                j = 0
                while j < subg_nodecount:
                    i = sccs[subg_arr[j]]-1
                    new_subg_treaps[i] = add_to_treap(new_subg_treaps[i],subg_arr[j])
                    j += 1

                j = 0
                while j < compcount:
                    treap_stack = ts_push(treap_stack,new_subg_treaps[j])
                    j += 1

                free_treap(subg_treap)

        free(subg_arr)
        free_tstack(treap_stack)
        free(sccs)

        return ret, mec_counter-1


    def reachable(self, fromset, direction, blocklist=set()):
        assert len(fromset) > 0
        assert direction in ["forward", "backward"]

        cdef int* reachablemask = <int *> malloc(self.nodecount * sizeof(int))
        cdef int* instack = <int *> malloc(self.nodecount * sizeof(int))
        cdef IntStack *stack = NULL
        cdef int currentidx
        cdef Node *currentnode
        cdef int neighbourcount
        cdef SAPPair* neighbours
        cdef int* blockmask = <int *> malloc(self.nodecount * sizeof(int))
        cdef int j = 0

        cdef int direction_int = 0
        if direction == "forward":
            direction_int = 1


        # setup reachable mask, blockmask and stack
        for i in range(self.nodecount):
            reachablemask[i] = 0
            instack[i] = 0
            blockmask[i] = 0
        
        for idx in fromset:
            instack[idx] = 1
            stack = push(stack, idx)

        for idx in blocklist:
            blockmask[idx] = 1

        while stack != NULL:
            stack, currentidx = pop(stack)
            currentnode = &self.nodes[currentidx]
            instack[currentidx] = 0
            reachablemask[currentidx] = 1

            if not blockmask[currentidx]:
                if direction_int == 1: # direction == "forward"
                    neighbourcount = currentnode[0].succcount
                    neighbours = currentnode[0].successors
                else:
                    neighbourcount = currentnode[0].predcount
                    neighbours = currentnode[0].predecessors

                j = 0
                while j < neighbourcount:
                    neighidx,_,_ = neighbours[j]
                    if not reachablemask[neighidx] and not instack[neighidx]:
                        instack[neighidx] = 1
                        stack = push(stack, neighidx)
                    j += 1

        ret = np.zeros(self.nodecount, dtype=int)
        for i in range(self.nodecount):
            ret[i] = reachablemask[i]

        free(blockmask)
        free(reachablemask)
        free(instack)
        freestack(stack)
        return ret
    
    cdef void __free_nodes(self):
        for i in range(self.nodecount):
            free(self.nodes[i].predecessors)
            free(self.nodes[i].successors)
        free(self.nodes)

    def __dealloc__(self):
        self.__free_nodes()



