#cython: language_level=3

from libc.stdlib cimport malloc, free
import numpy as np
from scipy.sparse import dok_matrix
from bidict import bidict

cdef struct Stack:
    int element
    Stack* btm

cdef Stack* push(Stack* stack, int element):
    cdef Stack *snew = <Stack *> malloc(sizeof(Stack))
    snew[0].element = element
    snew[0].btm = stack
    return snew

cdef (Stack*, int) pop(Stack* stack):
    cdef Stack tmp = stack[0]
    free(stack)
    return tmp.btm, tmp.element

cdef void freestack(Stack* stack):
    cdef Stack *current = stack
    cdef Stack *tmp = NULL
    while current != NULL:
        tmp = current[0].btm
        free(current)
        current = tmp

cdef struct Node:
    int *predecessors
    int *successors
    int predcount, succcount

cdef class Graph:
    cdef Node *nodes
    cdef int nodecount

    def __cinit__(self, P, index_by_state_action):
        self.nodecount = P.shape[1]
        self.nodes = <Node *> malloc(self.nodecount * sizeof(Node))

        for i in range(self.nodecount):
            self.nodes[i] = Node()
        
        for (i,d),p in P.items():
            s,_ = index_by_state_action.inv[i]
            self.add_successor(s,d)

    cdef void add_successor(self, int fromidx, int toidx):
        cdef Node* fromnode = &self.nodes[fromidx]
        cdef Node* tonode = &self.nodes[toidx]
        
        fromnode[0].succcount += 1
        cdef int *newsuccs = <int *> malloc(fromnode[0].succcount * sizeof(int))
        for i in range(fromnode[0].succcount-1):
            newsuccs[i] = fromnode[0].successors[i]
        newsuccs[fromnode[0].succcount-1] = toidx
        free(fromnode[0].successors)
        fromnode[0].successors = newsuccs

        tonode[0].predcount += 1
        cdef int *newpreds = <int *> malloc(tonode[0].predcount * sizeof(int))
        for i in range(tonode[0].predcount-1):
            newpreds[i] = tonode[0].predecessors[i]
        newpreds[tonode[0].predcount-1] = fromidx
        free(tonode[0].predecessors)
        tonode[0].predecessors = newpreds

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

    def reachable(self, fromset, direction, blocklist=set()):
        assert len(fromset) > 0
        assert direction in ["forward", "backward"]

        cdef int* reachablemask = <int *> malloc(self.nodecount * sizeof(int))
        cdef int* instack = <int *> malloc(self.nodecount * sizeof(int))
        cdef Stack *stack = NULL
        cdef int currentidx
        cdef Node *currentnode
        cdef int neighbourcount
        cdef int* neighbours
        cdef int* blockmask = <int *> malloc(self.nodecount * sizeof(int))

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
                if direction == "forward":
                    neighbourcount = currentnode[0].succcount
                    neighbours = currentnode[0].successors
                else:
                    neighbourcount = currentnode[0].predcount
                    neighbours = currentnode[0].predecessors

                for idx in range(neighbourcount):
                    neighidx = neighbours[idx]
                    if not reachablemask[neighidx] and not instack[neighidx]:
                        instack[neighidx] = 1
                        stack = push(stack, neighidx)

        ret = np.zeros(self.nodecount, dtype=np.bool)
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



