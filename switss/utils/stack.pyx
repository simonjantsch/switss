from libc.stdlib cimport malloc, free
from .treaps cimport free_treap, TNode

cdef struct IntStack:
    int element
    IntStack* btm

cdef struct TreapStack:
    TNode* element
    TreapStack* btm

cdef IntStack* push(IntStack* stack, int element):
    cdef IntStack *snew = <IntStack *> malloc(sizeof(IntStack))
    snew[0].element = element
    snew[0].btm = stack
    return snew

cdef (IntStack*, int) pop(IntStack* stack):
    cdef IntStack tmp = stack[0]
    free(stack)
    return tmp.btm, tmp.element

cdef void freestack(IntStack* stack):
    cdef IntStack *current = stack
    cdef IntStack *tmp = NULL
    while current != NULL:
        tmp = current[0].btm
        free(current)
        current = tmp

cdef TreapStack* ts_push(TreapStack* tstack, TNode* element):
    cdef TreapStack *ts_new = <TreapStack *> malloc(sizeof(TreapStack))
    ts_new[0].element = element
    ts_new[0].btm = tstack
    return ts_new

cdef (TreapStack*, TNode*) ts_pop(TreapStack* tstack):
    cdef TreapStack tmp = tstack[0]
    free(tstack)
    return tmp.btm, tmp.element

cdef void free_tstack(TreapStack* tstack):
    cdef TreapStack *current = tstack
    cdef TreapStack *tmp = NULL
    while current != NULL:
        tmp = current[0].btm
        free_treap(current.element)
        free(current)
        current = tmp

