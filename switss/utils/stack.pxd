from .treaps cimport TNode

cdef struct IntStack:
    int element
    IntStack* btm

cdef IntStack* push(IntStack* stack, int element)

cdef (IntStack*, int) pop(IntStack* stack)

cdef void freestack(IntStack* stack)

cdef struct TreapStack:
    TNode* element
    TreapStack* btm

cdef TreapStack* ts_push(TreapStack* tstack, TNode* element)

cdef (TreapStack*, TNode*) ts_pop(TreapStack* tstack)

cdef void free_tstack(TreapStack* tstack)

