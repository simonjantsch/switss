#cython: language_level=3

cdef struct TNode:
    int tree_key
    long heap_key
    int size
    TNode* left
    TNode* right

cdef (int) in_treap(TNode* treap, int key)

cdef TNode* add_to_treap(TNode* treap, int new_key)

cdef void free_treap(TNode* treap)

cdef TNode* treap_from_arr(int* arr, int arr_size)

cdef void treap_to_arr(TNode* treap, int* arr)




