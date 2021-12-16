from libc.stdlib cimport malloc, free, time

cdef struct TNode:
    int tree_key
    long heap_key
    int treap_size
    TNode* left, right

cdef long new_heap_key():
    return rand()

cdef (int) in_treap(Tnode* treap, int key):
    if treap == NULL:
        return 0
    elif key == treap.tree_key:
        return 1
    elif key < treap.tree_key:
        return in_treap(treap.left, key)
    else:
        return in_treap(treap.right, key)

cdef void split_treap(Tnode* treap, int key, Tnode* left, Tnode* right):
    if treap == NULL:
        left = NULL
        right = NULL
    elif key < treap.tree_key:
        split(treap.left, key, left, treap.left)
        right = treap
    else:
        split(treap.right, key, treap.right, right)
        left = treap

cdef void insert(TNode* treap, TNode new_node):
    if (treap == NULL):
        treap = new_node
        treap.size = 1

    elif treap.tree_key == new_node.tree_key:
        return

    elif treap.heap_key > new_node.heap_key:
        if treap.tree_key > new_node.tree_key:
            insert(treap.left, new_node)
        else:
            insert(treap.right, new_node)
        treap.size += 1

    else:
        split(treap, new_node.tree_key, new_node.left, new_node.right)
        new_node.size = treap.size + 1
        treap = new_node

cdef void treap_to_arr(TNode* treap, int* arr):
    cdef int idx = 0
    fill_array(treap, arr, idx)

cdef int fill_array(Tnode* treap, int* arr, int idx):
    if treap == NULL:
        return idx
    arr[idx] = treap.tree_key
    idx += 1
    idx = fill_array(treap.left, arr, idx)
    idx = fill_array(treap.right, arr, idx)
    return idx
