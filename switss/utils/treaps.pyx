#cython: language_level=3

from libc.stdlib cimport malloc, free, rand

cdef long new_heap_key():
    return rand()

cdef void update_size_from_successors(TNode* node):
    cdef int right_size = 0
    cdef int left_size = 0

    if node == NULL:
        return
    if node.right != NULL:
        right_size = node.right.size
    if node.left != NULL:
        left_size = node.left.size

    node.size = right_size + left_size + 1

cdef (int) in_treap(TNode* treap, int key):
    if treap == NULL:
        return 0
    elif key == treap.tree_key:
        return 1
    elif key < treap.tree_key:
        return in_treap(treap.left, key)
    else:
        return in_treap(treap.right, key)

cdef (TNode*, TNode*) split_treap(TNode* treap, int key):
    if treap == NULL:
        return NULL, NULL
    elif key < treap.tree_key:
        rec_left, rec_right = split_treap(treap.left, key)
        treap.left = rec_right
        update_size_from_successors(treap)
        return rec_left, treap

    else:
        rec_left, rec_right = split_treap(treap.right, key)
        treap.right = rec_left
        update_size_from_successors(treap)
        return treap, rec_right

cdef TNode* insert(TNode* treap, TNode* new_node):
    #assert new_node.left == NULL and new_node.right == NULL and new_node.size == 1
    cdef TNode* new_left
    cdef TNode* new_right

    if treap == NULL:
        return new_node

    elif treap.tree_key == new_node.tree_key:
        return treap

    elif treap.heap_key > new_node.heap_key:
        if treap.tree_key > new_node.tree_key:
            treap.left = insert(treap.left, new_node)
        else:
            treap.right = insert(treap.right, new_node)

        treap.size += 1
        return treap

    else:
        new_left, new_right = split_treap(treap, new_node.tree_key)
        new_node.size = treap.size + 1
        new_node.left = new_left
        new_node.right = new_right
        return new_node

cdef TNode* add_to_treap(TNode* treap, int new_key):
    cdef TNode* new_node = <TNode *> malloc(sizeof(TNode))
    heap_key = new_heap_key()
    new_node.tree_key = new_key
    new_node.heap_key = heap_key
    new_node.size = 1
    new_node.left = NULL
    new_node.right = NULL
    return insert(treap,new_node)

cdef void free_treap(TNode* treap):
    if treap == NULL:
        return
    free_treap(treap.left)
    free_treap(treap.right)
    free(treap)

cdef TNode* treap_from_arr(int* arr, int arr_size):
    cdef TNode* treap = NULL
    cdef int i = 0
    while i < arr_size:
        treap = add_to_treap(treap, arr[i])
    return treap

cdef void treap_to_arr(TNode* treap, int* arr):
    cdef int idx = 0
    fill_array(treap, arr, idx)

cdef int fill_array(TNode* treap, int* arr, int idx):
    if treap == NULL:
        return idx
    arr[idx] = treap.tree_key
    idx += 1
    idx = fill_array(treap.left, arr, idx)
    idx = fill_array(treap.right, arr, idx)
    return idx


def test_treap():
    cdef TNode* treap = NULL

    for i in range(20):
        treap = add_to_treap(treap,i)
        print("treap size:" + str(treap.size))

    print(in_treap(treap,5))
    print(in_treap(treap,7))
    print(in_treap(treap,11))
    print(in_treap(treap,20))
    print(in_treap(treap,31))
    print("========")
    cdef int* arr = <int*> malloc(treap.size * sizeof(int))
    treap_to_arr(treap,arr)
    for i in range(treap.size):
        print(arr[i])
    free(arr)
    free_treap(treap)
