#cython: language_level=3

from libc.stdlib cimport malloc, free, rand, srand

import numpy as np
import random

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

    elif key == treap.tree_key:
        return treap.left, treap.right

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

        update_size_from_successors(treap)
        return treap

    else:
        new_left, new_right = split_treap(treap, new_node.tree_key)
        new_node.left = new_left
        new_node.right = new_right
        update_size_from_successors(new_node)
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



def treap_test_specificseed(seed):
    srand(seed)
    cdef TNode* treap = NULL
    cdef int* treap_arr = NULL

    v1 = range(20)
    v2 = [1,2,5,6,8,8,9,7,99,7]
    v3 = range(999)
    v4 = [3*i for i in range(55)]

    for v in [v1,v2,v3,v4]:
        treap = NULL
        for i in v:
            print("adding " + str(i))
            treap = add_to_treap(treap,i)
            print("treap.size: " + str(treap.size))

        assert(treap.size == len(set(v)))

        for i in v:
            assert(in_treap(treap,i) == 1)

        treap_arr = <int*> malloc(treap.size * sizeof(int))
        treap_to_arr(treap,treap_arr)
        nump_arr = np.zeros(treap.size)
        for i in range(treap.size):
            nump_arr[i] = treap_arr[i]

        print(set(v))
        print(set(nump_arr))
        assert(set(v) == set(nump_arr))

        free(treap_arr)
        if treap == NULL:
            print("treap is null")
        free_treap(treap)


def treap_testcases():
    problematic_seeds = [394]
    random_seeds = []
    for i in range(20):
        random_seeds.append(random.randint(0, 1000))

    for seed in problematic_seeds + random_seeds:
        treap_test_specificseed(seed)
