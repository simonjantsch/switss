import numpy as np
from scipy.sparse import dok_matrix

def array_to_dok_matrix(arr):
    arr = np.array(arr)
    P = dok_matrix(arr.shape)
    for i,j in np.ndindex(*arr.shape):
        P[i,j] = arr[i,j]
    return P