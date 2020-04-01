import numpy as np
from scipy.sparse import dok_matrix

def array_to_dok_matrix(arr):
    """Casts a 2d `numpy.array` as a `scipy.sparse.dok_matrix`. 
    If the input is a 1d-array (or list), it will create a (Nx1) `dok_matrix`. 
    
    :param arr: The array.
    :type arr: numpy.array or list
    :return: The dok_matrix.
    :rtype: scipy.sparse.dok_matrix
    """    
    arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = np.matrix(arr).T
        
    P = dok_matrix(arr.shape)
    for i,j in np.ndindex(*arr.shape):
        P[i,j] = arr[i,j]
    return P