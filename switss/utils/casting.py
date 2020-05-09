import numpy as np
from scipy.sparse import dok_matrix, spmatrix

def cast_dok_matrix(obj):
    """Casts a 1d or 2d-object as a `scipy.sparse.dok_matrix`. 
    If the input is 1d, it will create a (:math:`N \\times 1`) `dok_matrix`. 
    
    :param arr: Input array/list/sparse matrix.
    :type arr: 1d or 2d-object type
    :return: Resulting dok_matrix.
    :rtype: scipy.sparse.dok_matrix
    """    
    if not isinstance(obj, spmatrix):
        obj = np.array(obj)
        if len(obj.shape) == 1:
            obj = np.array([obj]).T
    
    return dok_matrix(obj)