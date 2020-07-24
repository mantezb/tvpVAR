import numpy as np
import scipy.sparse as sps
import numpy.matlib


def repmat(x, r, c):
    '''
    repmat(x, r, c) is equivalent to numpy.matlib.repmat(x, r, c) except that it works correctly for
      sparse matrices.
    '''
    if sps.issparse(x):
        row = sps.hstack([x for _ in range(c)])
        return sps.vstack([row for _ in range(r)], format=x.format)
    else: return np.matlib.repmat(x, r, c)