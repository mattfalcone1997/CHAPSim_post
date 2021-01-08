#cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange


def autocov_calc_z( np.ndarray[np.float64_t,ndim=3] fluct1,
                    np.ndarray[np.float64_t,ndim=3] fluct2,
                    np.ndarray[np.float64_t,ndim=3] R_z,
                    np.int32_t max_z_sep):
    
    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t NCL3 = fluct1.shape[0]
    

    with nogil:
        for i in prange(max_z_sep):
            for j in prange(NCL3-i):
                for k in prange(fluct1.shape[1]):
                    for l in prange(fluct2.shape[2]):
                        R_z[i,k,l] += fluct1[j,k,l]*fluct2[i+j,k,l]
   
    
    for i in range(max_z_sep):
        R_z[i] /= NCL3-i

    return R_z

def autocov_calc_x( np.ndarray[np.float64_t,ndim=3] fluct1,
                    np.ndarray[np.float64_t,ndim=3] fluct2,
                    np.ndarray[np.float64_t,ndim=3] R_x,
                    np.int32_t max_x_sep):
    

    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t NCL3 = fluct1.shape[0]

    with nogil:
        for i in prange(NCL3):
            for j in prange(fluct1.shape[1]):
                for k in prange(max_x_sep):
                    for l in prange(fluct2.shape[2]-max_x_sep):
                        R_x[k,j,l] += fluct1[i,j,l]*fluct2[i,j,l+k]
    
    R_x /= NCL3
    return R_x
