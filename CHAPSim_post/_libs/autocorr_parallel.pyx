# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange
import cython

ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def autocov_calc_io_z( np.ndarray[np.float64_t,ndim=3] fluct1,
                    np.ndarray[np.float64_t,ndim=3] fluct2,
                    np.int32_t max_z_sep):
    
    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t NCL3 = fluct1.shape[0]
    
    cdef Py_ssize_t NCL1 = fluct1.shape[2]
    cdef Py_ssize_t NCL2 = fluct1.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=3] R_z = np.zeros((max_z_sep,NCL2,NCL1),dtype=DTYPE)

    with nogil, parallel():
        for i in prange(max_z_sep):
            for j in prange(NCL3-i):
                for k in prange(NCL2):
                    for l in prange(NCL1):
                        R_z[i,k,l] = R_z[i,k,l] + fluct1[j,k,l]*fluct2[i+j,k,l]
   
    
    for i in range(max_z_sep):
        R_z[i] /= NCL3-i

    return R_z

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def autocov_calc_io_x( np.ndarray[np.float64_t,ndim=3] fluct1,
                    np.ndarray[np.float64_t,ndim=3] fluct2,
                    np.int32_t max_x_sep):
    

    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t NCL3 = fluct1.shape[0]

    cdef Py_ssize_t NCL1 = fluct1.shape[2]
    cdef Py_ssize_t NCL2 = fluct1.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=3] R_x = np.zeros((max_x_sep,NCL2,NCL1-max_x_sep),dtype=DTYPE)

    with nogil, parallel():
        for i in prange(NCL3):
            for j in prange(NCL2):
                for k in prange(max_x_sep):
                    for l in prange(NCL1-max_x_sep):
                        R_x[k,j,l] = R_x[k,j,l] + fluct1[i,j,l]*fluct2[i,j,l+k]/NCL3
    
    return R_x

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def autocov_calc_tg_z( np.ndarray[np.float64_t,ndim=3] fluct1,
                    np.ndarray[np.float64_t,ndim=3] fluct2,
                    np.int32_t max_z_sep):
    
    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t NCL3 = fluct1.shape[0]
    
    cdef Py_ssize_t NCL1 = fluct1.shape[2]
    cdef Py_ssize_t NCL2 = fluct1.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=2] R_z = np.zeros((max_z_sep,NCL2),dtype=DTYPE)

    with nogil:
        for i in prange(max_z_sep):
            for j in prange(NCL3-i):
                for k in prange(NCL2):
                    for l in prange(NCL1):
                        R_z[i,k] += (fluct1[j,k,l]*fluct2[i+j,k,l]/(NCL1*(NCL3-i)))
    
    
    return R_z

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def autocov_calc_tg_x( np.ndarray[np.float64_t,ndim=3] fluct1,
                    np.ndarray[np.float64_t,ndim=3] fluct2,
                    np.int32_t max_x_sep):
    

    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t NCL3 = fluct1.shape[0]

    cdef Py_ssize_t NCL1 = fluct1.shape[2]
    cdef Py_ssize_t NCL2 = fluct1.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=2] R_x = np.zeros((max_x_sep,NCL2),dtype=DTYPE)

    with nogil:
        for i in prange(NCL3):
            for j in prange(NCL2):
                for l in prange(max_x_sep):
                    for k in prange(NCL1-l):
                    
                        R_x[l,j] += (fluct1[i,j,k]*fluct2[i,j,k+l]/(NCL3*(NCL1-l)))
    
    
    return R_x
