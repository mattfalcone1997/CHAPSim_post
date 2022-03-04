# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import  prange, parallel
import cython

from libc.stdio cimport *
from libc.stdlib cimport *


@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def velo_interp3D(np.ndarray[np.float_t, ndim=3] flow_array,
                                                    int NCL1,
                                                    int NCL2,
                                                    int NCL3,
                                                    int dim):

    cdef np.ndarray[np.float64_t, ndim=3] flow_array_centered 
    flow_array_centered = np.zeros((NCL3,NCL2, NCL1-1),
                            dtype=np.float64)
    
    cdef Py_ssize_t i, j, k            

    if dim == 2:
        with nogil:
            for i in prange(NCL3):
                for j in prange(NCL2):
                    for k in prange(NCL1-1):
                        flow_array_centered[i,j,k] = 0.5*(flow_array[i,j,k] \
                                             + flow_array[i,j,k+1])

                   
    elif dim == 1:
        with nogil:
            for i in prange(NCL3):
                for j in prange(NCL2-1):
                    for k in prange(NCL1-1):
                        flow_array_centered[i,j,k] = 0.5*(flow_array[i,j,k] \
                                            + flow_array[i,j+1,k])

                    flow_array_centered[i,NCL2-1,k] = 0.5*(flow_array[i,NCL2-1,k] \
                                            + flow_array[i,0,k])

    elif dim ==0:
        with nogil:
            for i in prange(NCL3-1):
                for j in prange(NCL2):
                    for k in prange(NCL1-1):
                        flow_array_centered[i,j,k] = 0.5*(flow_array[i,j,k] \
                                            + flow_array[i+1,j,k])
                   
                    flow_array_centered[NCL3-1,j,k] = 0.5*(flow_array[NCL3-1,j,k] \
                                            + flow_array[0,j,k])

    return flow_array_centered
    
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def fluct_calc_io_f8(np.ndarray[np.float64_t, ndim =3] inst_array,
                    np.ndarray[np.float64_t, ndim=2] avg_array):

    cdef Py_ssize_t NCL3 = inst_array.shape[0]
    cdef Py_ssize_t NCL2 = inst_array.shape[1]
    cdef Py_ssize_t NCL1 = inst_array.shape[2]

    cdef Py_ssize_t i, j, k

    cdef np.ndarray[np.float64_t, ndim =3] fluct_array

    fluct_array = np.zeros_like(inst_array)

    for i in prange(NCL3,nogil=True):
        for j in prange(NCL2):
            for k in prange(NCL1):
                fluct_array[i,j,k] = inst_array[i,j,k] - avg_array[j,k]

    return fluct_array

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def fluct_calc_io_f4(np.ndarray[np.float32_t, ndim =3] inst_array,
                    np.ndarray[np.float32_t, ndim=2] avg_array):

    cdef Py_ssize_t NCL3 = inst_array.shape[0]
    cdef Py_ssize_t NCL2 = inst_array.shape[1]
    cdef Py_ssize_t NCL1 = inst_array.shape[2]

    cdef Py_ssize_t i, j, k

    cdef np.ndarray[np.float32_t, ndim =3] fluct_array

    fluct_array = np.zeros_like(inst_array)

    for i in prange(NCL3,nogil=True):
        for j in prange(NCL2):
            for k in prange(NCL1):
                fluct_array[i,j,k] = inst_array[i,j,k] - avg_array[j,k]

    return fluct_array

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def fluct_calc_tg_f8(np.ndarray[np.float64_t, ndim =3] inst_array,
                    np.ndarray[np.float64_t, ndim=1] avg_array):

    cdef Py_ssize_t NCL3 = inst_array.shape[0]
    cdef Py_ssize_t NCL2 = inst_array.shape[1]
    cdef Py_ssize_t NCL1 = inst_array.shape[2]

    cdef Py_ssize_t i, j, k

    cdef np.ndarray[np.float64_t, ndim =3] fluct_array

    fluct_array = np.zeros_like(inst_array)

    for i in prange(NCL3,nogil=True):
        for j in prange(NCL2):
            for k in prange(NCL1):
                fluct_array[i,j,k] = inst_array[i,j,k] - avg_array[j]

    return fluct_array


@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def fluct_calc_tg_f4(np.ndarray[np.float32_t, ndim =3] inst_array,
                    np.ndarray[np.float32_t, ndim=1] avg_array):

    cdef Py_ssize_t NCL3 = inst_array.shape[0]
    cdef Py_ssize_t NCL2 = inst_array.shape[1]
    cdef Py_ssize_t NCL1 = inst_array.shape[2]

    cdef Py_ssize_t i, j, k

    cdef np.ndarray[np.float32_t, ndim =3] fluct_array

    fluct_array = np.zeros_like(inst_array)

    for i in prange(NCL3,nogil=True):
        for j in prange(NCL2):
            for k in prange(NCL1):
                fluct_array[i,j,k] = inst_array[i,j,k] - avg_array[j]

    return fluct_array

def fluct_calc_io(np.ndarray inst_array, np.ndarray avg_array):

    if isinstance(inst_array,np.float64):
        return  fluct_calc_io_f8(inst_array,avg_array)
    elif isinstance(inst_array,np.float32): 
        return  fluct_calc_io_f4(inst_array,avg_array)

def fluct_calc_tg(np.ndarray inst_array, np.ndarray avg_array):

    if isinstance(inst_array,np.float64):
        return  fluct_calc_tg_f8(inst_array,avg_array)
    elif isinstance(inst_array,np.float32): 
        return  fluct_calc_tg_f4(inst_array,avg_array)