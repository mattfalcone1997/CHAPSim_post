# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange
import cython

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def cumulativeInt_y1D_pipe(np.ndarray[np.float64_t,ndim=1] array,
                      np.ndarray[np.float64_t,ndim=1] staggered_y):
    
    cdef Py_ssize_t size = array.size
    
    cdef np.ndarray[np.float64_t,ndim=1] int_array
    int_array = np.zeros_like(array)

    cdef Py_ssize_t i, j
    cdef double dx, ycc
    for i in prange(size,nogil=True):
        for j in range(i+1):

            dx = staggered_y[j+1] - staggered_y[j]
            ycc = 0.5*(staggered_y[j+1] + staggered_y[j])
            int_array[i] += ycc*array[j]*dx
        ycc = 0.5*(staggered_y[i+1] + staggered_y[i])
        int_array[i] = (1./ycc)*int_array[i]

    return int_array

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def cumulativeInt_y1D_channel(np.ndarray[np.float64_t,ndim=1] array,
                            np.ndarray[np.float64_t,ndim=1] staggered_y):
    cdef Py_ssize_t size = array.size
    cdef Py_ssize_t mid = size // 2

    cdef np.ndarray[np.float64_t,ndim=1] int_array
    int_array = np.zeros_like(array)

    cdef Py_ssize_t i, j
    cdef double dx

    for i in prange(mid,size,nogil=True):
        for j in range(mid,i+1):

            dx = staggered_y[j+1] - staggered_y[j]
            int_array[i] += array[j]*dx

    for i in prange(mid,0,-1,nogil=True):
        for j in range(mid,i+1,-1):

            dx = staggered_y[j-1] - staggered_y[j]
            int_array[i] += array[j]*dx

    return int_array



@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def cumulativeInt_y1D(np.ndarray[np.float64_t,ndim=1] array,
                      np.ndarray[np.float64_t,ndim=1] staggered_y,
                      int channel):

    if staggered_y.size != array.size + 1:
        msg = ("This integration method must be"
                " called with the staggered data")
        raise RuntimeError(msg)

    if channel == 1:
        return cumulativeInt_y1D_channel(array,staggered_y)
    else:
        return cumulativeInt_y1D_pipe(array,staggered_y)
 

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def cumulativeInt_y2D_pipe(np.ndarray[np.float64_t,ndim=2] array,
                            np.ndarray[np.float64_t,ndim=1] staggered_y):

    cdef Py_ssize_t size_y = array.shape[0]
    cdef Py_ssize_t size_x = array.shape[1]
    
    cdef np.ndarray[np.float64_t,ndim=2] int_array
    int_array = np.zeros_like(array)

    cdef Py_ssize_t i, k, j
    cdef double dx, ycc

    for i in prange(size_y,nogil=True):
        for k in prange(size_x):
            for j in range(i+1):

                dx = staggered_y[j+1] - staggered_y[j]
                ycc = 0.5*(staggered_y[j+1] + staggered_y[j])
                int_array[i,k] += ycc*array[j,k]*dx

            ycc = 0.5*(staggered_y[i+1] + staggered_y[i])
            int_array[i,k] = (1./ycc)*int_array[i,k]

    return int_array
    
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def cumulativeInt_y2D_channel(np.ndarray[np.float64_t,ndim=2] array,
                            np.ndarray[np.float64_t,ndim=1] staggered_y):
    
    cdef Py_ssize_t size = array.size
    cdef Py_ssize_t mid = size // 2
    cdef Py_ssize_t size_x = array.shape[1]

    cdef np.ndarray[np.float64_t,ndim=2] int_array
    int_array = np.zeros_like(array)

    cdef Py_ssize_t i, j, k
    cdef double dx

    for i in prange(mid,size,nogil=True):
        for k in prange(size_x):
            for j in range(mid,i+1):

                dx = staggered_y[j+1] - staggered_y[j]
                int_array[i,k] += array[j,k]*dx

    for i in prange(mid,0,-1,nogil=True):
        for k in prange(size_x):
            for j in range(mid,i+1,-1):

                dx = staggered_y[j-1] - staggered_y[j]
                int_array[i,k] += array[j,k]*dx

    return int_array
    
    
def cumulativeInt_y2D(np.ndarray[np.float64_t,ndim=2] array,
                      np.ndarray[np.float64_t,ndim=1] staggered_y,
                      int channel):

    if staggered_y.size != array.shape[0] + 1:
        msg = ("This integration method must be"
                " called with the staggered data")
        raise RuntimeError(msg)

    if channel == 1:
        return cumulativeInt_y2D_channel(array,staggered_y)
    else:
        return cumulativeInt_y2D_pipe(array,staggered_y)
