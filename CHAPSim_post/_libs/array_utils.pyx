import cython

from libc.stdio cimport *
from libc.stdlib cimport *
cimport numpy as np
import numpy as np
from CHAPSim_post._libs.array_utils cimport *
    

def get_array_details(np.ndarray array):

    cdef np.ndarray[dtype=int,ndim=1] strides = np.zeros(array.ndim,dtype=np.int32)
    cdef np.ndarray[dtype=int,ndim=1] sizes = np.zeros(array.ndim,dtype=np.int32)

    for i in range(array.ndim):
        strides[i] = array.strides[i]/array.itemsize
        sizes[i] = array.shape[i]

    return strides, sizes

cdef int get_axis_index(int index,
                         int* strides,
                         int axis) nogil:

    cdef int i, larger_subtract =1
    cdef int axis_index = index
    cdef int axis_num
    for i in range(0,axis):
        axis_num = int(axis_index / strides[i])
        axis_index -= axis_num*strides[i]

    return int(axis_index / strides[axis])

cdef int get_total_size(int dim,
                        int* sizes) nogil :
    cdef int size = 1
    cdef int i

    for i in range(dim):
        size *= sizes[i]
    
    return size

cdef int get_axis_stride(int dim,
                        int axis,
                        int* sizes) nogil:
    cdef int stride = 1
    for j in range(axis+1,dim):
            stride *= sizes[j]
    
    return stride

cdef void get_strides(int dim,
                        int* sizes,
                        int* strides):
    cdef int i, j
    cdef int stride_int_
    for i in range(dim):
        strides[i] = 1
        for j in range(i+1,dim):
            strides[i] *= sizes[j]