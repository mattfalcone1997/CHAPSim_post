from libc.stdio cimport *
from libc.stdlib cimport *
cimport numpy as np


cdef int get_axis_index(int index,
                         int* strides,
                         int axis) nogil
cdef int axis_eliminate_size(int *sizes,int dim,int axis) nogil

cdef int get_total_size(int dim,
                        int* sizes) nogil 

cdef int get_axis_stride(int dim,
                        int axis,
                        int* sizes) nogil

cdef void get_strides(int dim,
                        int* sizes,
                        int* strides)