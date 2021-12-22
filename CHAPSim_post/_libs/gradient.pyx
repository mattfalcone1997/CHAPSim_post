import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange
import cython

from libc.stdio cimport *
from libc.stdlib cimport *

cdef void _perform_gradient_varx(double* input_array,
                                double* x_array,
                                int axis,
                                int* sizes,
                                int* strides,
                                int dim,
                                double* gradient_array) nogil:

    cdef int total_size = _get_total_size(dim, sizes)
    cdef int minus_index, plus_index
    cdef int axis_index
    cdef int axis_stride = _get_axis_stride(dim,axis,sizes)


    cdef double *a, *b, *c
    cdef int i

    a = <double*> malloc(sizes[axis]*sizeof(double))
    b = <double*> malloc(sizes[axis]*sizeof(double))
    c = <double*> malloc(sizes[axis]*sizeof(double))

    _get_grad_coeffs_varx(axis,sizes,x_array, a, b, c)

    for i in prange(total_size):
        axis_index = _get_axis_index(i,strides,axis)

        minus_index = i - axis_stride
        plus_index = i + axis_stride

        if axis_index == 0:
            gradient_array[i] = b[axis_index]*input_array[i] + \
                                c[axis_index]*input_array[plus_index]
        elif axis_index == sizes[axis] -1:
            gradient_array[i] = a[axis_index]*input_array[minus_index] +\
                                b[axis_index]*input_array[i]
        else:
            gradient_array[i] = a[axis_index]*input_array[minus_index] + \
                                b[axis_index]*input_array[i] + \
                                c[axis_index]*input_array[plus_index]

    free(a)
    free(b)
    free(c)

cdef void _perform_gradient_constx(double* input_array,
                                    double dx,
                                    int axis,
                                    int* sizes,
                                    int* strides,
                                    int dim,
                                    double* gradient_array) nogil:

    cdef int total_size = _get_total_size(dim, sizes)
    cdef int minus_index, plus_index
    cdef int a, b, c, axis_index
    cdef int axis_stride = _get_axis_stride(dim,axis,sizes)

    cdef double idx = 1.0/dx 
    cdef int i

    for i in prange(total_size):
        axis_index = _get_axis_index(i,strides,axis)
        
        minus_index = i - axis_stride
        plus_index = i + axis_stride

        if axis_index == 0:
            gradient_array[i] = idx*(input_array[plus_index] - input_array[i])
        elif axis_index == sizes[axis] - 1:
            gradient_array[i] = idx*(input_array[i] \
                                    - input_array[minus_index])
        else:
            gradient_array[i] = 0.5*idx*(input_array[plus_index] \
                                    - input_array[minus_index])


ctypedef enum index_flag:
    MINUS = 1
    ZERO = 2
    PLUS = 3

cdef void _get_grad_coeffs_varx(int axis,
                                int* sizes,
                                double* x_array,
                                double* a,
                                double* b,
                                double* c) nogil:
    

    cdef double dx1, dx2
    cdef int i

    for i in range(sizes[axis]):

        if i == 0:
            dx2 = x_array[i+1] - x_array[i]

            a[i] = 0
            b[i] = -1/dx2
            c[i]  = 1/dx2
        elif i == sizes[axis] -1:
            dx1 = x_array[i] - x_array[i-1]

            a[i] = -1/dx1
            b[i] = 1/dx1
            c[i] = 0.
        else:
            dx1 = x_array[i] - x_array[i-1]
            dx2 = x_array[i+1] - x_array[i]

            a[i] = -(dx2)/(dx1 * (dx1 + dx2))
            b[i] = (dx2 - dx1) / (dx1 * dx2)
            c[i] = dx1 / (dx2 * (dx1 + dx2))




cdef int _get_axis_index(int index,
                         int* strides,
                         int axis) nogil:

    cdef int i, larger_subtract =1
    cdef int axis_index = index
    cdef int axis_num
    for i in range(0,axis):
        axis_num = axis_index / strides[i]
        axis_index -= axis_num*strides[i]

    return axis_index/strides[axis]

cdef int _get_total_size(int dim,
                        int* sizes) nogil :
    cdef int size = 1
    cdef int i

    for i in range(dim):
        size *= sizes[i]
    
    return size

# cdef int _get_index_flag(int index,
#                         int axis,
#                         int* sizes,
#                         index_flag index_loc,
#                         int dim ) nogil:
    
#     if index_loc == MINUS:
#        return index - _get_axis_stride(dim,axis,sizes)
#     elif index_loc == PLUS:
#         return index + _get_axis_stride(dim,axis,sizes)
#     else:
#         return index

cdef int _get_axis_stride(int dim,
                        int axis,
                        int* sizes) nogil:
    cdef int stride = 1
    for j in range(axis+1,dim):
            stride *= sizes[j]
    
    return stride

# cdef int _get_index(int index,
#                     int* stride_count,
#                     int dim ) nogil:

#     cdef int return_index = 0
#     cdef int i
#     for i in range(dim):
#         return_index += stride_count[i]*index[i]

#     return return_index 


cdef void _get_strides(int dim,
                        int* sizes,
                        int* strides):
    cdef int i, j
    cdef int stride_int_
    for i in range(dim):
        strides[i] = 1
        for j in range(i+1,dim):
            strides[i] *= sizes[j]

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def gradient_calc(np.ndarray input_array,
                    np.ndarray[dtype=double,ndim=1] coord_array,
                    int axis):
    
    cdef int dim = input_array.ndim
    cdef int dtype = input_array.itemsize
    cdef double dx
    cdef int *strides, *sizes
    cdef np.ndarray[dtype=double,ndim=1] grad_array = np.zeros(input_array.size)

    cdef np.ndarray[dtype=double,ndim=1] input_copy = input_array.flatten()

    cdef coord_diff = np.diff(coord_array)
    cdef int const_dx = int(np.allclose(coord_diff,coord_diff[0]))
    
    strides = <int*> malloc(dim*sizeof(int))
    sizes = <int*> malloc(dim*sizeof(int))

    for i in range(dim):
        strides[i] = input_array.strides[i]/input_array.itemsize
        sizes[i] = input_array.shape[i]
    
    with nogil:
        if const_dx == 1:
            dx = coord_array[1] - coord_array[0]
            _perform_gradient_constx(&input_copy[0],
                                    dx,
                                    axis,
                                    sizes,
                                    strides,
                                    dim,
                                    &grad_array[0])

        else:
            _perform_gradient_varx(&input_copy[0],
                                    &coord_array[0],
                                    axis,
                                    sizes,
                                    strides,
                                    dim,
                                    &grad_array[0])

    free(strides)
    free(sizes)
    shape = [input_array.shape[i] for i in range(dim)]
    return grad_array.reshape(shape)
