import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange
import cython

from libc.stdio cimport *
from libc.stdlib cimport *
from CHAPSim_post._libs.array_utils cimport *
from CHAPSim_post._libs.array_utils import *

cdef void _perform_gradient_varx(double *input_array,
                                double *x_array,
                                int axis,
                                int *sizes,
                                int *strides,
                                int dim,
                                double *gradient_array) nogil:

    cdef int total_size = get_total_size(dim, &sizes[0])
    cdef int minus_index, minus2_index, plus_index, plus2_index
    cdef int axis_index
    cdef int axis_stride = get_axis_stride(dim,axis,sizes)


    cdef double *a
    cdef double *b
    cdef double *c
    cdef int i

    a = <double*> malloc(sizes[axis]*sizeof(double))
    b = <double*> malloc(sizes[axis]*sizeof(double))
    c = <double*> malloc(sizes[axis]*sizeof(double))

    _get_grad_coeffs_varx(axis,sizes,x_array, a, b, c)

    for i in prange(total_size):
        axis_index = get_axis_index(i,strides,axis)

        minus_index = i - axis_stride
        plus_index = i + axis_stride
        minus2_index = i - 2*axis_stride
        plus2_index = i + 2*axis_stride

        if axis_index == 0:
            gradient_array[i] = a[axis_index]*input_array[i] + \
                                b[axis_index]*input_array[plus_index] +\
                                c[axis_index]*input_array[plus2_index]
        elif axis_index == sizes[axis] -1:
            gradient_array[i] = a[axis_index]*input_array[minus2_index] +\
                                b[axis_index]*input_array[minus_index] +\
                                c[axis_index]*input_array[i]
        else:
            gradient_array[i] = a[axis_index]*input_array[minus_index] + \
                                b[axis_index]*input_array[i] + \
                                c[axis_index]*input_array[plus_index]

    free(a)
    free(b)
    free(c)

cdef void _perform_gradient_constx(double *input_array,
                                    double dx,
                                    int axis,
                                    int *sizes,
                                    int *strides,
                                    int dim,
                                    double *gradient_array) nogil:

    cdef int total_size = get_total_size(dim, sizes)
    cdef int minus_index, minus2_index, plus_index, plus2_index
    cdef int a, b, c, axis_index
    cdef int axis_stride = get_axis_stride(dim,axis,sizes)

    cdef double idx = 1.0/dx 
    cdef int i

    for i in prange(total_size):
        axis_index = get_axis_index(i,strides,axis)
        
        minus_index = i - axis_stride
        plus_index = i + axis_stride
        minus2_index = i - 2*axis_stride
        plus2_index = i + 2*axis_stride

        if axis_index == 0:
            gradient_array[i] = idx*(-1.5*input_array[i] +\
                                      2 * input_array[plus_index] \
                                      -0.5 * input_array[plus2_index])

        elif axis_index == sizes[axis] - 1:
            gradient_array[i] = idx*(0.5*input_array[minus2_index] \
                                      -2 * input_array[minus_index] \
                                      +1.5 * input_array[i])
        else:
            gradient_array[i] = 0.5*idx*(input_array[plus_index] \
                                    - input_array[minus_index])

cdef void _get_grad_coeffs_varx(int axis,
                                int *sizes,
                                double *x_array,
                                double *a,
                                double *b,
                                double *c) nogil:
    

    cdef double dx1, dx2
    cdef int i

    for i in range(sizes[axis]):

        if i == 0:
            dx1 = x_array[i+1] - x_array[i]
            dx2 = x_array[i+2] - x_array[i+1]
            

            a[i] = -(2*dx1 + dx2)/(dx1*(dx1 + dx2))
            b[i] = (dx1 + dx2)/(dx1*dx2)
            c[i]  = -dx1/(dx2*(dx1 + dx2))
        elif i == sizes[axis] -1:
            dx1 = x_array[i-1] - x_array[i-2]
            dx2 = x_array[i] - x_array[i-1]

            a[i] = dx2/( dx1*(dx1 + dx2) )
            b[i] = -( dx1 + dx2 )/(dx1*dx2)
            c[i] = ( 2*dx2 + dx1 )/( dx2*( dx1 + dx2 ) )
        else:
            dx1 = x_array[i] - x_array[i-1]
            dx2 = x_array[i+1] - x_array[i]

            a[i] = -(dx2)/(dx1 * (dx1 + dx2))
            b[i] = (dx2 - dx1) / (dx1 * dx2)
            c[i] = dx1 / (dx2 * (dx1 + dx2))

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def gradient_calc(np.ndarray input_array,
                    np.ndarray[dtype=double,ndim=1] coord_array,
                    int axis):
    
    cdef int dim = input_array.ndim
    cdef int dtype = input_array.itemsize
    cdef double dx

    cdef np.ndarray[dtype=int,ndim=1] strides 
    cdef np.ndarray[dtype=int,ndim=1] sizes 

    cdef np.ndarray[dtype=double,ndim=1] grad_array = np.zeros(input_array.size)
    cdef np.ndarray[dtype=double,ndim=1] input_copy = input_array.flatten()

    cdef coord_diff = np.diff(coord_array)
    cdef int const_dx = int(np.allclose(coord_diff,coord_diff[0]))
    
    strides, sizes = get_array_details(input_array)
    
    if const_dx == 1:
        dx = coord_array[1] - coord_array[0]
        _perform_gradient_constx(&input_copy[0],
                                dx,
                                axis,
                                &sizes[0],
                                &strides[0],
                                dim,
                                &grad_array[0])

    else:
        _perform_gradient_varx(&input_copy[0],
                                &coord_array[0],
                                axis,
                                &sizes[0],
                                &strides[0],
                                dim,
                                &grad_array[0])

    shape = [input_array.shape[i] for i in range(dim)]
    return grad_array.reshape(shape)
