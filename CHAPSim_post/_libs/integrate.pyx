# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange
from cython cimport floating, float, double, int
from CHAPSim_post._libs.array_utils cimport *
from CHAPSim_post._libs.array_utils import *


cdef floating _trapezoid_staggered(floating *input_array,
                                    floating *staggered_x,
                                    int axis,
                                    int *sizes,
                                    int *strides,
                                    int dim) nogil:

    cdef int total_size = get_total_size(dim, sizes)
    cdef int axis_stride = get_axis_stride(dim,axis,sizes)

    cdef int axis_index
    cdef floating delta_x, result=0
    cdef int i

    for i in prange(total_size):
        axis_index = get_axis_index(i,strides,axis)

        delta_x = ( staggered_x[axis_index+1] - staggered_x[axis_index] )
        result += input_array[i]*delta_x

    return result

cdef floating _trapezoid_centered(floating *input_array,
                                    floating *centered_x,
                                    int axis,
                                    int *sizes,
                                    int* strides,
                                    int dim) nogil:
    
    cdef int total_size = get_total_size(dim, sizes)
    cdef int axis_stride = get_axis_stride(dim,axis,sizes)

    cdef int axis_index, plus_index
    cdef floating delta_x, input_mid, result=0
    cdef int i
    cdef int loop_lim = total_size - axis_stride

    for i in prange(loop_lim):
        axis_index = get_axis_index(i,strides,axis)
        plus_index = i + axis_stride

        input_mid = 0.5*(input_array[plus_index] + input_array[i])
        delta_x = ( centered_x[axis_index+1] - centered_x[axis_index] )

        result += input_mid*delta_x

    return result

cdef floating _cum_trapezoid_staggered(floating *input_array,
                                    floating *staggered_x,
                                    int axis,
                                    int *sizes,
                                    int *strides,
                                    int dim,
                                    floating *output_array) nogil:

    cdef int total_size = get_total_size(dim, sizes)
    cdef int axis_stride = get_axis_stride(dim,axis,sizes)

    cdef int axis_index,start_index, end_index, cum_index
    cdef floating delta_x, result=0
    cdef int i, j
    cdef int limit = sizes[axis]
    cdef int limit_full

    for i in prange(total_size):
        axis_index = get_axis_index(i,strides,axis)
        start_index = i - axis_index*axis_stride
        end_index = i + axis_stride
        result=0

        for j in prange(start_index,end_index,axis_stride):
            cum_index = get_axis_index(j,strides,axis)
            delta_x = ( staggered_x[cum_index+1] - staggered_x[cum_index] )
            result += input_array[j]*delta_x

        output_array[i] = result


    

def IntegrateTrapz(np.ndarray input_array,np.ndarray x, int axis=0, bint staggered=True):
    
    cdef int dim = input_array.ndim
    if axis > dim:
        msg = "This axis cannot be larger than the dimensions of the array"
        raise ValueError(msg)

    if input_array.shape[axis] + 1 != x.size and staggered:
        msg = ("If staggered is selected the x array"
                " must be equal to the axis size +1. Sizes "
                f"{input_array.shape[axis]} and {x.size}")
        raise ValueError(msg)

    cdef np.ndarray[dtype=double,ndim=1] input_64
    cdef np.ndarray[dtype=float,ndim=1] input_32
    cdef np.ndarray[dtype=double,ndim=1] x_64 
    cdef np.ndarray[dtype=float,ndim=1] x_32
    
    cdef np.ndarray[dtype=int,ndim=1] strides
    cdef np.ndarray[dtype=int,ndim=1] sizes

    strides, sizes = get_array_details(input_array)

    if input_array.dtype == np.float64:
        input_64 = <np.ndarray[dtype=double,ndim=1]> input_array.flatten()

        x_64 = <np.ndarray[dtype=double,ndim=1]> x

        if staggered:
            return _trapezoid_staggered[double](&input_64[0],
                                                &x_64[0],
                                                axis,
                                                &sizes[0],
                                                &strides[0],
                                                dim)
        else:
            return _trapezoid_centered[double](&input_64[0],
                                                &x_64[0],
                                                axis,
                                                &sizes[0],
                                                &strides[0],
                                                dim)

    elif input_array.dtype == np.float32: 
        input_32 = <np.ndarray[dtype=float,ndim=1]> input_array.flatten() 
        x_32 = <np.ndarray[dtype=float,ndim=1]> x       
        if staggered:
            return _trapezoid_staggered[float](&input_32[0],
                                                &x_32[0],
                                                axis,
                                                &sizes[0],
                                                &strides[0],
                                                dim)
        else:
            return _trapezoid_centered[float](&input_32[0],
                                                &x_32[0],
                                                axis,
                                                &sizes[0],
                                                &strides[0],
                                                dim)

    else:
        msg = "Integrate can only handle type flost32 and float64"
        raise TypeError(msg)


def CumulatIntegrateTrapz(np.ndarray input_array,np.ndarray x, int axis=0):
    
    cdef int dim = input_array.ndim
    if axis > dim:
        msg = "This axis cannot be larger than the dimensions of the array"
        raise ValueError(msg)

    if input_array.shape[axis] + 1 != x.size:
        msg = (" x array must be equal to the axis size +1"
                f"{input_array.shape[axis]} and {x.size}")
        raise ValueError(msg)

    cdef np.ndarray[dtype=double,ndim=1] input_64
    cdef np.ndarray[dtype=float,ndim=1] input_32
    cdef np.ndarray[dtype=double,ndim=1] x_64 
    cdef np.ndarray[dtype=float,ndim=1] x_32
    cdef np.ndarray[dtype=double,ndim=1] out_64 
    cdef np.ndarray[dtype=float,ndim=1] out_32
    
    cdef np.ndarray[dtype=int,ndim=1] strides
    cdef np.ndarray[dtype=int,ndim=1] sizes

    strides, sizes = get_array_details(input_array)
    shape = [input_array.shape[i] for i in range(dim)]

    if input_array.dtype == np.float64:
        input_64 = <np.ndarray[dtype=double,ndim=1]> input_array.flatten()
        out_64 = np.zeros(input_array.size,dtype=np.float64) 
        x_64 = <np.ndarray[dtype=double,ndim=1]> x

        _cum_trapezoid_staggered[double](&input_64[0],
                                            &x_64[0],
                                            axis,
                                            &sizes[0],
                                            &strides[0],
                                            dim,
                                            &out_64[0])

        return out_64.reshape(shape)

    elif input_array.dtype == np.float32: 
        input_32 = <np.ndarray[dtype=float,ndim=1]> input_array.flatten() 
        x_32 = <np.ndarray[dtype=float,ndim=1]> x     
        out_32 = np.zeros(input_array.size,dtype=np.float32)   

        _cum_trapezoid_staggered[float](&input_32[0],
                                        &x_32[0],
                                        axis,
                                        &sizes[0],
                                        &strides[0],
                                        dim,
                                        &out_32[0])

        return out_32.reshape(shape)
    
# @cython.cdivision(True)
# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# def cumulativeInt_y1D_pipe(np.ndarray[np.float64_t,ndim=1] array,
#                       np.ndarray[np.float64_t,ndim=1] staggered_y):
    
#     cdef Py_ssize_t size = array.size
    
#     cdef np.ndarray[np.float64_t,ndim=1] int_array
#     int_array = np.zeros_like(array)

#     cdef Py_ssize_t i, j
#     cdef double dx, ycc
#     for i in prange(size,nogil=True):
#         for j in range(i+1):

#             dx = staggered_y[j+1] - staggered_y[j]
#             ycc = 0.5*(staggered_y[j+1] + staggered_y[j])
#             int_array[i] += ycc*array[j]*dx
#         ycc = 0.5*(staggered_y[i+1] + staggered_y[i])
#         int_array[i] = (1./ycc)*int_array[i]

#     return int_array

# @cython.cdivision(True)
# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# def cumulativeInt_y1D_channel(np.ndarray[np.float64_t,ndim=1] array,
#                             np.ndarray[np.float64_t,ndim=1] staggered_y):
#     cdef Py_ssize_t size = array.size
#     cdef Py_ssize_t mid = size // 2

#     cdef np.ndarray[np.float64_t,ndim=1] int_array
#     int_array = np.zeros_like(array)

#     cdef Py_ssize_t i, j
#     cdef double dx

#     for i in prange(mid,size,nogil=True):
#         for j in range(mid,i+1):

#             dx = staggered_y[j+1] - staggered_y[j]
#             int_array[i] += array[j]*dx

#     for i in prange(mid,0,-1,nogil=True):
#         for j in range(mid,i+1,-1):

#             dx = staggered_y[j-1] - staggered_y[j]
#             int_array[i] += array[j]*dx

#     return int_array



# @cython.cdivision(True)
# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# def cumulativeInt_y1D(np.ndarray[np.float64_t,ndim=1] array,
#                       np.ndarray[np.float64_t,ndim=1] staggered_y,
#                       int channel):

#     if staggered_y.size != array.size + 1:
#         msg = ("This integration method must be"
#                 " called with the staggered data")
#         raise RuntimeError(msg)

#     if channel == 1:
#         return cumulativeInt_y1D_channel(array,staggered_y)
#     else:
#         return cumulativeInt_y1D_pipe(array,staggered_y)
 

# @cython.cdivision(True)
# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# def cumulativeInt_y2D_pipe(np.ndarray[np.float64_t,ndim=2] array,
#                             np.ndarray[np.float64_t,ndim=1] staggered_y):

#     cdef Py_ssize_t size_y = array.shape[0]
#     cdef Py_ssize_t size_x = array.shape[1]
    
#     cdef np.ndarray[np.float64_t,ndim=2] int_array
#     int_array = np.zeros_like(array)

#     cdef Py_ssize_t i, k, j
#     cdef double dx, ycc

#     for i in prange(size_y,nogil=True):
#         for k in prange(size_x):
#             for j in range(i+1):

#                 dx = staggered_y[j+1] - staggered_y[j]
#                 ycc = 0.5*(staggered_y[j+1] + staggered_y[j])
#                 int_array[i,k] += ycc*array[j,k]*dx

#             ycc = 0.5*(staggered_y[i+1] + staggered_y[i])
#             int_array[i,k] = (1./ycc)*int_array[i,k]

#     return int_array
    
# @cython.cdivision(True)
# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# def cumulativeInt_y2D_channel(np.ndarray[np.float64_t,ndim=2] array,
#                             np.ndarray[np.float64_t,ndim=1] staggered_y):
    
#     cdef Py_ssize_t size = array.size
#     cdef Py_ssize_t mid = size // 2
#     cdef Py_ssize_t size_x = array.shape[1]

#     cdef np.ndarray[np.float64_t,ndim=2] int_array
#     int_array = np.zeros_like(array)

#     cdef Py_ssize_t i, j, k
#     cdef double dx

#     for i in prange(mid,size,nogil=True):
#         for k in prange(size_x):
#             for j in range(mid,i+1):

#                 dx = staggered_y[j+1] - staggered_y[j]
#                 int_array[i,k] += array[j,k]*dx

#     for i in prange(mid,0,-1,nogil=True):
#         for k in prange(size_x):
#             for j in range(mid,i+1,-1):

#                 dx = staggered_y[j-1] - staggered_y[j]
#                 int_array[i,k] += array[j,k]*dx

#     return int_array
    
    
# def cumulativeInt_y2D(np.ndarray[np.float64_t,ndim=2] array,
#                       np.ndarray[np.float64_t,ndim=1] staggered_y,
#                       int channel):

#     if staggered_y.size != array.shape[0] + 1:
#         msg = ("This integration method must be"
#                 " called with the staggered data")
#         raise RuntimeError(msg)

#     if channel == 1:
#         return cumulativeInt_y2D_channel(array,staggered_y)
#     else:
#         return cumulativeInt_y2D_pipe(array,staggered_y)
