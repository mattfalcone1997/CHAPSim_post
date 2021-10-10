# cython: language_level=3
# disutils: define_macros=MY_DTYPE

import numpy as np
cimport numpy as np
from cython.parallel import parallel,prange

import cython

ctypedef np.float32_t DTYPE_t
DTYPE = np.float32


@cython.boundscheck(False) 
@cython.wraparound(False) 
def autocov_calc_z( np.ndarray[DTYPE_t,ndim=3] fluct1,
                    np.ndarray[DTYPE_t,ndim=3] fluct2,
                    np.ndarray[DTYPE_t,ndim=3] R_z,
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

@cython.boundscheck(False) 
@cython.wraparound(False) 
def autocov_calc_x( np.ndarray[DTYPE_t,ndim=3] fluct1,
                    np.ndarray[DTYPE_t,ndim=3] fluct2,
                    np.ndarray[DTYPE_t,ndim=3] R_x,
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

@cython.boundscheck(False) 
@cython.wraparound(False) 
def cy_gradient_calc2D_dx(np.ndarray[DTYPE_t,ndim=2] flow_array,
                            DTYPE_t dx,
                            np.int32_t dim):
    cdef Py_ssize_t i, j, end
    cdef DTYPE_t dxi = 1.0/dx

    cdef np.ndarray[DTYPE_t,ndim=2] grad_array = np.empty([flow_array.shape[0],
                                                                flow_array.shape[1]],
                                                                dtype=DTYPE)

    if dim == 0:
        end = flow_array.shape[0]-1

        with nogil:
            for i in prange(1,end):
                for j in prange(flow_array.shape[1]):
                    grad_array[i,j] = 0.5*dxi*(flow_array[i+1,j] - flow_array[i-1,j])

            for j in prange(flow_array.shape[1]):
                grad_array[0,j] = dxi*(flow_array[1,j] - flow_array[0,j])
                grad_array[end,j] = dxi*(flow_array[end,j] - flow_array[end-1,j])

    else:
        end = flow_array.shape[1]-1
        with nogil:
            for i in prange(flow_array.shape[0]):
                for j in prange(1,end):
                    grad_array[i,j] = 0.5*dxi*(flow_array[i,j+1] - flow_array[i,j-1])

                for i in prange(flow_array.shape[0]):
                    grad_array[i,0] = dxi*(flow_array[i,1] - flow_array[i,0])
                    grad_array[i,end] = dxi*(flow_array[i,end] - flow_array[i,end])

    return grad_array

def cy_gradient_calc3D_dx(np.ndarray[DTYPE_t,ndim=3] flow_array,
                            DTYPE_t dx,
                            np.int32_t dim):

    cdef Py_ssize_t i, j,k, end
    cdef DTYPE_t dxi = 1.0/dx

    cdef np.ndarray[DTYPE_t,ndim=3] grad_array = np.empty([flow_array.shape[0],
                                                            flow_array.shape[1],
                                                            flow_array.shape[2]],
                                                            dtype=DTYPE)

    if dim == 0:
        end = flow_array.shape[0]-1

        with nogil:
            for i in prange(1,end):
                for j in prange(flow_array.shape[1]):
                    for k in prange(flow_array.shape[2]):
                        grad_array[i,j,k] = 0.5*dxi*(flow_array[i+1,j,k] - flow_array[i-1,j,k])

            for j in prange(flow_array.shape[1]):
                for k in prange(flow_array.shape[2]):
                    grad_array[0,j,k] = dxi*(flow_array[1,j,k] - flow_array[0,j,k])
                    grad_array[end,j,k] = dxi*(flow_array[end,j,k] - flow_array[end-1,j,k])

    elif dim == 1:
        end = flow_array.shape[1]-1
        with nogil:
            for i in prange(flow_array.shape[0]):
                for j in prange(1,end):
                    for k in prange(flow_array.shape[2]):
                        grad_array[i,j,k] = 0.5*dxi*(flow_array[i,j+1,k] - flow_array[i,j-1,k])

                for k in prange(flow_array.shape[2]):
                    grad_array[i,0,k] = dxi*(flow_array[i,1,k] - flow_array[i,0,k])
                    grad_array[i,end,k] = dxi*(flow_array[i,end,k] - flow_array[i,end-1,k])
    else:
        end = flow_array.shape[2]-1
        with nogil:
            for i in prange(flow_array.shape[0]):
                for j in prange(flow_array.shape[1]):
                    for k in prange(1,end):
                        grad_array[i,j,k] = 0.5*dxi*(flow_array[i,j,k+1] - flow_array[i,j,k-1])

                for j in prange(flow_array.shape[1]):
                    grad_array[i,j,0] = dxi*(flow_array[i,j,1] - flow_array[i,j,0])
                    grad_array[i,j,end] = dxi*(flow_array[i,j,end] - flow_array[i,j,end-1])
    return grad_array

@cython.boundscheck(False) 
@cython.wraparound(False) 
def cy_gradient_calc2D_var_x(np.ndarray[DTYPE_t,ndim=2] flow_array,
                        np.ndarray[DTYPE_t,ndim=1] dx_array,
                        np.int32_t dim):
    cdef Py_ssize_t i, j,k, end
    cdef DTYPE_t a_i, b_i,c_i
    cdef DTYPE_t a_b,b_b, dx1, dx2

    cdef np.ndarray[DTYPE_t,ndim=2] grad_array = np.empty([flow_array.shape[0],
                                                            flow_array.shape[1]],
                                                            dtype=DTYPE)

    if dim == 0:
        end = flow_array.shape[0]-1

        with nogil:
            for i in prange(1,end):
                dx1 = dx_array[i-1]
                dx2 = dx_array[i]

                a_i = -(dx2)/(dx1 * (dx1 + dx2))
                b_i = (dx2 - dx1) / (dx1 * dx2)
                c_i = dx1 / (dx2 * (dx1 + dx2))

                for j in prange(flow_array.shape[1]):
                    grad_array[i,j] = a_i*flow_array[i-1,j] + b_i*flow_array[i,j] + c_i*flow_array[i+1,j]
            
            dx1 = dx_array[0]
            dx2 = dx_array[end-1]

            a_b = 1/dx1
            b_b = 1/dx2
            for j in prange(flow_array.shape[1]):
                grad_array[0,j] = a_b*(flow_array[1,j] - flow_array[0,j])
                grad_array[end,j] = b_b*(flow_array[end,j] - flow_array[end-1,j])

    else:
        end = flow_array.shape[1]-1
        with nogil:
            for j in prange(1,end):
                dx1 = dx_array[j-1]
                dx2 = dx_array[j]

                a_i = -(dx2)/(dx1 * (dx1 + dx2))
                b_i = (dx2 - dx1) / (dx1 * dx2)
                c_i = dx1 / (dx2 * (dx1 + dx2))
                for i in prange(flow_array.shape[0]):
                    grad_array[i,j] = a_i*flow_array[i,j-1] + b_i*flow_array[i,j] + c_i*flow_array[i,j+1]

            dx1 = dx_array[0]
            dx2 = dx_array[end-1]

            a_b = 1/dx1
            b_b = 1/dx2
            for i in prange(flow_array.shape[0]):
                grad_array[i,0] = a_b*(flow_array[i,1] - flow_array[i,0])
                grad_array[i,end] = b_b*(flow_array[i,end] - flow_array[i,end])

    return grad_array


@cython.boundscheck(False) 
@cython.wraparound(False) 
def cy_gradient_calc3D_var_x(np.ndarray[DTYPE_t,ndim=3] flow_array,
                        np.ndarray[DTYPE_t,ndim=1] dx_array,
                        np.int32_t dim):

    cdef Py_ssize_t i, j,k, end
    cdef DTYPE_t a_i, b_i,c_i
    cdef DTYPE_t a_b,b_b, dx1, dx2

    cdef np.ndarray[DTYPE_t,ndim=3] grad_array = np.empty([flow_array.shape[0],
                                                            flow_array.shape[1],
                                                            flow_array.shape[2]],
                                                            dtype=DTYPE)

    if dim == 0:
        end = flow_array.shape[0]-1

        with nogil:
            for i in prange(1,end):
                dx1 = dx_array[i-1]
                dx2 = dx_array[i]

                a_i = -(dx2)/(dx1 * (dx1 + dx2))
                b_i = (dx2 - dx1) / (dx1 * dx2)
                c_i = dx1 / (dx2 * (dx1 + dx2))

                for j in prange(flow_array.shape[1]):
                    for k in prange(flow_array.shape[2]):
                        grad_array[i,j,k] = a_i*flow_array[i-1,j,k] + b_i*flow_array[i,j,k] + c_i*flow_array[i+1,j,k]
            
            dx1 = dx_array[0]
            dx2 = dx_array[end-1]

            a_b = 1/dx1
            b_b = 1/dx2
            for j in prange(flow_array.shape[1]):
                for k in prange(flow_array.shape[2]):
                    grad_array[0,j,k] = a_b*(flow_array[1,j,k] - flow_array[0,j,k])
                    grad_array[end,j,k] = b_b*(flow_array[end,j,k] - flow_array[end-1,j,k])

    elif dim ==1:
        end = flow_array.shape[1]-1

        with nogil:

            for i in prange(flow_array.shape[0]):
                for j in prange(1,end):
                    dx1 = dx_array[j-1]
                    dx2 = dx_array[j]

                    a_i = -(dx2)/(dx1 * (dx1 + dx2))
                    b_i = (dx2 - dx1) / (dx1 * dx2)
                    c_i = dx1 / (dx2 * (dx1 + dx2))
                    for k in prange(flow_array.shape[2]):
                        grad_array[i,j,k] = a_i*flow_array[i,j-1,k] + b_i*flow_array[i,j,k] + c_i*flow_array[i,j+1,k]

            dx1 = dx_array[0]
            dx2 = dx_array[end-1]

            a_b = 1/dx1
            b_b = 1/dx2
            for i in prange(flow_array.shape[0]):
                for k in prange(flow_array.shape[2]):
                    grad_array[i,0,k] = a_b*(flow_array[i,1,k] - flow_array[i,0,k])
                    grad_array[i,end,k] = b_b*(flow_array[i,end,k] - flow_array[i,end-1,k])
    else:
        end = flow_array.shape[2]-1
        with nogil:
            

            for i in prange(flow_array.shape[0]):
                for j in prange(flow_array.shape[1]):
                    for k in prange(1,end):
        
                        dx1 = dx_array[k-1]
                        dx2 = dx_array[k]

                        a_i = -(dx2)/(dx1 * (dx1 + dx2))
                        b_i = (dx2 - dx1) / (dx1 * dx2)
                        c_i = dx1 / (dx2 * (dx1 + dx2))
                        grad_array[i,j,k] = a_i*flow_array[i,j,k-1] + b_i*flow_array[i,j,k] + c_i*flow_array[i,j,k+1]

            dx1 = dx_array[0]
            dx2 = dx_array[end-1]

            a_b = 1/dx1
            b_b = 1/dx2
            for i in prange(flow_array.shape[0]):
                for j in prange(flow_array.shape[1]):
                    grad_array[i,j,0] = a_b*(flow_array[i,j,1] - flow_array[i,j,0])
                    grad_array[i,j,end] = b_b*(flow_array[i,j,end] - flow_array[i,j,end-1])
    return grad_array

