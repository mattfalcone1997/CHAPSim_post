# cython: language_level=3

from libc.stdio cimport *
from libc.stdlib cimport *

import numpy as np
cimport numpy as np
from cython.parallel import  prange, parallel
import cython


cdef void read_float64(FILE* file_ptr,
                        int count,
                        int position,
                        double* array) nogil:

    cdef int success
    cdef int pos_bytes
    cdef int el_size

    pos_bytes = position
    el_size = sizeof(double)

    success = fseek(file_ptr, position, SEEK_SET)

    fread(array,el_size,count,file_ptr)

cdef void read_int32(FILE* file_ptr, 
                    int count,
                    int position,
                    int* array) nogil:

    cdef int success
    cdef int pos_bytes
    cdef int el_size

    pos_bytes = position
    el_size = sizeof(int)

    success = fseek(file_ptr, pos_bytes, SEEK_SET)
    
    fread(array,el_size,count,file_ptr)

class ReadParallel:

    def __init__(self, file_list, mode):
        cdef int len_list = len(file_list) 
        self._len = len_list
        self._file_array = [file.encode('utf-8') for file in file_list]
        self._mode =  mode.encode('utf-8')

    @cython.cdivision(True)
    @cython.boundscheck(False) 
    @cython.wraparound(False) 
    def read_parallel_float64(self, int count,int position):

        cdef  results_list =  np.zeros((self._len,count))
        
        cdef FILE* File
        
        cdef char* c_mode = self._mode
        cdef Py_ssize_t n = self._len
        cdef Py_ssize_t i, j
        
        cdef char** c_files = <char**> malloc(sizeof(char*)*n)

        for i in range(n):
            c_files[i] =  self._file_array[i]

        cdef double [:,:] result_view = results_list

        for i in prange(n,nogil=True, schedule='dynamic'):

            # result_local = <double*> malloc(sizeof(double)*count)
            File = fopen(c_files[i],c_mode)
            if File == NULL:

                with gil:
                    msg = "File was not found"
                    raise FileNotFoundError(msg)
            read_float64(File,count,position,&result_view[i,0])

            # for j in prange(count):
            #     result_view[i,j] = result_local[j]

            fclose(File)
            # free(result_local)

        free(c_files)
       
        return results_list

    def read_parallel_int32(self, int count,int position):

        cdef results_list =  np.zeros((self._len,count))
        
        cdef FILE* File
        cdef char* c_mode = self._mode

        cdef Py_ssize_t n = self._len
        cdef Py_ssize_t i, j
        
        cdef char** c_files = <char**> malloc(sizeof(char*)*n)

        for i in range(n):
            c_files[i] =  self._file_array[i]

        cdef int [:,:] result_view = results_list

        for i in prange(n,nogil=True, schedule='dynamic'):

            File = fopen(c_files[i],c_mode)

            read_int32(File,count,position,&result_view[i,0])

            fclose(File)

        free(c_files)
        

        return results_list

                

        







