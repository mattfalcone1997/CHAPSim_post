import os
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup

def configuration(parent_package='',top_path=None):
    config = Configuration(None,parent_package,top_path)
    libraries = [
            #'gomp',
            #'blas',
            ]
    config.add_extension('f_autocorr_parallel',
                         ['autocorr_parallel.f90'],
                         libraries = libraries,
                         f2py_options = [],
                         # this is the flag gfortran needs to process OpenMP directives
                         extra_compile_args = ['-fopenmp','-O3','-ftree-vectorize'],
                         extra_link_args = [],
                         )
    return config