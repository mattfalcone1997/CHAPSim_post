#setup for module CHAPSim_post
__name__="CHAPSim_post"

import numpy as np
import subprocess
import os
import shutil
import warnings
# warnings.simplefilter("error")
import CHAPSim_post as cp

from ._instant import CHAPSim_Inst
from ._instant import CHAPSim_Inst_tg
from ._instant import CHAPSim_Inst_io

from ._average import CHAPSim_AVG_io
from ._average import CHAPSim_AVG_tg_base
from ._average import CHAPSim_AVG_tg
from ._average import CHAPSim_AVG

from ._meta import CHAPSim_meta

from ._fluct import CHAPSim_fluct_io
from ._fluct import CHAPSim_fluct_tg

from ._budget import CHAPSim_budget_io
from ._budget import CHAPSim_budget_tg

from ._quadrant_a import CHAPSim_Quad_Anl_io
from ._quadrant_a import CHAPSim_Quad_Anl_tg

from ._joint_pdf import CHAPSim_joint_PDF_io

_avg_io_class = CHAPSim_AVG_io
_avg_tg_class = CHAPSim_AVG_tg
_avg_tg_base_class = CHAPSim_AVG_tg_base

_instant_class = CHAPSim_Inst

_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg

_meta_class = CHAPSim_meta


try:
   from .f_autocorr_parallel import autocov_calc_z, autocov_calc_x
except ImportError as e:
    print(e.args)
    if shutil.which("ifort") is not None:
        F90_exec = shutil.which("ifort")
    elif shutil.which("gfortran") is not None:
        F90_exec = shutil.which("gfortran")

    path = os.path.dirname(__file__)
    with open(os.path.join(path,"autocorr_parallel.f90")) as source:
        fortran_code = source.read()

    from numpy import f2py
    cwd = os.getcwd()
    os.chdir(path)
    extra_args = f"--opt='-O3 -fopenmp' --f90exec='{F90_exec}'"
    if F90_exec =="gfortran":
        F90_exec += "-lgomp" 
    err = f2py.compile(fortran_code,"f_autocorr_parallel",
                    extra_args=extra_args, verbose=False,extension=".f90")
    if err != 0:
        print("Using gfortran as backend")
        F90_exec="gfortran"
        extra_args=f"--opt='-O3 -fopenmp' --f90exec='{F90_exec}' -lgomp "
        err = f2py.compile(fortran_code,"f_autocorr_parallel",
                    extra_args=extra_args,verbose=False,extension=".f90")
        if err != 0:
            msg = "There was an issue compiling the fortran accelerator code"
            warnings.warn(msg)
    os.chdir(cwd)

from ._autocov import CHAPSim_autocov_io
from ._autocov import CHAPSim_autocov_tg