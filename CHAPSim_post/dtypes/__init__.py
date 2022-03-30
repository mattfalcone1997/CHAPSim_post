"""
==================
`` dtypes`` module
==================

This module that contains all of the main classes for the visualisation and storage of the data from the ``post`` module of CHAPSim_post. The intention is for this module to be extensive documented as it should represent a 'black-box' code which should not require in-depth knowledge of the user, with class and function interfaces sufficient for use.py


"""
    
from .flowstruct import *
from .vtk import *
from .coords import *
from .io import *
from .core import datastruct, metastruct, hdfHandler