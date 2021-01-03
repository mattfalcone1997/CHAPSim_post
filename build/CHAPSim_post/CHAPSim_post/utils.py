"""
# utils.py
This submodule contains additional utilities for the low level code.
This will eventually also form part of a a revamped CHAPSim_Tools
module 
"""

def check_list_vals(x_list):
    if isinstance(x_list,(float,int)):
        x_list=[x_list]
    elif not isinstance(x_list,(tuple,list)):
        msg =  f"{x_list.__name__} must be of type float, int, tuple or list"
        raise TypeError(msg)
    else: # x_list is a tuple or list
        if not all([isinstance(x,(float,int)) for x in x_list]):
            msg = "If tuple or list provided, all items must be of instance float or int"
            raise TypeError(msg)
    
    return x_list