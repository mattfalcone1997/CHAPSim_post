import os
import numpy as np
__name__ = "CHAPSim_post"
__path__ = [os.path.dirname(__file__)]

class Params:
    def __init__(self):
        self.__params = {"TEST":False,
                        "autocorr_mode":2,
                        "ForceMode":False,
                        "Spacing":int(1),
                        "dtype": np.dtype('f8'),
                        "SymmetryAVG" : True,
                        "dissipation_correction": False,
                        "gradient_method":"numpy",
                        "gradient_order": 2,
                        "AVG_Style": "overline",
                        "relax_HDF_type_checks":False}

    def __getitem__(self,key):
        if key not in self.__params.keys():
            msg = f"Parameter not present must be one of the following: {self.__params.keys()}"
            raise KeyError(msg)

        return self.__params[key]

    def __setitem__(self,key,value):
        if key not in self.__params.keys():
            msg = f"Parameter not present must be one of the following: {self.__params.keys}"
            raise KeyError(msg)

        if key == 'dtype':
            if isinstance(value,str):
                value = np.dtype(value)
            elif not isinstance(value,np.dtype):
                msg = f"For key {key}, the value must be of type {str} or {np.dtype}"
                raise TypeError(msg)

        if not isinstance(value,type(self.__params[key])):
            msg = f"Parameter {key} requires arguments of type {type(self.__params[key])}"
            raise TypeError(msg)
        else:
            self.__params[key] = value

rcParams = Params()


