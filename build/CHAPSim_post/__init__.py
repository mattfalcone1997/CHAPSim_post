import os
import numpy as np

from CHAPSim_post.utils.misc_utils import Params
import CHAPSim_post.utils.style as style

__name__ = "CHAPSim_post"
__path__ = [os.path.dirname(__file__)]

Param_dict = {"TEST":False,
            "autocorr_mode":2,
            "ForceMode":False,
            "Spacing":int(1),
            "dtype": np.dtype('f8'),
            "SymmetryAVG" : True,
            "dissipation_correction": False,
            "gradient_method":"cython",
            "gradient_order": 2,
            "AVG_Style": "overline",
            "relax_HDF_type_checks":False,
            "use_parallel": 'thread',
            "use_cupy" : False}

rcParams = Params()
rcParams.update(Param_dict)




styleParams = style.styleParameters()
