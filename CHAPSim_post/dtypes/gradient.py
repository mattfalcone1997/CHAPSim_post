import numpy as np
# from abc import ABC
import weakref
from .flowstruct import *
import itertools

class vectorCalc():
    def __init__(self,flowstruct):
        
        if not self.check_valid(flowstruct):
            msg = "The flow struct is in valid for vector calc operations"
            raise TypeError(msg)
        
        self._flowstruct = weakref.ref(flowstruct)
    
    @staticmethod
    def check_valid(flowstruct):
        valid_type = (FlowStruct1D,
                      FlowStruct2D,
                      FlowStruct1D_time,
                      FlowStruct3D)
        if isinstance(flowstruct,valid_type):
            return True
        else:
            return False 
    
    @property
    def flowstruct(self):
        return self._flowstruct.ref()
    
    def _compute_grad(self,array,comp):
        # out_array = np.zeros_like(array)
        if comp == 'y':
            dim = self.flowstruct.get_dim_from_axis('y')            
            dim_after = self._flowstruct._dim - dim 
            slicer = ([np.newaxis]*dim,slice(None),[np.newaxis]*dim_after)
            slicer = tuple(itertools.chain(*slicer))
            y_coords = self.flowstruct.CoordDF['y'][slicer]
            
            in_array = array*y_coords
        else:
            in_array = array
            
        out_array = self.flowstruct.Domain.Grad_calc(self.flowstruct.CoordDF,
                                                        in_array,
                                                        comp)
        if comp == 'y':
            return out_array/y_coords
        else:
            return out_array
        
    def ScalarGrad(self,comps,coords,time):
        shape = (len(comps),*self.flowstruct.shape)
        out_array = np.zeros(shape)
        
        if len(comps) != len(coords):
            msg = ("The input coordinates must be the"
                    " smae length as the input comps)")
            raise ValueError(msg)
        
        for i, comp in enumerate(comps):
            array = self.flowstruct[time,comp]
            out_array[i] = self._compute_grad(array, coords[i])
            
        return out_array
        
                
        
    
    def VectorDiv(self,comps,coords,time):
        grad_array = self.ScalarGrad(comps,coords,time)
        
        return np.sum(grad_array,axis=0)
    
    def ScalarLaplace(self,comps,coords,time):
        shape = (len(comps),*self.flowstruct.shape)
        out_array = np.zeros(shape)
        
        if len(comps) != len(coords):
            msg = ("The input coordinates must be the"
                    " smae length as the input comps)")
            raise ValueError(msg)
        
        for i, comp in enumerate(comps):
            array = self.flowstruct[time,comp]
            grad_array = self._compute_grad(array, coords[i])
            out_array[i] = self._compute_grad(grad_array, coords[i])
            
        return out_array