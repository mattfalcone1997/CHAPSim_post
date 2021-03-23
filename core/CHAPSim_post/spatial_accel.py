
import h5py
import numpy as np
import os
from scipy import integrate
import sys

from . import post as cp

from . import plot as cplt

from CHAPSim_post.utils import misc_utils,indexing
from CHAPSim_post import POD

class CHAPSim_Inst(cp.CHAPSim_Inst):
    pass
_instant_class = CHAPSim_Inst

class CHAPSim_AVG_io(cp.CHAPSim_AVG_io):
    def _int_thickness_calc(self,PhyTime):
        disp_thickness, mom_thickness, shape_factor = super()._int_thickness_calc(PhyTime)
        return 2* disp_thickness, 2*mom_thickness, shape_factor
    def _tau_calc(self,PhyTime):
        
        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        for i in range(self.shape[1]):
            tau_star[i] = mu_star*(u_velo[0,i]-0.0)/(ycoords[0]-0.0)
    
        return tau_star
    def _bulk_velo_calc(self, PhyTime):
        return self.flow_AVGDF[PhyTime,'u'][-1,:].copy()

    def _y_plus_calc(self,PhyTime):

        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(delta_v_star.size,self.NCL[1])
        y_plus = np.zeros(y_plus_shape)
        y_coord = self.CoordDF['y']
        for i in range(len(delta_v_star)):
            y_plus[i] = y_coord/delta_v_star[i]
        return y_plus
_avg_io_class = CHAPSim_AVG_io

class CHAPSim_meta(cp.CHAPSim_meta):
    pass
_meta_class = CHAPSim_meta
