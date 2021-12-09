"""
# _averaged.py
File contains the implementation of the classes to process the 
averaged results from the CHAPSim DNS solver 
"""

import warnings
import numpy as np

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import os
import gc
import itertools
import copy
from functools import partial
from abc import ABC, abstractmethod, abstractproperty

import CHAPSim_post as cp

from CHAPSim_post.utils import docstring, indexing,parallel, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd


from ._meta import CHAPSim_meta
from ._common import Common, postArray

from CHAPSim_post._libs import file_handler

_meta_class = CHAPSim_meta

class _AVG_base(Common,ABC):
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        if fromfile:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extract_avg(*args,**kwargs)

    @abstractmethod
    def _extract_avg(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def y_coord_index_norm(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def ycoords_from_coords(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def ycoords_from_norm_coords(self,*args,**kwargs):
        pass

    @property
    def shape(self):
        avg_index = self.flow_AVGDF.index[0]
        return self.flow_AVGDF[avg_index].shape

    @property
    def times(self):
        return np.array(self.flow_AVGDF.times)

    def get_times(self):
        return ["%.9g"%x for x in self.times]

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    def save_hdf(self,file_name,write_mode,key=None):
        """
        Saving the CHAPSim_AVG classes to a file in hdf5 file format

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,mode=write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        hdf_obj.create_dataset("NCL",data=self.NCL)

        self._meta_data.save_hdf(file_name,'a',key=key+'/meta_data')
        self.flow_AVGDF.to_hdf(file_name,key=key+'/flow_AVGDF',mode='a')
        self.PU_vectorDF.to_hdf(file_name,key=key+'/PU_vectorDF',mode='a')
        self.UU_tensorDF.to_hdf(file_name,key=key+'/UU_tensorDF',mode='a')
        self.UUU_tensorDF.to_hdf(file_name,key=key+'/UUU_tensorDF',mode='a')
        self.Velo_grad_tensorDF.to_hdf(file_name,key=key+'/Velo_grad_tensorDF',mode='a')
        self.PR_Velo_grad_tensorDF.to_hdf(file_name,key=key+'/PR_Velo_grad_tensorDF',mode='a')
        self.DUDX2_tensorDF.to_hdf(file_name,key=key+'/DUDX2_tensorDF',mode='a')

    @abstractmethod
    def _hdf_extract(self,*args,**kwargs):
        pass

    def _Reverse_decomp(self,flow_AVGDF,PU_vectorDF,UU_tensorDF,
                        UUU_tensorDF,Velo_grad_tensorDF,
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF):

        for index in PU_vectorDF.index:
            P_mean = flow_AVGDF[index[0],'p']
            u_mean = flow_AVGDF[index]
            PU_vectorDF[index] -= P_mean*u_mean

        for index in UU_tensorDF.index:
            u1_mean = flow_AVGDF[index[0],index[1][0]]
            u2_mean = flow_AVGDF[index[0],index[1][1]]
            UU_tensorDF[index] -= u1_mean*u2_mean


        for index in UUU_tensorDF.index:
            u1u2 = UU_tensorDF[index[0],index[1][:2]]
            u2u3 = UU_tensorDF[index[0],index[1][1:]]
            comp13 = index[1][0] + index[1][2]
            u1u3 = UU_tensorDF[index[0],comp13]
            u1_mean = flow_AVGDF[index[0],index[1][0]]
            u2_mean = flow_AVGDF[index[0],index[1][1]]
            u3_mean = flow_AVGDF[index[0],index[1][2]]
            UUU_tensorDF[index] -= (u1_mean*u2_mean*u3_mean + u1_mean*u2u3 \
                        + u2_mean*u1u3 + u3_mean*u1u2)

        for index in PR_Velo_grad_tensorDF.index:
            p_mean = flow_AVGDF[index[0],'p']
            u_grad = Velo_grad_tensorDF[index]
            PR_Velo_grad_tensorDF[index] -= p_mean*u_grad

        for index in DUDX2_tensorDF.index:
            comp1 = index[1][1] + index[1][3]
            comp2 = index[1][5] + index[1][7]
            u1x_grad = Velo_grad_tensorDF[index[0],comp1]
            u2x_grad = Velo_grad_tensorDF[index[0],comp2]
            DUDX2_tensorDF[index] -= u1x_grad*u2x_grad

        return flow_AVGDF,PU_vectorDF,UU_tensorDF,\
                        UUU_tensorDF,Velo_grad_tensorDF,\
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF

    @abstractmethod
    def _AVG_extract(self, *args,**kwargs):
        raise NotImplementedError

    def check_PhyTime(self,PhyTime):
        """
        Checks whether the physical time provided is valid
        and if not whether it can be recovered.

        Parameters
        ----------
        PhyTime : float or int
            Input Physical time to be checked

        Returns
        -------
        float or int
            Correct or corrected physical time
        """

        warn_msg = f"PhyTime invalid ({PhyTime}), variable being set to only PhyTime present in datastruct"
        err_msg = KeyError("PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible")
        
        return self.flow_AVGDF.check_outer(PhyTime,err_msg,warn_msg) 

class _AVG_developing(_AVG_base):
    @abstractmethod
    def _return_index(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _return_xaxis(self,*args,**kwargs):
        raise NotImplementedError

    @abstractproperty
    def _shape_devel(self):
        pass

    def _wall_unit_calc(self,PhyTime):
        
        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star

        REN = self.metaDF['REN']
        
        tau_w = self._tau_calc(PhyTime)
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star = (nu_star/u_tau_star)/REN
        return u_tau_star, delta_v_star

    def _y_plus_calc(self,PhyTime):

        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(self._shape_devel[1],int(self.NCL[1]*0.5))
        y_plus = np.zeros(y_plus_shape)
        y_coord = self.CoordDF['y'][:int(self.NCL[1]*0.5)]
        for i in range(len(delta_v_star)):
            y_plus[i] = (1-abs(y_coord))/delta_v_star[i]
        return y_plus    

    def _int_thickness_calc(self,PhyTime):

        U0_index = 0 if self.Domain.is_polar else int(self.shape[0]*0.5)
        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']

        U0 = U_mean[U0_index]
        theta_integrand = np.zeros_like(U_mean)
        delta_integrand = np.zeros_like(U_mean)

        for i, _ in enumerate(theta_integrand):
            theta_integrand[i] = (U_mean[i]/U0)*(1 - U_mean[i]/U0)
            delta_integrand[i] = 1 - U_mean[i]/U0

        mom_thickness = 0.5*integrate_simps(theta_integrand,y_coords,axis=0)
        disp_thickness = 0.5*integrate_simps(delta_integrand,y_coords,axis=0)
        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor

    def _bulk_velo_calc(self,PhyTime):
            
        u_velo = self.flow_AVGDF[PhyTime,'u'].squeeze()
        ycoords = self.CoordDF['y']

        bulk_velo = 0.5*integrate_simps(u_velo,ycoords,axis=0)
            
        return bulk_velo

    def _tau_calc(self,PhyTime):
        
        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        for i in range(self._shape_devel[1]):
            tau_star[i] = -mu_star*(u_velo[-1,i]-0.0)/(ycoords[-1]-1.0)
        

        return tau_star

    def _Cf_calc(self,PhyTime):
        rho_star = 1.0
        REN = self.metaDF['REN']
        tau_star = self.tau_calc(PhyTime)
        bulk_velo = self.bulk_velo_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star

        return skin_friction

    def _eddy_visc_calc(self,PhyTime):
        uv = self.UU_tensorDF[PhyTime,'uv']
        dUdy = self.Velo_grad_tensorDF[PhyTime,'uy']
        dVdx = self.Velo_grad_tensorDF[PhyTime,'vx']
        REN = self.metaDF['REN']

        mu_t = -uv*REN/(dUdy + dVdx)
        return mu_t

class CHAPSim_AVG_io(_AVG_developing):
    def _extract_file_io(self,PhyTime,path_to_folder,abs_path,time0=None):

        full_path = misc_utils.check_paths(path_to_folder,'2_averaged_rawdata',
                                                            '2_averagd_D')

        file, NSTATIS,PhyTime, NCL1, NCL2 = self._get_io_file_boilerplate(full_path,PhyTime,abs_path)
        
        offset = 4*4 + 3*8
        dummy_size = NCL1*NCL2*50*21
        
        if time0 is None:
            
            AVG_info = np.fromfile(file,dtype='float64',offset=offset,count=dummy_size)
        else:
            file0, NSTATIS0,_, _, _ = self._get_io_file_boilerplate(full_path,time0,abs_path)
            
            parallel_file = file_handler.ReadParallel([file,file0],'rb')
            AVG_info, AVG_info0 = parallel_file.read_parallel_float64(dummy_size,offset)
            
            AVG_info = (AVG_info*NSTATIS - AVG_info0*NSTATIS0)/(NSTATIS - NSTATIS0)
        
        return AVG_info, PhyTime, NCL1, NCL2

    def _get_io_file_boilerplate(self,path, PhyTime,abs_path):
        instant = "%0.9E" % PhyTime
        
        file_string = "DNS_perioz_AVERAGD_T" + instant + "_FLOW.D"
        
        if not abs_path:
            file_path = os.path.abspath(os.path.join(path, file_string))
        else:
            file_path = os.path.join(path, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)

        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL1 = int_info[0]
        NCL2 = int_info[1]
        NSTATIS = int_info[3]
        
        dummy_size = NCL1*NCL2*50*21
        r_info = np.fromfile(file,dtype='float64',count=3)
        PhyTime = r_info[0]
        
        file.close()
        return file_path, NSTATIS, PhyTime, NCL1, NCL2
        
    def _extract_avg(self,time,path_to_folder=".",time0=None,abs_path=True):
        """
        Instantiates an instance of the CHPSim_AVG_io class from the result data

        Parameters
        ----------
        time : float or list or floats
            Physcial time to be extracted
        meta_data : CHAPSim_meta, optional
            metadata class provided to the class, by default None
        path_to_folder : str, optional
            path to the results folder, by default "."
        time0 : float, optional
            A previous time to average between, by default None
        abs_path : bool, optional
            Whether the path provided is absolute or relative, by default True
        """


        meta_data = self._module._meta_class(path_to_folder,abs_path,False)
        self._meta_data = meta_data.copy()

        time = misc_utils.check_list_vals(time)

        for PhyTime in time:
            if 'DF_list' not in locals():
                DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
            else:

                local_DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                for i, _ in enumerate(DF_list):
                    DF_list[i].concat(local_DF_list[i])

        DF_list=self._Reverse_decomp(*DF_list)


        self.flow_AVGDF,self.PU_vectorDF,\
        self.UU_tensorDF,self.UUU_tensorDF,\
        self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
        self.DUDX2_tensorDF = DF_list
    
    def _hdf_extract(self,file_name,key=None):
        """
        Creates class by extracting data from the hdf5 file

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        if key is None:
            key = self.__class__.__name__

        misc_utils.check_path_exists(file_name,True)

        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')

        self.flow_AVGDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/flow_AVGDF')
        self.PU_vectorDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/PU_vectorDF')
        self.UU_tensorDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/UU_tensorDF')
        self.UUU_tensorDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/UUU_tensorDF')
        self.Velo_grad_tensorDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/Velo_grad_tensorDF')
        self.PR_Velo_grad_tensorDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/PR_Velo_grad_tensorDF')
        self.DUDX2_tensorDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/DUDX2_tensorDF')

    def _AVG_extract(self,Time_input,time0,path_to_folder,abs_path):

        
        AVG_info, PhyTime, NCL1, NCL2 = self._extract_file_io(Time_input,
                                                       path_to_folder,
                                                       abs_path,
                                                       time0)

        AVG_info = AVG_info.reshape(21,50,NCL2,NCL1)
            
        #Velo_AVG = np.zeros((3,NCL2,NCL1))
        Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        Pr_Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        DUDX2_tensor = np.zeros((81,NCL2,NCL1))
        
        for i in range(3):
            for j in range(3):
                Velo_grad_tensor[i*3+j,:,:] = AVG_info[6+j,i,:,:]
                Pr_Velo_grad_tensor[i*3+j,:,:] = AVG_info[9+j,i,:,:]
                
        for i in range(9):
            for j in range(9):
                DUDX2_tensor[i*9+j] = AVG_info[12+j,i,:,:] 
        
        flow_AVG = AVG_info[0,:4,:,:].copy()
        PU_vector = AVG_info[2,:3,:,:].copy()
        UU_tensor = AVG_info[3,:6,:,:].copy()
        UUU_tensor = AVG_info[5,:10,:,:].copy()

        del AVG_info; gc.collect()



        Phy_string = '%.9g' % PhyTime
        flow_index = [[Phy_string]*4,['u','v','w','p']]
        vector_index = [[Phy_string]*3,['u','v','w']]
        sym_2_tensor_index = [[Phy_string]*6,['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[Phy_string]*10,['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[Phy_string]*9,['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']

        dudx_list = list(itertools.product(du_list,dx_list))
        dudx_list = ["".join(dudx) for dudx in dudx_list]
        comp_string_list = list(itertools.product(dudx_list,dudx_list))
        comp_string_list = ["".join(comp_string) for comp_string in comp_string_list]

        tensor_4_index = [[Phy_string]*81,comp_string_list]

        flow_AVGDF = cd.FlowStruct2D(self._coorddata,flow_AVG,index=flow_index) 
        PU_vectorDF = cd.FlowStruct2D(self._coorddata,PU_vector,index=vector_index) 
        UU_tensorDF = cd.FlowStruct2D(self._coorddata,UU_tensor,index=sym_2_tensor_index) 
        UUU_tensorDF = cd.FlowStruct2D(self._coorddata,UUU_tensor,index=sym_3_tensor_index) 
        Velo_grad_tensorDF = cd.FlowStruct2D(self._coorddata,Velo_grad_tensor,index=tensor_2_index) 
        PR_Velo_grad_tensorDF = cd.FlowStruct2D(self._coorddata,Pr_Velo_grad_tensor,index=tensor_2_index) 
        DUDX2_tensorDF = cd.FlowStruct2D(self._coorddata,DUDX2_tensor,index=tensor_4_index) 

        if cp.rcParams["SymmetryAVG"] and self.metaDF['iCase'] == 1:
            flow_AVGDF = 0.5*(flow_AVGDF + \
                            flow_AVGDF.symmetrify(dim=0))
            PU_vectorDF = 0.5*(PU_vectorDF +\
                            PU_vectorDF.symmetrify(dim=0))
            UU_tensorDF = 0.5*(UU_tensorDF + \
                            UU_tensorDF.symmetrify(dim=0))
            UUU_tensorDF = 0.5*(UUU_tensorDF + \
                            UUU_tensorDF.symmetrify(dim=0))
            Velo_grad_tensorDF = 0.5*(Velo_grad_tensorDF + \
                            Velo_grad_tensorDF.symmetrify(dim=0))
            PR_Velo_grad_tensorDF = 0.5*(PR_Velo_grad_tensorDF + \
                            PR_Velo_grad_tensorDF.symmetrify(dim=0))
            DUDX2_tensorDF = 0.5*(DUDX2_tensorDF + \
                            DUDX2_tensorDF.symmetrify(dim=0))
        
        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]

    @property
    def _shape_devel(self):
        return self.shape

    def _return_index(self,x_val):
        return indexing.coord_index_calc(self.CoordDF,'x',x_val)

    def _return_xaxis(self):
        return self.CoordDF['x']

    def int_thickness_calc(self, PhyTime=None):
        """
        Calculates the integral thicknesses and shape factor 

        Parameters
        ----------
        PhyTime : float or int, optional
            Physical ime, by default None

        Returns
        -------
        %(ndarray)s:
            Displacement thickness
        %(ndarray)s:
            Momentum thickness
        %(ndarray)s:
            Shape factor
        """

        PhyTime = self.check_PhyTime(PhyTime)
        return self._int_thickness_calc(PhyTime)

    @docstring.sub
    def wall_unit_calc(self,PhyTime=None):
        """
        returns arrays for the friction velocity and viscous lengthscale

        Parameters
        ----------
        PhyTime : float or int, optional
            Physical time, by default None

        Returns
        -------
        %(ndarray)s:
            Friction velocity array
        %(ndarray)s:
            Viscous lengthscale array
            
        """
        PhyTime = self.check_PhyTime(PhyTime)
        return self._wall_unit_calc(PhyTime)

    def bulk_velo_calc(self,PhyTime=None):
        """
        Method to calculate the bulk velocity against the streamwise coordinate

        Parameters
        ----------
        PhyTime : (int,float), optional
            Physical time, by default None

        Returns
        -------
        %(ndarray)s
            array containing the bulk velocity
        """

        PhyTime = self.check_PhyTime(PhyTime)
        return self._bulk_velo_calc(PhyTime)

    def tau_calc(self,PhyTime=None):
        """
        method to return the wall shear stress array

        Parameters
        ----------
        PhyTime : (float,int), optional
            Physical time, if value is invalid or None the routine will
            attempt recovery, by default None

        Returns
        -------
        %(ndarray)s
            Wall shear stress array 
        """
        PhyTime = self.check_PhyTime(PhyTime)            
        return self._tau_calc(PhyTime)

    def y_coord_index_norm(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        
        if avg_time is None:
            avg_time = max(self.times)
        else:
            avg_time = self.check_PhyTime(avg_time)

        avg_time = self.check_PhyTime(avg_time)

        if mode=='half_channel':
            norm_distance=np.ones((self.NCL[0]))
        elif mode == 'disp_thickness':
            norm_distance, _,_ = self.int_thickness_calc(avg_time)
        elif mode == 'mom_thickness':
            _, norm_distance, _ = self.int_thickness_calc(avg_time)
        elif mode == 'wall':
            _, norm_distance = self.wall_unit_calc(avg_time)
        else:
            msg = ("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
                                    " or 'wall. Value used was %s\n"%mode)
            raise ValueError(msg)
        

        if x_vals is None:
            x_index=list(range(self.shape[-1]))
        else:
            x_vals = misc_utils.check_list_vals(x_vals)
            x_index =[self._return_index(x) for x in x_vals]

        y_indices = []
        for i,x in enumerate(x_index):
            if self.Domain.is_polar:
                norm_coordDF = -1*(self.CoordDF-1)/norm_distance[x]
            elif self.Domain.is_channel:
                norm_coordDF = (1-abs(self.CoordDF))/norm_distance[x]
            else:
                norm_coordDF = self.CoordDF/norm_distance[x]

            y_index = indexing.coord_index_calc(norm_coordDF,'y', coord_list)
            y_indices.append(y_index)

        return y_indices

    def ycoords_from_coords(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if avg_time is None:
            avg_time = max(self.times)
        else:
            avg_time = self.check_PhyTime(avg_time)

        if mode=='half_channel':
            norm_distance=np.ones((self.NCL[0]))
        elif mode == 'disp_thickness':
            norm_distance, _,_ = self.int_thickness_calc(avg_time)
        elif mode == 'mom_thickness':
            _, norm_distance, _ = self.int_thickness_calc(avg_time)
        elif mode == 'wall':
            _, norm_distance = self.wall_unit_calc(avg_time)
        else:
            msg = ("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
                        " or 'wall. Value used was %s\n"%mode)
            raise ValueError(msg)
        indices = self.y_coord_index_norm(coord_list,avg_time=avg_time,x_vals=x_vals,mode=mode)

        if x_vals is None:
            x_index=list(range(self.shape[-1]))
        else:
            x_vals = misc_utils.check_list_vals(x_vals)
            x_index =[self._return_index(x) for x in x_vals]

        true_ycoords = []
        for x,index in zip(x_index,indices):
            if self.Domain.is_polar:
                norm_coordDF = -1*(self.CoordDF-1)/norm_distance[x]
            elif self.Domain.is_channel:
                norm_coordDF = (1-abs(self.CoordDF))/norm_distance[x]
            else:
                norm_coordDF = self.CoordDF/norm_distance[x]
            true_ycoords.append(norm_coordDF['y'][index])

        return true_ycoords

    def ycoords_from_norm_coords(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if avg_time is None:
            avg_time = max(self.times)
        else:
            avg_time = self.check_PhyTime(avg_time)

        indices = self.y_coord_index_norm(coord_list,avg_time=avg_time,x_vals=x_vals,mode=mode)  
        true_ycoords = []
        for index in zip(indices):
            true_ycoords.append(self.CoordDF['y'][index])
    
        return true_ycoords

    def plot_shape_factor(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)

        _, _, shape_factor = self.int_thickness_calc(PhyTime)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_coords = self.CoordDF['x']
        line_kw = cplt.update_line_kw(line_kw,label = r"$H$")
        ax.cplot(x_coords,shape_factor,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"$H$")

        return fig, ax

    def plot_mom_thickness(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)
        _, theta, _ = self.int_thickness_calc(PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        x_coords = self.CoordDF['x']

        line_kw = cplt.update_line_kw(line_kw,label=r"$\theta$")
        ax.cplot(x_coords,theta,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"$\theta$")
        fig.tight_layout()

        return fig, ax

    def plot_disp_thickness(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        delta, _, _ = self.int_thickness_calc(PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_coords = self.CoordDF['x']

        line_kw = cplt.update_line_kw(line_kw,label=r"$\delta^*$")
        ax.cplot(x_coords,delta,**line_kw)

        xlabel = self.Domain.create_label(r'$x$')
        ax.set_xlabel(xlabel)

        ax.set_ylabel(r"$\delta^*$")
        fig.tight_layout()

        return fig, ax

    def plot_mean_flow(self,comp,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = misc_utils.check_list_vals(x_vals)
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)

        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        fig, ax = self.flow_AVGDF.plot_line(comp,'y',x_vals,
                            time=PhyTime,labels=labels,
                            fig=fig,ax=ax,line_kw=line_kw,**kwargs)


        x_label = self.Domain.create_label(r"$y$")
        y_label = self.Domain.create_label(r"$\bar{%s}$"%comp)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # self.Domain.styleAxes(ax)
        
        ncol = cplt.get_legend_ncols(len(x_vals))
        ax.clegend(vertical=False,ncol=ncol)

        return fig, ax

    def _get_uplus_yplus_transforms(self,PhyTime,x_val):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        x_index = self.CoordDF.index_calc('x',x_val)[0]
        
        if self.Domain.is_polar:
            x_transform = lambda y:  -1*(y - 1.0)/delta_v[x_index]
            y_transform = lambda u: u/u_tau[x_index]
        else:
            x_transform = lambda y:  (y + 1.0)/delta_v[x_index]
            y_transform = lambda u: u/u_tau[x_index]
        
        return x_transform, y_transform
    
    def plot_flow_wall_units(self,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = misc_utils.check_list_vals(x_vals)
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)

        for x in x_vals:
            label = self.Domain.create_label(r"$x = %.3g$"%x)
            x_transform, y_transform = self._get_uplus_yplus_transforms(PhyTime, x)
            fig, ax = self.flow_AVGDF.plot_line('u','y',
                                                x_vals,
                                                transform_xdata = x_transform,
                                                transform_ydata = y_transform,
                                                labels=[label],
                                                channel_half=True,
                                                time=PhyTime,
                                                fig=fig,
                                                ax=ax,
                                                line_kw=line_kw,
                                                **kwargs)

        labels = [l.get_label() for l in ax.get_lines()]
        if not r"$\bar{u}^+=y^+$" in labels:
            uplus_max = np.amax([l.get_ydata() for l in ax.get_lines()])
            u_plus_array = np.linspace(0,uplus_max,100)

            ax.cplot(u_plus_array,u_plus_array,label=r"$\bar{u}^+=y^+$",color='r',linestyle='--')

        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")
        ax.set_xscale('log')
        ncol = cplt.get_legend_ncols(len(x_vals))
        ax.clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_Reynolds(self,comp,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        if comp not in self.UU_tensorDF.inner_index:
            comp = comp[::-1]
            if not comp not in self.UU_tensorDF.inner_index:
                msg = "Reynolds stress component %s not found"%comp
                raise ValueError(msg) 

        PhyTime = self.check_PhyTime(PhyTime)
        x_vals = misc_utils.check_list_vals(x_vals)
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)

        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        if comp == 'uv':
            transform_y = lambda x: -1.*x
        else:
            transform_y = None

        fig, ax = self.UU_tensorDF.plot_line(comp,'y',x_vals,time=PhyTime,labels=labels,fig=fig,channel_half=True,
                                    transform_ydata=transform_y, ax=ax,line_kw=line_kw,**kwargs)

        sign = "-" if comp == 'uv' else ""

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        
        avg_label = cp.styleParams.AVGStyle(uu_label)
        y_label = r"$%s %s$"%(sign,avg_label)

        x_label = self.Domain.create_label(r"$y$")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # self.Domain.styleAxes(ax)
        
        ncol = cplt.get_legend_ncols(len(x_vals))
        ax.clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_Reynolds_x(self,comp,y_vals_list,y_mode='half-channel',PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        if comp not in self.UU_tensorDF.inner_index:
            comp_uu = comp[::-1]
            if not comp_uu not in self.UU_tensorDF.inner_index:
                msg = "Reynolds stress component %s not found"%comp_uu
                raise ValueError(msg)

        PhyTime = self.check_PhyTime(PhyTime)

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))

        line_kw = cplt.update_line_kw(line_kw)
        avg_label = cp.styleParams.AVGStyle(uu_label)
        if y_vals_list == 'max':
            line_kw['label'] = r"$%s_{max}$"%avg_label
            
            fig, ax = self.UU_tensorDF.plot_line_max(comp,'x',time=PhyTime,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        else:
            msg = "This method needs to be reimplemented only max can currently be used"
            raise NotImplementedError(msg)

        ax.set_ylabel(r"$%s$"%avg_label)

        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_bulk_velocity(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        bulk_velo = self.bulk_velo_calc(PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_coords = self.CoordDF['x']
        line_kw = cplt.update_line_kw(line_kw,label=r"$U_{b}$")

        ax.cplot(x_coords,bulk_velo,**line_kw)
        ax.set_ylabel(r"$U_b^*$")
        
        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax
        
    def plot_skin_friction(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)
        skin_friction = self._Cf_calc(PhyTime)
        x_coords = self.CoordDF['x']

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw,label=r"$C_f$")
        ax.cplot(x_coords,skin_friction,**line_kw)
        ax.set_ylabel(r"$C_f$")
        
        x_label = self.Domain.create_label(r"$x$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_eddy_visc(self,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        x_vals = misc_utils.check_list_vals(x_vals)
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        
        PhyTime = self.check_PhyTime(PhyTime)

        mu_t = self._eddy_visc_calc(PhyTime)

        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        fig, ax = self.UU_tensorDF.plot_line_data(mu_t,'y',x_vals,labels=labels,
                                fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        ax.set_ylabel(r"$\mu_t/\mu_0$")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)

        ax.set_xlim([-1,-0.1])

        # self.Domain.styleAxes(ax)

        ncol = cplt.get_legend_ncols(len(x_vals))
        ax.clegend(vertical=False,ncol=ncol)

        return fig, ax

class CHAPSim_AVG_tg(_AVG_base):
    def _extract_avg(self,PhyTimes,path_to_folder='.',time0=None,abs_path=True):
        """
        Instantiates an instance of the CHPSim_AVG_tg class from the result data
       
        Parameters
        ----------
        time : float or list or floats
            Physcial time to be extracted
        meta_data : CHAPSim_meta, optional
            metadata class provided to the class, by default None
        path_to_folder : str, optional
            path to the results folder, by default "."
        time0 : float, optional
            A previous time to average between, by default None
        abs_path : bool, optional
            Whether the path provided is absolute or relative, by default True
        """

        PhyTimes = misc_utils.check_list_vals(PhyTimes)

        if cp.rcParams['TEST']:
            PhyTimes=PhyTimes[-5:]
        PhyTimes.sort()

        self._meta_data = self._module._meta_class(path_to_folder,abs_path,tgpost=True)

        DF_list = self._AVG_extract(PhyTimes,path_to_folder,abs_path,self.metaDF,time0)
        times = ['%.9g' % time for time in PhyTimes]     

        self._times=times

        DF_list=self._Reverse_decomp(*DF_list)

        self.flow_AVGDF,self.PU_vectorDF,\
        self.UU_tensorDF,self.UUU_tensorDF,\
        self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
        self.DUDX2_tensorDF = DF_list

    def _hdf_extract(self,file_name,key=None):
        """
        Creates class by extracting data from the hdf5 file

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        if key is None:
            key = self.__class__.__name__

        misc_utils.check_path_exists(file_name,True)

        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')

        self.flow_AVGDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/flow_AVGDF')
        self.PU_vectorDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/PU_vectorDF')
        self.UU_tensorDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/UU_tensorDF')
        self.UUU_tensorDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/UUU_tensorDF')
        self.Velo_grad_tensorDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/Velo_grad_tensorDF')
        self.PR_Velo_grad_tensorDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/PR_Velo_grad_tensorDF')
        self.DUDX2_tensorDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/DUDX2_tensorDF')

    def _AVG_array_extract(self,PhyTime,path_to_folder,abs_path,metaDF,time0):

        AVG_info = self._extract_file_tg(PhyTime,path_to_folder,abs_path,time0)
        
        factor = metaDF['NCL1_tg']*metaDF['NCL3'] if cp.rcParams["dissipation_correction"] else 1.0
        
        flow_AVG = AVG_info[:4]
        PU_vector = AVG_info[4:7]
        UU_tensor = AVG_info[7:13]
        UUU_tensor = AVG_info[13:23]
        Velo_grad_tensor = AVG_info[23:32]
        Pr_Velo_grad_tensor = AVG_info[32:41]
        DUDX2_tensor = AVG_info[41:]*factor

        return [flow_AVG, PU_vector, UU_tensor, UUU_tensor,Velo_grad_tensor, Pr_Velo_grad_tensor, DUDX2_tensor]

    def _AVG_index_contruct(self,time):
        Phy_string = '%.9g' % time

        flow_comp_index = ['u','v','w','p']
        pu_comp_index = ['u','v','w']
        uu_comp_index = ['uu','uv','uw','vv','vw','ww']
        uuu_comp_index = ['uuu','uuv','uuw','uvv',
                          'uvw','uww','vvv','vvw','vww','www']

        dudx_index = list(itertools.product(['u','v','w'],['x','y','z']))
        dudx_comp_index = [''.join(comp) for comp in dudx_index]

        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']

        dudx_list = itertools.product(du_list,dx_list)
        dudx_comp = ["".join(dudx) for dudx in dudx_list]

        dudxdudx_list = list(itertools.product(dudx_comp,dudx_comp))
        dudxdudx_comp_index = ["".join(comp_string) for comp_string in dudxdudx_list]

        def _index_construct(index):
            return list(zip([Phy_string]*len(index),index))

        flow_index = _index_construct(flow_comp_index)
        pu_index = _index_construct(pu_comp_index)
        uu_index = _index_construct(uu_comp_index)
        uuu_index = _index_construct(uuu_comp_index)
        dudx_index = _index_construct(dudx_comp_index)
        dudxdudx_index = _index_construct(dudxdudx_comp_index)

        return [flow_index, pu_index, uu_index, uuu_index, dudx_index,dudx_index , dudxdudx_index]



    def _AVG_extract(self,PhyTimes,path_to_folder,abs_path,metaDF,time0):

        array_list = [ [] for _ in range(7)]
        index_list = [ [] for _ in range(7)]

        for time in PhyTimes:
            array_list_time = self._AVG_array_extract(time,path_to_folder,abs_path,metaDF,time0)
            index_list_time = self._AVG_index_contruct(time)
            
            for i in range(7):
                array_list[i].append(array_list_time[i])
                index_list[i].extend(index_list_time[i])

        flow_AVG = np.concatenate(array_list[0])
        PU_vector = np.concatenate(array_list[1])
        UU_tensor = np.concatenate(array_list[2])
        UUU_tensor = np.concatenate(array_list[3])
        Velo_grad_tensor = np.concatenate(array_list[4])
        Pr_Velo_grad_tensor = np.concatenate(array_list[5])
        DUDX2_tensor = np.concatenate(array_list[6])

        flow_index, pu_index, uu_index, uuu_index, dudx_index, pr_dudx_index, dudxdudx_index =  index_list

        flow_AVGDF = cd.FlowStruct1D(self._coorddata,flow_AVG,index=flow_index)
        PU_vectorDF = cd.FlowStruct1D(self._coorddata,PU_vector,index=pu_index)
        UU_tensorDF = cd.FlowStruct1D(self._coorddata,UU_tensor,index=uu_index)
        UUU_tensorDF = cd.FlowStruct1D(self._coorddata,UUU_tensor,index=uuu_index)
        Velo_grad_tensorDF = cd.FlowStruct1D(self._coorddata,Velo_grad_tensor,index=dudx_index)
        PR_Velo_grad_tensorDF = cd.FlowStruct1D(self._coorddata,Pr_Velo_grad_tensor,index=pr_dudx_index)
        DUDX2_tensorDF = cd.FlowStruct1D(self._coorddata,DUDX2_tensor,index=dudxdudx_index)

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]

    def _get_tg_boilerplate(self,path, PhyTime,abs_path):
        instant = "%0.9E" % PhyTime
        
        file_string = "DNS_perixz_AVERAGD_T" + instant + "_FLOW.D"
        
        if not abs_path:
            file_path = os.path.abspath(os.path.join(path, file_string))
        else:
            file_path = os.path.join(path, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)
        int_info = np.fromfile(file,dtype='int32',count=4)  
        
        NCL2 = int_info[0]
        NSZ = int_info[1]
        ITERG = int_info[2]
        NSTATIS = int_info[3]
        dummy_size = NCL2*NSZ
        r_info = np.fromfile(file,dtype='float64',count=3)
        
        PhyTime = r_info[0]
        
        file.close()
        return file_path, NSTATIS, PhyTime, NCL2, NSZ
    
    def _extract_file_tg(self,PhyTime,path_to_folder,abs_path,time0):
        
        full_path = misc_utils.check_paths(path_to_folder,'2_averaged_rawdata',
                                                            '2_averagd_D')


        file, NSTATIS, PhyTime, NCL2, NSZ = self._get_tg_boilerplate(full_path,
                                                                     PhyTime,
                                                                     abs_path)
        

        offset = 4*4 + 3*8
        dummy_size = NCL2*NSZ
        
        ioflowflg = self.metaDF['iDomain'] in [2,3]

        if ioflowflg and time0 is not None:
            file0, NSTATIS0,_, _, _ = self._get_tg_boilerplate(full_path,
                                                               time0,
                                                               abs_path)
            
            parallel_file = file_handler.ReadParallel([file,file0],'rb')
            AVG_info, AVG_info0 = parallel_file.read_parallel_float64(dummy_size,offset)
            
            AVG_info = (AVG_info*NSTATIS - AVG_info0*NSTATIS0)/(NSTATIS - NSTATIS0)
            
        else:
            AVG_info = np.fromfile(file,dtype='float64',offset=offset,count=dummy_size)

        AVG_info = AVG_info.reshape(NSZ,NCL2)
        
        return AVG_info

    def wall_unit_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star

        REN = self.metaDF['REN']
        
        tau_w = self.tau_calc(PhyTime)
    
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star = (nu_star/u_tau_star)/REN
        return u_tau_star, delta_v_star

    def int_thickness_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        if self.Domain.is_polar:
            U0_index = int(np.floor(self.NCL[1]*0.5))
        else:
            U0_index = 0

        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']
        U0 = U_mean[U0_index]

        theta_integrand = (U_mean/U0)*(1 - U_mean/U0)
        delta_integrand = 1 - U_mean/U0

        mom_thickness = 0.5*integrate_simps(theta_integrand,y_coords)
        disp_thickness = 0.5*integrate_simps(delta_integrand,y_coords)

        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor

    def tau_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']

        mu_star = 1.0
        return -mu_star*(u_velo[-1]-0.0)/(ycoords[-1]-1.0)

    def bulk_velo_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        u_velo = self.flow_AVGDF[PhyTime,'u'].squeeze()
        ycoords = self.CoordDF['y']

        bulk_velo = 0.5*integrate_simps(u_velo,ycoords)
            
        return bulk_velo

    def y_coord_index_norm(self,coord_list,avg_time=None,inst_time=None,mode='half_channel'):
        
        if avg_time is None:
            avg_time = max(self.times)
        else:
            avg_time = self.check_PhyTime(avg_time)

        avg_time = self.check_PhyTime(avg_time)

        if mode=='half_channel':
            norm_distance=np.ones((self.NCL[0]))
        elif mode == 'disp_thickness':
            norm_distance, _,_ = self.int_thickness_calc(avg_time)
        elif mode == 'mom_thickness':
            _, norm_distance, _ = self.int_thickness_calc(avg_time)
        elif mode == 'wall':
            _, norm_distance = self.wall_unit_calc(avg_time)
        else:
            msg = ("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
                    " or 'wall. Value used was %s\n"%mode)
            raise ValueError(msg)
        

        y_indices = []
        if self.Domain.is_polar:
            norm_coordDF = -1*(self.CoordDF-1)/norm_distance
        else:
            norm_coordDF = (1-abs(self.CoordDF))/norm_distance
        y_index = indexing.coord_index_calc(norm_coordDF,'y', coord_list)
        y_indices.append(y_index)

        return y_indices

    def ycoords_from_coords(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if avg_time is None:
            avg_time = max(self.times)
        else:
            avg_time = self.check_PhyTime(avg_time)

        if mode=='half_channel':
            norm_distance=np.ones((self.NCL[0]))
        elif mode == 'disp_thickness':
            norm_distance, _,_ = self.int_thickness_calc(avg_time)
        elif mode == 'mom_thickness':
            _, norm_distance, _ = self.int_thickness_calc(avg_time)
        elif mode == 'wall':
            _, norm_distance = self.wall_unit_calc(avg_time)
        else:
            msg = ("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
                        " or 'wall. Value used was %s\n"%mode)
            raise ValueError(msg)
        
        index = self.y_coord_index_norm(coord_list,avg_time=avg_time,mode=mode)[0]

        true_ycoords = []
        if self.Domain.is_polar:
            norm_coordDF = -1*(self.CoordDF-1)/norm_distance
        else:
            norm_coordDF = (1-abs(self.CoordDF))/norm_distance
        true_ycoords.append(norm_coordDF['y'][index])

        return true_ycoords

    def ycoords_from_norm_coords(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if avg_time is None:
            avg_time = max(self.times)
        else:
            avg_time = self.check_PhyTime(avg_time)
            
        index = self.y_coord_index_norm(coord_list,avg_time=avg_time,mode=mode)[0]
        true_ycoords = []
        true_ycoords.append(self.CoordDF['y'][index])
    
        return true_ycoords

    def _Cf_calc(self,PhyTime):
        PhyTime = self.check_PhyTime(PhyTime)

        rho_star = 1.0
        REN = self.metaDF['REN']
        tau_star = self.tau_calc(PhyTime)
        bulk_velo = self.bulk_velo_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star

        return skin_friction

    def _eddy_visc_calc(self,PhyTime):
        uv = self.UU_tensorDF[PhyTime,'uv']
        dUdy = self.Velo_grad_tensorDF[PhyTime,'uy']
        dVdx = self.Velo_grad_tensorDF[PhyTime,'vx']
        REN = self.metaDF['REN']

        mu_t = -uv*REN/(dUdy + dVdx)
        return mu_t

    def plot_mean_flow(self,comp,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)

        fig, ax = self.flow_AVGDF.plot_line(comp,time=PhyTime,
                            fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        x_label = self.Domain.create_label(r"$y$")
        y_label = self.Domain.create_label(r"$\bar{u}$")
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # self.Domain.styleAxes(ax)

        return fig, ax
    
    def plot_flow_wall_units(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)
        fig, ax = self.flow_AVGDF.plot_line('u',channel_half=True,time=PhyTime,
                            fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        if self.Domain.is_polar:
            flip_axis = lambda x: -1*(x - 1.0)
            ax.apply_func('x',flip_axis)
        else:
            ax.shift_xaxis(1.0)

        u_tau, delta_v = self.wall_unit_calc(PhyTime)

        ax.set_xscale('log')
        ax.normalise('y',u_tau)
        ax.normalise('x',delta_v)

        labels = [l.get_label() for l in ax.get_lines()]
        if not r"$\bar{u}^+=y^+$" in labels:
            uplus_max = ax.get_ylim()[1]
            u_plus_array = np.linspace(0,uplus_max,100)

            ax.cplot(u_plus_array,u_plus_array,label=r"$\bar{u}^+=y^+$",color='r',linestyle='--')

        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")

        return fig, ax

    def plot_Reynolds(self,comp,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        if comp not in self.UU_tensorDF.inner_index:
            comp = comp[::-1]
            if not comp not in self.UU_tensorDF.inner_index:
                msg = "Reynolds stress component %s not found"%comp
                raise ValueError(msg) 

        PhyTime = self.check_PhyTime(PhyTime)

        if comp == 'uv':
            transform_y = lambda x: -1.*x
        else:
            transform_y = None

        fig, ax = self.UU_tensorDF.plot_line(comp,time=PhyTime,fig=fig,channel_half=True,
                                    transform_ydata=transform_y,
                                    ax=ax,line_kw=line_kw,**kwargs)
    
        sign = "-" if comp == 'uv' else ""

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = cp.styleParams.AVGStyle(uu_label)

        y_label = r"$%s %s$"%(sign,avg_label)
        x_label = self.Domain.create_label(r"$y$")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # self.Domain.styleAxes(ax)

        return fig, ax

    def plot_eddy_visc(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)

        mu_t = self._eddy_visc_calc(PhyTime)

        fig, ax = self.UU_tensorDF.plot_line_data(mu_t,fig=fig,ax=ax,
                                                  line_kw=line_kw,**kwargs)

        ax.set_ylabel(r"$\mu_t/\mu_0$")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)

        ax.set_xlim([-1,-0.1])

        # self.Domain.styleAxes(ax)

        return fig, ax

class AVG_tg_array(postArray):
    _available_methods = ['plot_mean_flow',
                          'plot_flow_wall_units',
                          'plot_Reynolds',
                          'plot_eddy_visc']
    def __init__(self,*args,from_file=False,**kwargs):

        if isinstance(args[0],dict) or from_file:
            super().__init__(*args,from_file=from_file,**kwargs)
        else:
            if not 'paths_to_folder' in kwargs.keys():
                msg = ("Paths to folder are not provided a data dict"
                        " or the from_hdf classmethod must be used")
                raise ValueError(msg)

            if not 'labels' in kwargs.keys():
                msg = ("labels must be provideed")
                raise ValueError(msg)


            paths_to_folders = kwargs['paths_to_folders']
            labels = kwargs['labels']
            times = kwargs.get('times',None)
            time0 = kwargs.get('times0',None)

            self._data_dict = {}
            for i, path in enumerate(paths_to_folders):
                if times is None:
                    time = misc_utils.max_time_calc(path)
                else:
                    assert len(times) == len(paths_to_folders)
                    time = times[i]

                self._data_dict[labels[i]] = self._module._avg_tg_class(time,path_to_folder=path,time0=time0)  


    def __getattr__(self,attr):
        attr_list = ['_data_dict']
        if attr in attr_list:
            msg = "This class' __getattr__ method invoked before all necessary attributes set"
            raise AttributeError(msg)

        if attr not in self._available_methods:
            msg = "This class' __getattr__ method can only be used with plot methods"
            raise AttributeError(msg)

        if not all([hasattr(data,attr) for data in self._data_dict.values()]):
            msg = f"Not all average classes have the plotting method {attr}"
            raise AttributeError(msg)

        def _plotting_method(*args,**kwargs):
            if 'line_kw' in kwargs.keys():
                if 'label' in kwargs['line_kw']:
                    msg = ("Label already present in line_kw keyword,"
                            " this will be overridden")
                    warnings.warn(msg)

            for label, data in self._data_dict.items():
                _local_plot = getattr(data,attr)
                if 'line_kw' in kwargs.keys():
                    kwargs['line_kw']['label'] = label
                else:
                    kwargs['line_kw'] = {'label' : label}
                fig, ax = _local_plot(*args,**kwargs)

            return fig, ax

        return _plotting_method

class CHAPSim_AVG_temp(_AVG_developing,CHAPSim_AVG_tg):

    @classmethod
    def with_phase_average(cls,paths_to_folders,shift_times,time0=None,abs_path=True,*args,**kwargs):
        if not isinstance(paths_to_folders,(tuple,list)):
            msg = f"To use this method, path_to_folder must be a tuple or a list not a {type(paths_to_folders)}"
            raise TypeError(msg)

        if len(paths_to_folders) != len(shift_times):
            msg= f"The length of shift times and paths_to_folder must be the same"
            raise ValueError(msg)

        
        for (i,path),val in zip(enumerate(paths_to_folders),shift_times):
            if i ==0:
                PhyTimes = [x+val for x in misc_utils.time_extract(path,abs_path)]
                if time0 is not None:
                    PhyTimes = list(filter(lambda x: x > (time0+val), PhyTimes))

            else:
                times = [x+val for x in misc_utils.time_extract(path,abs_path)]
                PhyTimes = sorted(set(PhyTimes).intersection(times))

        avg_data = None

        for (i,path),val in zip(enumerate(paths_to_folders),shift_times):
            local_PhyTimes = sorted([time - val for time in PhyTimes])
            local_time0 = None if time0 is None else time0 -val
            avg_data_local = cls(path,local_time0,PhyTimes=local_PhyTimes,abs_path=abs_path,**kwargs)

            avg_data_local._shift_times(val)

            coe1 = i/(i+1)
            coe2 = 1/(i+1)

            if i == 0:
                avg_data = avg_data_local
            else:
                avg_data._incremental_ensemble_avg(avg_data_local,coe1,coe2)

        return avg_data

    def _del_times(self,PhyTimes):

        for time in PhyTimes:
            self.flow_AVGDF.remove_time(time)
            self.PU_vectorDF.remove_time(time)
            self.UU_tensorDF.remove_time(time)
            self.UUU_tensorDF.remove_time(time)
            self.Velo_grad_tensorDF.remove_time(time)
            self.PR_Velo_grad_tensorDF.remove_time(time)
            self.DUDX2_tensorDF.remove_time(time)


    def _shift_times(self,time):

        self.flow_AVGDF = self.flow_AVGDF.shift_times(time)
        self.PU_vectorDF = self.PU_vectorDF.shift_times(time)
        self.UU_tensorDF = self.UU_tensorDF.shift_times(time)
        self.UUU_tensorDF = self.UUU_tensorDF.shift_times(time)
        self.Velo_grad_tensorDF = self.Velo_grad_tensorDF.shift_times(time)
        self.PR_Velo_grad_tensorDF = self.PR_Velo_grad_tensorDF.shift_times(time)
        self.DUDX2_tensorDF = self.DUDX2_tensorDF.shift_times(time)
        
    def _incremental_ensemble_avg(self,other_avg,coe1,coe2):
        if not isinstance(other_avg,self.__class__):
            msg = "The other_avg object must be of the same type"
            raise TypeError(msg)

        self.flow_AVGDF = coe1*self.flow_AVGDF + coe2*other_avg.flow_AVGDF
        self.PU_vectorDF = coe1*self.PU_vectorDF + coe2*other_avg.PU_vectorDF
        self.UU_tensorDF = coe1*self.UU_tensorDF + coe2*other_avg.UU_tensorDF
        self.UUU_tensorDF = coe1*self.UUU_tensorDF + coe2*other_avg.UUU_tensorDF
        self.Velo_grad_tensorDF = coe1*self.Velo_grad_tensorDF + coe2*other_avg.Velo_grad_tensorDF
        self.PR_Velo_grad_tensorDF = coe1*self.PR_Velo_grad_tensorDF + coe2*other_avg.PR_Velo_grad_tensorDF
        self.DUDX2_tensorDF = coe1*self.DUDX2_tensorDF + coe2*other_avg.DUDX2_tensorDF

    @property
    def _shape_devel(self):
        shape = (super().shape[0],
                len(self.times))
        return shape
        
    def _extract_avg(self,path_to_folder='.',time0=None,abs_path=True,PhyTimes=None,*args,**kwargs):
        """
        Instantiates an instance of the CHPSim_AVG_tg class from the result data
       
        Parameters
        ----------
        path_to_folder : str, optional
            path to the results folder, by default "."
        time0 : float, optional
            A previous time to average between, by default None
        abs_path : bool, optional
            Whether the path provided is absolute or relative, by default True
        """

        times = misc_utils.time_extract(path_to_folder,abs_path)

        if PhyTimes is not None:
            PhyTimes = misc_utils.check_list_vals(PhyTimes)
            times = list(set(times).intersection(PhyTimes))      

        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        return super()._extract_avg(times,path_to_folder=path_to_folder,time0=None,abs_path=abs_path,*args,**kwargs)

    def _hdf_extract(self,file_name,key=None):
        """
        Creates class by extracting data from the hdf5 file

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        if key is None:
            key = self.__class__.__name__

        misc_utils.check_path_exists(file_name,True)

        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')

        self.flow_AVGDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/flow_AVGDF')
        self.PU_vectorDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/PU_vectorDF')
        self.UU_tensorDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/UU_tensorDF')
        self.UUU_tensorDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/UUU_tensorDF')
        self.Velo_grad_tensorDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/Velo_grad_tensorDF')
        self.PR_Velo_grad_tensorDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/PR_Velo_grad_tensorDF')
        self.DUDX2_tensorDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/DUDX2_tensorDF')

    # def _AVG_array_extract(self,PhyTime,path_to_folder,abs_path,metaDF,time0):

    #     AVG_info, NSTATIS1, _, _ = self._extract_file_tg(PhyTime,path_to_folder,abs_path)
        
    #     factor = metaDF['NCL1_tg']*metaDF['NCL3'] if cp.rcParams["dissipation_correction"] else 1.0
    #     ioflowflg = self.metaDF['iDomain'] in [2,3]

    #     if ioflowflg and time0:
    #         AVG_info0, NSTATIS0, _, _ = self._extract_file_tg(time0,path_to_folder,abs_path)
    #         AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)

    #     flow_AVG = AVG_info[:4]
    #     PU_vector = AVG_info[4:7]
    #     UU_tensor = AVG_info[7:13]
    #     UUU_tensor = AVG_info[13:23]
    #     Velo_grad_tensor = AVG_info[23:32]
    #     Pr_Velo_grad_tensor = AVG_info[32:41]
    #     DUDX2_tensor = AVG_info[41:]*factor

    #     return [flow_AVG, PU_vector, UU_tensor, UUU_tensor,Velo_grad_tensor, Pr_Velo_grad_tensor, DUDX2_tensor]

    # def _AVG_index_contruct(self,time):
    #     Phy_string = '%.9g' % time

    #     flow_comp_index = ['u','v','w','P']
    #     pu_comp_index = ['u','v','w']
    #     uu_comp_index = ['uu','uv','uw','vv','vw','ww']
    #     uuu_comp_index = ['uuu','uuv','uuw','uvv',
    #                       'uvw','uww','vvv','vvw','vww','www']

    #     du_list = ['du','dv','dw']
    #     dx_list = ['dx','dy','dz']

    #     dudx_list = itertools.product(du_list,dx_list)
    #     dudx_comp_index = ["".join(dudx) for dudx in dudx_list]

    #     dudxdudx_list = list(itertools.product(dudx_list,dudx_list))
    #     dudxdudx_comp_index = ["".join(comp_string) for comp_string in dudxdudx_list]

    #     def _index_construct(index):
    #         return list(zip([Phy_string]*len(index),index))

    #     flow_index = _index_construct(flow_comp_index)
    #     pu_index = _index_construct(pu_comp_index)
    #     uu_index = _index_construct(uu_comp_index)
    #     uuu_index = _index_construct(uuu_comp_index)
    #     dudx_index = _index_construct(dudx_comp_index)
    #     dudxdudx_index = _index_construct(dudxdudx_comp_index)

    #     return [flow_index, pu_index, uu_index, uuu_index, dudx_index,dudx_index , dudxdudx_index]


    def _AVG_extract(self,PhyTimes,path_to_folder,abs_path,metaDF,time0):

        array_list = [ [] for _ in range(7)]
        index_list = [ [] for _ in range(7)]

        for time in PhyTimes:
            array_list_time = self._AVG_array_extract(time,path_to_folder,abs_path,metaDF,time0)
            index_list_time = self._AVG_index_contruct(time)
            
            for i in range(7):
                array_list[i].append(array_list_time[i])
                index_list[i].extend(index_list_time[i])

        flow_AVG = np.concatenate(array_list[0])
        PU_vector = np.concatenate(array_list[1])
        UU_tensor = np.concatenate(array_list[2])
        UUU_tensor = np.concatenate(array_list[3])
        Velo_grad_tensor = np.concatenate(array_list[4])
        Pr_Velo_grad_tensor = np.concatenate(array_list[5])
        DUDX2_tensor = np.concatenate(array_list[6])

        flow_index, pu_index, uu_index, uuu_index, dudx_index, pr_dudx_index, dudxdudx_index =  index_list


        flow_AVGDF = cd.FlowStruct1D_time(self._coorddata,flow_AVG,index=flow_index)
        PU_vectorDF = cd.FlowStruct1D_time(self._coorddata,PU_vector,index=pu_index)
        UU_tensorDF = cd.FlowStruct1D_time(self._coorddata,UU_tensor,index=uu_index)
        UUU_tensorDF = cd.FlowStruct1D_time(self._coorddata,UUU_tensor,index=uuu_index)
        Velo_grad_tensorDF = cd.FlowStruct1D_time(self._coorddata,Velo_grad_tensor,index=dudx_index)
        PR_Velo_grad_tensorDF = cd.FlowStruct1D_time(self._coorddata,Pr_Velo_grad_tensor,index=pr_dudx_index)
        DUDX2_tensorDF = cd.FlowStruct1D_time(self._coorddata,DUDX2_tensor,index=dudxdudx_index)

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]

    def _return_index(self,PhyTime):
        if not isinstance(PhyTime,str):
            PhyTime = "{:.9g}".format(PhyTime)

        if PhyTime not in self.get_times():
            raise ValueError("time %s must be in times"% PhyTime)
        for i in range(len(self.get_times())):
            if PhyTime==self.get_times()[i]:
                return i

    def _return_xaxis(self):
        return self.times

    def wall_unit_calc(self,PhyTime=None):
        """
        returns arrays for the friction velocity and viscous lengthscale


        Returns
        -------
        %(ndarray)s
            friction velocity array
        %(ndarray)s
            viscous lengthscale array
        """
        if PhyTime is None:
            return self._wall_unit_calc(None)
        else:
            return super().wall_unit_calc(PhyTime)

    def int_thickness_calc(self,PhyTime=None):
        """
        Calculates the integral thicknesses and shape factor 

        Returns
        -------
        %(ndarray)s:
            Displacement thickness
        %(ndarray)s:
            Momentum thickness
        %(ndarray)s:
            Shape factor
        """

        if PhyTime is None:
            return self._int_thickness_calc(None)
        else:
            return super().int_thickness_calc(PhyTime)

    def y_coord_index_norm(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if inst_time is not None and avg_time is None:
            avg_time = inst_time
        return super().y_coord_index_norm(coord_list,avg_time=avg_time,mode=mode)

    def ycoords_from_coords(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if inst_time is not None and avg_time is None:
            avg_time = inst_time
        return super().ycoords_from_coords(coord_list,avg_time=avg_time,mode=mode)

    
    def ycoords_from_norm_coords(self,coord_list,avg_time=None,inst_time=None,x_vals=None,mode='half_channel'):
        if inst_time is not None and avg_time is None:
            avg_time = inst_time
        return super().ycoords_from_norm_coords(coord_list,avg_time=avg_time,mode=mode)


    def plot_shape_factor(self,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the shape factor from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        _,_, H = self.int_thickness_calc(None)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times

        line_kw = cplt.update_line_kw(line_kw,label=r"$H$")
        ax.cplot(times,H,**line_kw)

        ax.set_xlabel(r"$%s$"% cp.styleParams.timeStyle )
        ax.set_ylabel(r"$H$")
        
        return fig, ax

    def plot_disp_thickness(self,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the displacement thickness from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        delta,_, _ = self.int_thickness_calc(None)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times

        line_kw = cplt.update_line_kw(line_kw,label=r"$\delta^*$")
        ax.cplot(times,delta,**line_kw)

        time_label = cp.styleParams.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        ax.set_ylabel(r"$\delta^*$")
        
        return fig, ax

    def plot_mom_thickness(self,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the momentum thickness from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        _,theta, _ = self.int_thickness_calc(None)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times

        line_kw = cplt.update_line_kw(line_kw,label=r"$\theta$")
        ax.cplot(times,theta,**line_kw)

        time_label = cp.styleParams.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        ax.set_ylabel(r"$\theta$")
        
        return fig, ax

    def bulk_velo_calc(self,PhyTime=None):
        """
        Method to calculate the bulk velocity against the streamwise coordinate
        
        Returns
        -------
        %(ndarray)s
            array containing the bulk velocity
        """
        if PhyTime is None:
            return self._bulk_velo_calc(None)
        else:
            return super().bulk_velo_calc(PhyTime)

    def plot_bulk_velocity(self,fig=None,ax=None,line_kw=None,**kwargs):

        bulk_velo = self.bulk_velo_calc(None)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        times = self.times
        line_kw = cplt.update_line_kw(line_kw,label=r"$U_{b0}$")

        ax.cplot(times,bulk_velo,**line_kw)
        ax.set_ylabel(r"$U_b^*$")
        
        time_label = cp.styleParams.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        return fig, ax

    def tau_calc(self,PhyTime=None):
        """
        method to return the wall shear stress array

        Returns
        -------
        %(ndarray)s
            Wall shear stress array 
        """
        if PhyTime is None:
            return self._tau_calc(None)
        else:
            return super().tau_calc(PhyTime)

    def plot_skin_friction(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        skin_friction = self._Cf_calc(PhyTime)
        times = self.times

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw,label=r"$C_f$")
        ax.cplot(times,skin_friction,**line_kw)

        ax.set_ylabel(r"$C_f$")

        time_label = cp.styleParams.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        return fig, ax

    def plot_mean_flow(self,comp,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):
        line_kw = cplt.update_line_kw(line_kw)

        for time in PhyTimes:

            time_label = cp.styleParams.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_mean_flow(comp,time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        ncol = cplt.get_legend_ncols(len(PhyTimes))
        ax.clegend(vertical=False,ncol=ncol)
        return fig, ax

    def plot_flow_wall_units(self,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):
        line_kw = cplt.update_line_kw(line_kw)

        for time in PhyTimes:
            
            time_label = cp.styleParams.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_flow_wall_units(time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        ncol = cplt.get_legend_ncols(len(PhyTimes))
        ax.clegend(vertical=False,ncol=ncol)
        return fig, ax

    def plot_Reynolds(self,comp,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):
        line_kw = cplt.update_line_kw(line_kw)
        for time in PhyTimes:
            
            time_label = cp.styleParams.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_Reynolds(comp,PhyTime=time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        ncol = cplt.get_legend_ncols(len(PhyTimes))
        ax.clegend(vertical=False,ncol=ncol)
        return fig, ax

    def plot_eddy_visc(self,PhyTimes,fig=None,ax=None,line_kw=None,**kwargs):

        line_kw = cplt.update_line_kw(line_kw)
        for time in PhyTimes:
            time_label = cp.styleParams.timeStyle
            line_kw['label'] = r"$%s = %.3g$"%(time_label,time)

            fig, ax = super().plot_eddy_visc(time,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        ncol = cplt.get_legend_ncols(len(PhyTimes))
        ax.clegend(vertical=False,ncol=ncol)
        return fig, ax

    def plot_Reynolds_x(self,comp,y_vals_list,y_mode='half-channel',fig=None,ax=None,line_kw=None,**kwargs):
        
        if comp not in self.UU_tensorDF.inner_index:
            comp = comp[::-1]
            if not comp not in self.UU_tensorDF.inner_index:
                msg = "Reynolds stress component %s not found"%comp
                raise ValueError(msg)

        uu_label = self.Domain.create_label(r"%s'%s'"%tuple(comp))
        avg_label = cp.styleParams.AVGStyle(uu_label)

        line_kw = cplt.update_line_kw(line_kw)
        if y_vals_list == 'max':
            line_kw['label'] = r"$%s_{max}$"%avg_label
            
            fig, ax = self.UU_tensorDF.plot_line_time_max(comp,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        else:
            msg = "This method needs to be reimplemented only max can currently be used"
            raise NotImplementedError(msg)

        ax.set_ylabel(r"$%s$"%avg_label)

            
        time_label = cp.styleParams.timeStyle
        ax.set_xlabel(r"$%s$"%time_label)

        return fig, ax

    

