"""
# _instant.py
File contains the implementation of the classes to processes the 
instantaneous results from the CHAPSim DNS solver 
"""
from abc import abstractmethod, ABC, abstractproperty
import numpy as np
import matplotlib as mpl

import logging
import os
import warnings
import gc
import copy
import itertools
from CHAPSim_post.utils import docstring, gradient, indexing, misc_utils, parallel
from CHAPSim_post import rcParams
import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd

from CHAPSim_post._libs import file_handler, post

from ._average import CHAPSim_AVG_io,CHAPSim_AVG_temp, CHAPSim_AVG_tg
from ._common import Common, classproperty

_avg_io_class = CHAPSim_AVG_io
_avg_temp_class = CHAPSim_AVG_temp
_avg_tg_class = CHAPSim_AVG_tg

from ._meta import CHAPSim_meta
_meta_class=CHAPSim_meta

try:
    import cupy as cnpy
    _cupy_avail = True
except:
    pass

logger = logging.getLogger(__name__)
class _Inst_base(Common,ABC):
    """
    ## CHAPSim_Inst
    This is a module for processing and visualising instantaneous data from CHAPSim
    """
    
    _default_comps = ('u', 'v', 'w','p')
    _update_time = True
    
    @docstring.copy_fromattr("_inst_extract")
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        if not fromfile:
            self._inst_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)
    
    @classmethod
    def update_time(cls,val):
        cls._update_time = val
    
    @classmethod
    def update_default_comps(cls,comps):
        if not all(isinstance(x, str) for x in comps):
            raise TypeError("All components must be characters")

        if not all(len(x) == 1 for x in comps):
            raise ValueError("All components must be length 1")
        
        cls._default_comps = tuple(comps)
        
    @abstractproperty
    def _avg_class(self):
        pass
    

    def _inst_extract(self,time,path_to_folder='.',avg_data=None, comps=None, abs_path = True, time0=None):
        """
        Instantiates CHAPSim_Inst by extracting data from the 
        CHAPSim rawdata results folder

        Parameters
        ----------
        time : int, float, or list
            Physical times to extract from results folder
        meta_data : CHAPSim_meta, optional
            a metadata instance, if not provided, it will be extracted, by default None
        path_to_folder : str, optional
            Path to the results folder, by default '.'
        abs_path : bool, optional
            Whether the path provided is an absolute path, by default True
        tgpost : bool, optional
            Whether the turbulence generator or spatially developing 
            region are processed, by default False

        """

        
        self._meta_data = self._module._meta_class(path_to_folder,abs_path,self._tgpost)

        if comps is None:
            self._comps = list(self._default_comps)
        else:
            self._comps = list(comps)

        time = misc_utils.check_list_vals(time)

        for PhyTime in time:
            if not hasattr(self, 'InstDF'):
                self.InstDF = self._flow_extract(PhyTime,path_to_folder,abs_path)
            else: #Variable already exists
                local_DF = self._flow_extract(PhyTime,path_to_folder,abs_path)
                # concat_DF = [self.InstDF,local_DF]
                self.InstDF.concat(local_DF)

        self._avg_data = self._create_avg_data(path_to_folder,abs_path,time0,avg_data=avg_data)

        self._update_instant()
        
    def _update_instant(self):
        pass
    
    @abstractmethod
    def _create_avg_data(self,path_to_folder,abs_path,time0):
        pass

    @classmethod
    def from_hdf(cls,file_name,key=None):
        """
        Creates an instance of CHAPSim_inst by extracting an existing 
        saved instance from hdf file

        Parameters
        ----------
        file_name : str
            File path to existing hdf5 file
        key : str, optional
            path-like, hdf5 key to access the data within the file,
             by default None (class name)

        """
        return cls(file_name,fromfile=True,key=key)
    
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')

        self.InstDF = cd.FlowStruct3D.from_hdf(file_name,key=key+'/InstDF')#pd.read_hdf(file_name,base_name+'/InstDF').data(shape)
        self._avg_data = self._avg_class.from_hdf(file_name,key=key+'/avg_data')
        
    @property
    def shape(self):
        inst_index = self.InstDF.index[0]
        return self.InstDF[inst_index].shape

    def save_hdf(self,file_name,write_mode,key=None):
        """
        Saves the instance of the class to hdf5 file

        Parameters
        ----------
        file_name : str
            File path to existing hdf5 file
        write_mode : str
            The write mode for example append "a" or "w" see documentation for 
            h5py.File
        key : str, optional
            path-like, hdf5 key to access the data within the file,
             by default None (class name)
        """

        if key is None:
            key = self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self._meta_data.save_hdf(file_name,'a',key=key+'/meta_data')
        self.InstDF.to_hdf(file_name,key=key+'/InstDF',mode='a')
        self._avg_data.save_hdf(file_name,'a',key=key+'/avg_data')

    def _file_extract(self,file_list):
        #A list of all the relevant files for this timestep                           
        open_list =[]
        #opening all the relevant files and placing them in a list
        for file in file_list:
            file_temp = open(file,'rb')
            open_list.append(file_temp)
        
        comp_size = len(self._comps)
        #allocating arrays
        int_info=np.zeros((comp_size,4))
        r_info = np.zeros((comp_size,3))

        i=0
        #reading metadata from file
        for file in open_list:
            int_info[i]=np.fromfile(file,dtype='int32',count=4)
            r_info[i]=np.fromfile(file,dtype='float64',count=3)
            i+=1

        PhyTime=r_info[0,0]
        NCL1=int(int_info[0,0])
        NCL2=int(int_info[0,1])
        NCL3=int(int_info[0,2])

        dummy_size=NCL1*NCL2*NCL3

        parallel_file = file_handler.ReadParallel(file_list,'rb')
        
        # parallelExec = parallel.ParallelConcurrent()
        offset = 4*4+3*8
        result = parallel_file.read_parallel_float64(dummy_size,offset)
        # result = parallelExec.map_async(np.fromfile,file_list,dtype = 'float64',offset=offset,count=dummy_size)

        for open_file in open_list:
            open_file.close()

        return result, NCL1, NCL2, NCL3, PhyTime



    def _flow_extract(self,Time_input,path_to_folder,abs_path):
        """ Extract velocity and pressure from the instantanous files """
        instant = "%0.9E" % Time_input
        file_folder = "1_instant_D"

        if self._tgpost:
            file_string = "DNS_perixz_INSTANT_T" + instant
        else:
            file_string = "DNS_perioz_INSTANT_T" + instant
        
        veloVector = ['_%s'%comp.upper() for comp in self._comps]
        file_ext = ".D"
        
        full_path = misc_utils.check_paths(path_to_folder,'1_instant_rawdata',
                                                            '1_instant_D')
        file_list=[]
        for velo in veloVector:
            if not abs_path:
                file_list.append(os.path.abspath(os.path.join(full_path, file_string + velo + file_ext)))
            else:
                file_list.append(os.path.join(full_path, file_string + velo + file_ext))
        

        flow_info, NCL1, NCL2, NCL3, PhyTime = self._file_extract(file_list)
        
        #Reshaping and interpolating flow data so that it is centred
        comp_size = len(self._comps)
        flow_info=flow_info.reshape((comp_size,NCL3,NCL2,NCL1))
        flow_info = self._velo_interp(flow_info,NCL3,NCL2,NCL1)
        
        gc.collect()

        Phy_string = '%.9g' % PhyTime

        # creating datastruct index
        index = [[Phy_string]*len(self._comps),self._comps]

        # creating datastruct so that data can be easily accessible elsewhere
        Instant_DF = cd.FlowStruct3D(self._coorddata,flow_info,index=index,copy=False)# pd.DataFrame(flow_info1,index=index)

        # for file in open_list:
        #     file.close()
            
        return Instant_DF

    def _velo_interp(self,flow_info,NCL3, NCL2, NCL1):
        """ Convert the velocity info so that it is defined at the cell centre """
       
        #This doesn't interpolate pressure as it is already located at the cell centre
        #interpolation reduces u extent, therefore to maintain size, V, W reduced by 1
        
        comp_size = len(self._comps)
        flow_interp = np.zeros((comp_size,NCL3,NCL2,NCL1-1))
        
        for i, comp in enumerate(self._comps):
            if comp in ('u','v','w'):
                dim = ord('w') - ord(comp)
                flow_interp[i] = post.velo_interp3D(flow_info[i],
                                                    NCL1,
                                                    NCL2,
                                                    NCL3,
                                                    dim) #0.5*(flow_info[i,:,:,:-1] + flow_info[i,:,:,1:])
            else:
                flow_interp[i] = flow_info[i,:,:,:-1]

        return flow_interp

    def check_PhyTime(self,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = f"PhyTime provided ({PhyTime}) is not in the {self.__class__.__name__} datastruct, recovery impossible"
        
        err = ValueError(err_msg)
        warn = UserWarning(warn_msg)
        return self.InstDF.check_times(PhyTime,err,warn_msg)

    @docstring.sub
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime=None,y_mode='wall',fig=None,ax=None,contour_kw=None,**kwargs):
        """
        Plot a contour along a given plane at different locations in the third axis

        Parameters
        ----------
        comp : str
            Component of the instantaneous data to be extracted e.g. "u" for the
            streamwise velocity

        axis_vals : int, float (or list of them)
            locations in the third axis to be plotted
        avg_data : CHAPSim_AVG
            Used to aid plotting certain locations using in the y direction 
            if wall units are used for example
        plane : str, optional
            Plane for the contour plot for example "xz" or "rtheta" (for pipes),
             by default 'xz'
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        y_mode : str, optional
            Only relevant if the xz plane is being used. The y value can be selected using a 
            number of different normalisations, by default 'wall'
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        pcolor_kw : dict, optional
            Arguments passed to the pcolormesh function, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        axis_vals = misc_utils.check_list_vals(axis_vals)

        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.InstDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = self._avg_data.ycoords_from_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
            int_vals = self._avg_data.ycoords_from_norm_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
        else:
            int_vals = axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)
        
        x_size, z_size = self.InstDF.get_unit_figsize(plane)
        figsize=[x_size*len(axis_vals),z_size]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i,val in enumerate(int_vals):
            fig, ax1 = self.InstDF.plot_contour(comp,plane,val,time=PhyTime,fig=fig,ax=ax[i],contour_kw=contour_kw)

            xlabel = self.Domain.create_label(r"$%s$"%plane[0])
            ylabel = self.Domain.create_label(r"$%s$"%plane[1])

            ax[i].axes.set_xlabel(xlabel)
            ax[i].axes.set_ylabel(ylabel)

            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
            
            cbar=fig.colorbar(ax1,ax=ax[i])
            cbar.set_label(r"$%s^\prime$"%comp)

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax

    @docstring.sub
    def plot_vector(self,plane,axis_vals,avg_data,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        """
        Create vector plot of a plane of the instantaneous flow

        Parameters
        ----------
        plane : str, optional
            Plane for the contour plot for example "xz" or "rtheta" (for pipes),
            by default 'xz'
        axis_vals : int, float (or list of them)
            locations in the third axis to be plotted
        avg_data : CHAPSim_AVG
            Used to aid plotting certain locations using in the y direction 
            if wall units are used for example
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        y_mode : str, optional
            Only relevant if the xz plane is being used. The y value can be selected using a 
            number of different normalisations, by default 'wall'
        spacing : tuple, optional
            [description], by default (1,1)
        scaling : int, optional
            [description], by default 1
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        quiver_kw : dict, optional
            Argument passed to matplotlib quiver plot, by default None
        
        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        axis_vals = misc_utils.check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.InstDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = self._avg_data.ycoords_from_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
            int_vals = self._avg_data.ycoords_from_norm_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
        else:
            int_vals = axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)

        x_size, z_size = self.InstDF.get_unit_figsize(plane)
        figsize=[x_size*len(axis_vals),z_size]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i, val in enumerate(int_vals):
            fig, ax[i] = self.InstDF.plot_vector(plane,val,time=PhyTime,spacing=spacing,scaling=scaling,
                                                    fig=fig,ax=ax[i],quiver_kw=quiver_kw)
            xlabel = self.Domain.create_label(r"$%s$"%plane[0])
            ylabel = self.Domain.create_label(r"$%s$"%plane[1])

            ax[i].axes.set_xlabel(xlabel)
            ax[i].axes.set_ylabel(ylabel)

            ax[i].axes.set_title(r"$%s = %.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax[i].axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
    
    @docstring.sub
    def lambda2_calc(self,PhyTime=None):
        """
        Calculation of lambda to visualise vortex cores

        Parameters
        ----------
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        x_start_index : int, optional
            streamwise location to start to calculation, by default None
        x_end_index : int, optional
            streamwise location to end to calculation, by default None
        y_index : int, optional
            First y_index of the mesh to be calculated, by default None

        Returns
        -------
        %(ndarray)s
            Array of the lambda2 calculation
        """
        
        PhyTime = self.check_PhyTime(PhyTime)

        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
        
        
        velo_grad = np.zeros((*self.shape,9))
        comp_iter = itertools.product(velo_list,coord_list)
        
        for i, (u, x) in enumerate(comp_iter):
            velo_field = self.InstDF[PhyTime,u]

            velo_grad[:,:,:,i] = gradient.Grad_calc(self.InstDF.CoordDF,velo_field,x)
        
        velo_grad = velo_grad.reshape((*self.InstDF.shape,3,3))
        velo_grad_T = np.einsum('ijklm -> ijkml',velo_grad)
        strain_rate = 0.5*( velo_grad + velo_grad_T)
        
        rot_rate = 0.5*(velo_grad - velo_grad_T)

        del velo_field; del velo_grad; del velo_grad_T
        
        S2_Omega2 = np.matmul(strain_rate,strain_rate) + np.matmul(rot_rate,rot_rate)
        del strain_rate ; del rot_rate

        if rcParams['use_cupy'] and _cupy_avail:
            S2_Omega2_eigvals, e_vecs = cnpy.linalg.eigh(S2_Omega2)
            del e_vecs; del S2_Omega2
            lambda2 = cnpy.sort(S2_Omega2_eigvals,axis=3)[:,:,:,1].get()
        else:
            S2_Omega2_eigvals, e_vecs = np.linalg.eigh(S2_Omega2)
            del e_vecs; del S2_Omega2
            lambda2 = np.sort(S2_Omega2_eigvals,axis=3)[:,:,:,1]
        
        del S2_Omega2_eigvals        
        
        return cd.FlowStruct3D(self.InstDF._coorddata,{(PhyTime,'lambda_2'):lambda2})

    @docstring.sub
    def plot_lambda2(self,vals_list,x_split_pair=None,PhyTime=None,y_limit=None,y_mode='half_channel',Y_plus=True,colors=None,surf_kw=None,fig=None,ax=None,**kwargs):
        """
        Creates isosurfaces for the lambda 2 criterion

        Parameters
        ----------
        vals_list : list of floats
            isovalues to be plotted
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        ylim : float, optional
            wall-normal extent of the isosurface plot, by default None
        Y_plus : bool, optional
            Whether the above value is in wall units, by default True
        avg_data : CHAPSim_AVG, optional
            Instance of avg_data need if Y_plus is True, by default None
        colors : list of str, optional
            list to represent the order of the colors to be plotted, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects

        """
        PhyTime = self.check_PhyTime(PhyTime)
        vals_list = misc_utils.check_list_vals(vals_list)

        if y_limit is not None:
            y_lim_int = self._avg_data.ycoords_from_norm_coords([y_limit],inst_time=PhyTime,mode=y_mode)[0][0]
        else:
            y_lim_int = None

        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
        fig, ax = cplt.create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        lambda_2DF = self.lambda2_calc(PhyTime)
        for i,val in enumerate(vals_list):
            if colors is not None:
                color = colors[i%len(colors)]
                surf_kw['facecolor'] = color
            fig, ax1 = lambda_2DF.plot_isosurface('lambda_2',val,time=PhyTime,y_limit=y_lim_int,
                                            x_split_pair=x_split_pair,fig=fig,ax=ax,
                                            surf_kw=surf_kw)
            ax.axes.set_ylabel(r'$x/\delta$')
            ax.axes.set_xlabel(r'$z/\delta$')
            ax.axes.invert_xaxis()

        return fig, ax1

    def Q_crit_calc(self,PhyTime=None):
        PhyTime = self.check_PhyTime(PhyTime)

        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
                
        D = np.zeros((*self.shape,3,3))

        for i,velo in enumerate(velo_list):
            velo_field = self.InstDF[PhyTime,velo]
            for j,coord in enumerate(coord_list):
                D[:,:,:,i,j] = gradient.Grad_calc(self.CoordDF,velo_field,coord)

        del velo_field

        Q = 0.5*(np.trace(D,axis1=3,axis2=4,dtype=rcParams['dtype'])**2 - \
            np.trace(np.matmul(D,D,dtype=rcParams['dtype']),axis1=3,axis2=4,dtype=rcParams['dtype']))
        del D
        return cd.flowstruct3D(self._coorddata,{(PhyTime,'Q'):Q})

    def plot_Q_criterion(self,vals_list,x_split_pair=None,PhyTime=None,y_limit=None,y_mode='half_channel',colors=None,surf_kw=None,fig=None,ax=None,**kwargs):
        PhyTime = self.check_PhyTime(PhyTime)
        vals_list = misc_utils.check_list_vals(vals_list)

        if y_limit is not None:
            y_lim_int = self._avg_data.ycoords_from_norm_coords([y_limit],inst_time=PhyTime,mode=y_mode)[0][0]
        else:
            y_lim_int = None

        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
        fig, ax = cplt.create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        Q = self.Q_crit_calc(PhyTime)
        for i,val in enumerate(vals_list):
            if colors is not None:
                color = colors[i%len(colors)]
                surf_kw['facecolor'] = color
            fig, ax1 = Q.plot_isosurface('Q',val,time=PhyTime,y_limit=y_lim_int,
                                            x_split_pair=x_split_pair,fig=fig,ax=ax,
                                            surf_kw=surf_kw)
            ax.axes.set_ylabel(r'$x/\delta$')
            ax.axes.set_xlabel(r'$z/\delta$')
            ax.axes.invert_xaxis()

        return fig, ax1

    def vorticity_calc(self,PhyTime=None):
        """
        Calculate the vorticity vector

        Parameters
        ----------
        PhyTime : float, optional
            Physical time, by default None

        Returns
        -------
        datastruct
            Datastruct with the vorticity vector in it
        """

        PhyTime = self.check_PhyTime(PhyTime)

        vorticity = np.zeros((3,*self.InstDF.shape),dtype='f8')
        velo_vector = self.InstDF.values[:3]

        vorticity = gradient.Curl(self.InstDF.CoordDF,
                                 velo_vector,
                                 polar=self.Domain.is_polar)

        index = [(PhyTime,x) for x in ['x','y','z']]
        return cd.FlowStruct3D(self.InstDF._coorddata,vorticity,index=index)

    @docstring.sub
    def plot_vorticity_contour(self,comp,plane,axis_vals,PhyTime=None,x_split_list=None,y_mode='half_channel',pcolor_kw=None,fig=None,ax=None,**kwargs):
        """
        Creates a contour plot of the vorticity contour

        Parameters
        ----------
        comp : str
            Component of the vorticity to be extracted e.g. "x" for 
            \omega_z, the spanwise vorticity
        plane : str, optional
            Plane for the contour plot for example "xz" or "rtheta" (for pipes),
            by default 'xz'
        axis_vals : int, float (or list of them)
            locations in the third axis to be plotted
        PhyTime : float, optional
            Physical time to be plotted, None can be used if the instance contains a single 
            time, by default None
        avg_data : CHAPSim_AVG
            Used to aid plotting certain locations using in the y direction 
            if wall units are used for example
        x_split_list : list, optional
            Separating domain into different streamwise lengths useful if the domain is much 
            longer than its width, by default None
        y_mode : str, optional
            Only relevant if the xz plane is being used. The y value can be selected using a 
            number of different normalisations, by default 'wall'
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        VorticityDF = self.vorticity_calc(PhyTime=PhyTime)

        plane = self.Domain.out_to_in(plane)
        axis_vals = misc_utils.check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = VorticityDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = self._avg_data.ycoords_from_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
            int_vals = self._avg_data.ycoords_from_norm_coords(axis_vals,inst_time=PhyTime,mode=y_mode)[0]
        else:
            int_vals = axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)

        x_size, z_size = VorticityDF.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax, axes_output = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i,val in enumerate(int_vals):
            fig, ax1 = VorticityDF.plot_contour(comp,plane,val,time=PhyTime,fig=fig,ax=ax[i],pcolor_kw=pcolor_kw)
            ax1.axes.set_xlabel(r"$%s/\delta$" % plane[0])
            ax1.axes.set_ylabel(r"$%s/\delta$" % plane[1])
            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
            
            cbar=fig.colorbar(ax1,ax=ax[i])
            cbar.set_label(r"$%s^\prime$"%comp)

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_entrophy(self):
        pass
    def __str__(self):
        return self.InstDF.__str__()
    def __iadd__(self,inst_data):
        assert self.CoordDF.equals(inst_data.CoordDF), "CHAPSim_Inst are not from the same case"
        assert self.NCL == inst_data.NCL, "CHAPSim_Inst are not from the same case"

        self.InstDF.concat(inst_data.InstDF)
        return self

class CHAPSim_Inst_io(_Inst_base):
    _tgpost = False 
    
    @classproperty
    def _avg_class(cls):
        return cls._module._avg_io_class
    
    def _create_avg_data(self,path_to_folder,abs_path,time0,avg_data=None):
        time = misc_utils.max_time_calc(path_to_folder,abs_path)
        if avg_data is not None :
            if time in avg_data.times or self._update_time:
                return avg_data
            else:
                msg = ("The averaged data does not contain the required times. Re-extracting average data.\n"
                        f"Required times: {time}. Times present: {avg_data.times}")
                warnings.warn(msg)
        return self._module._avg_io_class(time,path_to_folder=path_to_folder,abs_path=abs_path,time0=time0)

class CHAPSim_Inst_tg(_Inst_base):
    _tgpost = True
    
    @classproperty
    def _avg_class(cls):
        return cls._module._avg_tg_class
    
    def _create_avg_data(self,path_to_folder,abs_path,time0,avg_data=None):
        time = misc_utils.max_time_calc(path_to_folder,abs_path)
        if avg_data is not None:
            if time in avg_data.times or self._update_time:
                return avg_data
            else:
                msg = ("The averaged data does not contain the required times. Re-extracting average data.\n"
                        f"Required times: {time}. Times present: {avg_data.times}")
                warnings.warn(msg)

        return self._module._avg_tg_class(time,path_to_folder=path_to_folder,abs_path=abs_path,time0=time0)
    def _velo_interp(self, flow_info, NCL3, NCL2, NCL1):
        io_flg = self.metaDF['iDomain'] == 3 

        if io_flg:
            new_flow_shape = (*flow_info.shape[:-1],flow_info.shape[-1]+1)
            new_flow_array = np.zeros(new_flow_shape)
            new_flow_array[:,:,:,:-1] =  flow_info
            new_flow_array[:,:,:,-1] = flow_info[:,:,:,0]
            x_size = NCL1+1
        else:
            new_flow_array = flow_info
            x_size = NCL1
            
        return  super()._velo_interp(new_flow_array, NCL3, NCL2, x_size)


class CHAPSim_Inst_temp(_Inst_base):
    _tgpost = True
    @classproperty
    def _avg_class(cls):
        return cls._module._avg_temp_class
    
    def _create_avg_data(self,path_to_folder,abs_path,time0,avg_data=None):
        times = self.InstDF.times
                                                  
        
        if avg_data is not None:
            if all(time in avg_data.times for time in times) or self._update_time:
                return avg_data
            else:
                inst_intersect = avg_data._get_intersect([times,avg_data.times],
                                                          path=path_to_folder)
                avg_intersect = avg_data._get_intersect([avg_data.times,times],
                                                          path=path_to_folder)

                if all(time in inst_intersect for time in times):
                    new_avg_data = avg_data.copy()
                    avg_times = [time for time in new_avg_data.times if not time in avg_intersect]
                    new_avg_data._del_times(avg_times)

                    new_avg_data.set_times(inst_intersect)
                    if not all(time in times for time  in new_avg_data.times):
                        raise ValueError("There has been a problem with the"
                                         f" times: {new_avg_data.times} {times}")
                    return new_avg_data
                
                else:
                    msg = ("The averaged data does not contain the "
                            "required times. Re-extracting average data.")
                    warnings.warn(msg)
                    
                    logger.debug(f"Instant times: {times}. AVG times:"
                                f" {avg_data.times}")
        return self._module._avg_temp_class(path_to_folder=path_to_folder,abs_path=abs_path,PhyTimes=times)

