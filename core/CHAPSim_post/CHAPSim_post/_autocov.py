"""
# _autocov.py
Module for processing autocovariance, autocorrelation and spectra from
instantaneous results from CHAPSim DNS solver
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from scipy import fft

import sys
import warnings
import gc
import time
from abc import ABC, abstractmethod

import CHAPSim_post as cp
import CHAPSim_post.CHAPSim_post._utils as utils

from .. import CHAPSim_plot as cplt
from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd

# from ._f90_ext_base import autocov_calc_z, autocov_calc_x

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg_base
_avg_io_class = CHAPSim_AVG_io
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._fluct import CHAPSim_fluct_io,CHAPSim_fluct_tg
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg

from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta

class CHAPSim_autocov_base(ABC):
    _module = sys.modules[__name__]
    def __init__(self,*args,**kwargs):
        fromfile=kwargs.pop('fromfile',False)
        if not fromfile:
            self._autocov_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)

    @abstractmethod
    def _autocov_extract(self,*args,**kwargs):
        raise NotImplementedError

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key =  'CHAPSim_autocov'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(key)
        group.attrs['shape_x'] = np.array(self.shape_x)
        group.attrs['shape_z'] = np.array(self.shape_z)
        group.attrs['comp'] = np.array([np.string_(x) for x in self.comp])
        hdf_file.close()

        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',key+'/avg_data')
        self.autocorrDF.to_hdf(file_name,key=key+'/autocorrDF',mode='a')#,format='fixed',data_columns=True)

    @abstractmethod
    def plot_autocorr_line(self,comp,axis_vals,y_vals,y_mode='half_channel',norm_xval=None,norm=True,fig=None,ax=None,**kwargs):
        
        if not comp in ('x','z'):
            msg = f"comp must be the string 'x' or 'z' not {comp} "

        axis_vals = utils.check_list_vals(axis_vals)
        y_vals = utils.check_list_vals(y_vals)

        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)


        Ruu = self.autocorrDF[comp]
        shape = Ruu.shape

        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5*len(y_vals)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(y_vals),fig=fig,ax=ax,**kwargs)

        if norm_xval is not None:
            if norm_xval ==0:
                norm_xval = np.amin(self._avg_data._return_xaxis())
            x_axis_vals=[norm_xval]*len(axis_vals)
        else:
            x_axis_vals=axis_vals

        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,x_axis_vals,y_mode)

        coord = self._meta_data.CoordDF[comp][:shape[0]]
        for j,_ in enumerate(y_vals):
            for i,_ in enumerate(axis_index):
                ax[j].cplot(coord,Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                
                #creating title label
                y_unit=r"y" if y_mode=='half_channel' \
                        else r"\delta_u" if y_mode=='disp_thickness' \
                        else r"\theta" if y_mode=='mom_thickness' \
                        else r"y^+" if norm_xval !=0 else r"y^{+0}"
 

                ax[j].set_title(r"$%s=%.3g$"%(y_unit,y_vals[j]),loc='left')
            ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)
            ax[j].set_xlabel(r"$%s/\delta$"%comp)
            
        

        return fig, ax

    @abstractmethod
    def plot_spectra(self,comp,axis_vals,y_vals,y_mode='half_channel',norm_xval=None,fig=None,ax=None,**kwargs):
        if not comp in ('x','z'):
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        axis_vals = utils.check_list_vals(axis_vals)
        y_vals = utils.check_list_vals(y_vals)

        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
        
        Ruu = self.autocorrDF[comp]
        shape = Ruu.shape
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5*len(y_vals)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(y_vals),fig=fig,ax=ax,**kwargs)


        if norm_xval is not None:
            if norm_xval ==0:
                x_axis_vals = [np.amin(self._avg_data._return_xaxis())]*len(axis_vals)
            else:
                x_axis_vals=[norm_xval]*len(axis_vals)
        else:
            x_axis_vals=axis_vals

        coord = self._meta_data.CoordDF[comp][:shape[0]]

        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,x_axis_vals,y_mode)
        

        for j,_ in enumerate(y_vals):
            for i,_ in enumerate(axis_index):
                wavenumber_spectra = fft.rfft(Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                delta_comp = coord[1]-coord[0]

                comp_size= Ruu[:,y_index_axis_vals[i][j],axis_index[i]].size
                wavenumber_comp = 2*np.pi*fft.rfftfreq(comp_size,delta_comp)
                
                y_unit=r"y" if y_mode=='half_channel' \
                        else r"\delta_u" if y_mode=='disp_thickness' \
                        else r"\theta" if y_mode=='mom_thickness' \
                        else r"y^+" if norm_xval !=0 else r"y^{+0}"

                ax[j].cplot(wavenumber_comp,2*np.abs(wavenumber_spectra))
                ax[j].set_title(r"$%s=%.3g$"%(y_unit,y_vals[j]),loc='left')

            string= (ord(self.comp[0])-ord('u')+1,ord(self.comp[1])-ord('u')+1,comp)
            ax[j].set_ylabel(r"$E_{%d%d}(\kappa_%s)$"%string)
            ax[j].set_xlabel(r"$\kappa_%s$"%comp)
        
        return fig, ax

    @abstractmethod
    def autocorr_contour_y(self,comp,axis_vals,Y_plus=False,Y_plus_0=False,
                                Y_plus_max =None,norm=True,
                                show_positive=True,fig=None,ax=None,pcolor_kw=None,**kwargs):
        if comp not in ('x','z'):
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)

        shape = self.autocorrDF[comp].shape 
        Ruu = self.autocorrDF[comp][:,:,axis_index]

        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,4*len(axis_vals)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)


        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        max_val = -np.float('inf'); min_val = np.float('inf')

        for i, _ in enumerate(axis_vals):
            y_coord = self._meta_data.CoordDF['y'].copy()
            coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
            if Y_plus:
                avg_time = self._avg_data.flow_AVGDF.index[0][0]
                _, delta_v_star = self._avg_data.wall_unit_calc(avg_time)
                y_coord = y_coord[:int(y_coord.size/2)]
                if i==0:
                    Ruu = Ruu[:,:y_coord.size]
                if Y_plus_0:
                    y_coord = (1-np.abs(y_coord))/delta_v_star[0]
                else:   
                    y_coord = (1-np.abs(y_coord))/delta_v_star[axis_index[i]]
            
            min_val = min(min_val,np.amin(np.squeeze(Ruu[:,:,i])))
            max_val = max(max_val,np.amax(np.squeeze(Ruu[:,:,i])))

            X,Y = np.meshgrid(coord,y_coord)
            ax[i] = ax[i].pcolormesh(X,Y,np.squeeze(Ruu[:,:,i]).T,**pcolor_kw)
            
            ax[i].axes.set_xlabel(r"$\Delta %s/\delta$" %comp)
            if Y_plus and Y_plus_0:
                ax[i].axes.set_ylabel(r"$Y^{+0}$")
            elif Y_plus and not Y_plus_0:
                ax[i].axes.set_ylabel(r"$Y^{+}$")
            else:
                ax[i].axes.set_ylabel(r"$y/\delta$")

            if Y_plus_max is not None:
                ax[i].axes.set_ylim(top=Y_plus_max)

            fig.colorbar(ax[i],ax=ax[i].axes)
            fig.tight_layout()

        for a in ax:     
            a.set_clim(min_val,max_val)

        if not show_positive:
            for a in ax:   
                cmap = a.get_cmap()
                min_val,max_val = a.get_clim()
                new_color = cmap(np.linspace(0,1,256))[::-1]
                new_color[-1] = np.array([1,1,1,1])
                a.set_cmap(mpl.colors.ListedColormap(new_color))
                a.set_clim(min_val,0)

        return fig, ax

    @abstractmethod
    def autocorr_contour_x(self,comp,axis_vals,axis_mode='half_channel',norm=True,fig=None,ax=None,pcolor_kw=None,**kwargs):
        if not comp in ('x','z'):
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        axis_vals = utils.check_list_vals(axis_vals)

        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        
        Ruu_all = self.autocorrDF[comp]
        shape = Ruu_all.shape

        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i]
        
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,4*len(axis_vals)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)
        
        x_axis =self._avg_data._return_xaxis()
        coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        ax_out=[]
        for i in range(len(axis_vals)):

            X,Y = np.meshgrid(x_axis,coord)
            ax[i] = ax[i].pcolormesh(X,Y,Ruu[:,i],**pcolor_kw)            
            ax[i].axes.set_ylabel(r"$\Delta %s/\delta$" %comp)
            title = r"$%s=%.3g$"%("y" if axis_mode=='half_channel' \
                        else r"\delta_u" if axis_mode=='disp_thickness' \
                        else r"\theta" if axis_mode=='mom_thickness' else r"y^+", axis_vals[i] )
            ax[i].axes.set_title(title,loc='left')
            fig.colorbar(ax[i],ax=ax[i].axes)

        fig.tight_layout()
        
        return fig, ax

    @abstractmethod
    def spectrum_contour(self,comp,axis_vals,axis_mode='half_channel',fig=None,ax=None,pcolor_kw=None,**kwargs):
        if not comp in ('x','z'):
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")

        axis_vals = utils.check_list_vals(axis_vals)
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        Ruu_all = self.autocorrDF[comp]#[:,axis_vals,:]
        shape = Ruu_all.shape
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i]

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,4*len(axis_vals)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
        x_axis =self._avg_data._return_xaxis()

        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        for i in range(len(axis_vals)):
            wavenumber_spectra = np.zeros((int(0.5*shape[0])+1,shape[2]))
            for j in range(shape[2]):
                wavenumber_spectra[:,j]=fft.rfft(Ruu[:,i,j])
            comp_size= shape[0]
            wavenumber_comp = 2*np.pi*fft.rfftfreq(comp_size,coord[1]-coord[0])
            
            X, Y = np.meshgrid(x_axis,wavenumber_comp)
            ax[i] = ax[i].pcolormesh(X,Y,np.abs(wavenumber_spectra),**pcolor_kw)
            
            ax[i].axes.set_ylabel(r"$\kappa_%s$"%comp)
            
            title = r"$%s=%.3g$"%("y" if axis_mode=='half_channel' \
                        else r"\delta_u" if axis_mode=='disp_thickness' \
                        else r"\theta" if axis_mode=='mom_thickness' else "y^+", axis_vals[i] )
            
            ax[i].axes.set_ylim([np.amin(wavenumber_comp[1:]),np.amax(wavenumber_comp)])
            ax[i].axes.set_title(title)# ,fontsize=15,loc='left')
            
            fig.colorbar(ax[i],ax=ax[i].axes)

        return fig, ax

    def __str__(self):
        return self.autocorrDF.__str__()

class CHAPSim_autocov_io(CHAPSim_autocov_base):
    _module = sys.modules[__name__]
    def _autocov_extract(self,comp1,comp2,path_to_folder=".",time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):

        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.Params['TEST']:
            times.sort(); times= times[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder)
        self.comp=(comp1,comp2)
        self.NCL = self._meta_data.NCL

        try:
            self._avg_data = self._module._avg_io_class(max(times),self._meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            self._avg_data = self._module._avg_io_class(max(times),self._meta_data,path_to_folder,time0)

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")
        
        self.shape_x = (max_x_sep,self.NCL[1],self.NCL[0]-max_x_sep)
        self.shape_z = (max_z_sep,self.NCL[1],self.NCL[0])
        
        for i,timing in enumerate(times):
            
            fluct_data = self._module._fluct_io_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)
            coe3 = i/(i+1)
            coe2 = 1/(i+1)

            if i==0:
                R_x, R_z = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_R_x, local_R_z = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                if R_x.shape != local_R_x.shape or R_z.shape != local_R_z.shape:
                    msg = "There is a problem. the shapes of the local and averaged array are different"
                    raise ValueError(msg)
                R_x = R_x*coe3 + local_R_x*coe2
                R_z = R_z*coe3 + local_R_z*coe2

        self.autocorrDF = cd.datastruct.from_dict({'x':R_x,'z':R_z})#.data([shape_x,shape_z])
   
    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  'CHAPSim_autocov_io'
        
        hdf_file = h5py.File(file_name,'r')
        self.shape_x = tuple(hdf_file[key].attrs["shape_x"][:])
        self.shape_z = tuple(hdf_file[key].attrs["shape_z"][:])
        self.comp = tuple(np.char.decode(hdf_file[key].attrs["comp"][:]))
        hdf_file.close()       

        self.autocorrDF = cd.datastruct.from_hdf(file_name,shapes=(self.shape_x,self.shape_z),key=key+'/autocorrDF')#pd.read_hdf(file_name,key=base_name+'/autocorrDF').data([shape_x,shape_z])
        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')
        self.NCL=self._meta_data.NCL
        self._avg_data = self._module._avg_io_class.from_hdf(file_name,key+'/avg_data')
    
    def save_hdf(self, file_name, write_mode, key=None):
        if not key:
            key = 'CHAPSim_autocov_io'
        super().save_hdf(file_name, write_mode, key=key)
    
    def _autocov_calc(self,fluct_data,comp1,comp2,PhyTime,max_x_sep,max_z_sep):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        NCL=fluct_data.NCL        

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp1]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp2]

        time1=time.time()
        R_x = self._autocov_calc_x(fluct_vals1,fluct_vals2,*NCL,max_x_sep)#.mean(axis=1)
        R_z = self._autocov_calc_z(fluct_vals1,fluct_vals2,*NCL,max_z_sep)
        print(time.time()-time1,flush=True)
        
        return R_x, R_z

    def _autocov_calc_z(self,fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        
        if cp.Params['autocorr_mode'] in (1,2):    
            if cp.Params['autocorr_mode'] == 1:
                from ._f90_ext_base import autocov_calc_z
                R_z = np.zeros((max_z_step,NCL2,NCL1),order='F')
            else:
                from ._cy_ext_base import autocov_calc_z
                R_z = np.zeros((max_z_step,NCL2,NCL1))

            if max_z_step >0:
                try:
                    autocov_calc_z(fluct1,fluct2,R_z,max_z_step)
                except Exception as e:
                    msg = f"Exception raised by accelerator routine of type {type(e).__name__}: {e.__str__()}: "
                    if cp.Params['ForceMode']:    
                        raise RuntimeError(msg+"Parameter ForceMode set to true raising RuntimeError")
                    else:
                        warnings.simplefilter('once')
                        warnings.warn(msg + "Using numba backend")
                        return self._autocov_numba_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step)
                else:
                    return np.ascontiguousarray(R_z)
        else:
            if cp.Params['autocorr_mode'] != 0 and cp.Params['ForceMode']:
                msg = f"Calculation mode {cp.Params['autocorr_mode']} doesn't exist, ForceMode is True raising RuntimeError"
                raise RuntimeError(msg)
            elif cp.Params['autocorr_mode'] != 0 and cp.Params['ForceMode']:
                msg = f"Calculation mode {cp.Params['autocorr_mode']} doesn't exist, using numba backend"
                warnings.warn(msg)
            return self._autocov_numba_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step)
    
    @staticmethod
    
    def _autocov_numba_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        from numba import njit
        @njit(parallel=True,fastmath=True)
        def numba_method(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
            R_z = np.zeros((max_z_step,NCL2,NCL1))

            if max_z_step >0:
                for iz0 in numba.prange(max_z_step):
                    for iz in numba.prange(NCL3-max_z_step):
                        R_z[iz0,:,:] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
            R_z /= (NCL3-max_z_step)
            return R_z
        return numba_method(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step)



    def _autocov_calc_x(self,fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step):
        if cp.Params['autocorr_mode'] in (1,2):    
            if cp.Params['autocorr_mode'] == 1:
                from ._f90_ext_base import autocov_calc_x
                R_x = np.zeros((max_x_step,NCL2,NCL1-max_x_step),order='F')
            else:
                from ._cy_ext_base import autocov_calc_x
                R_x = np.zeros((max_x_step,NCL2,NCL1-max_x_step))

            if max_x_step >0:
                try:
                    autocov_calc_x(fluct1,fluct2,R_x,max_x_step)
                except Exception as e:
                    msg = f"Exception raised by accelerator routine of type {type(e).__name__}: {e.__str__()}: "
                    if cp.Params['ForceMode']:    
                        raise RuntimeError(msg+"Parameter ForceMode set to true raising RuntimeError")
                    else:
                        warnings.simplefilter('once')
                        warnings.warn(msg + "Using numba backend")
                        return self._autocov_numba_x(fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step)
                else:
                    return np.ascontiguousarray(R_x)
                                                        
        else:
            if cp.Params['autocorr_mode'] != 0 and cp.Params['ForceMode']:
                msg = f"Calculation mode {cp.Params['autocorr_mode']} doesn't exist, ForceMode is True raising RuntimeError"
                raise RuntimeError(msg)
            elif cp.Params['autocorr_mode'] != 0 and cp.Params['ForceMode']:
                msg = f"Calculation mode {cp.Params['autocorr_mode']} doesn't exist, using numba backend"
                warnings.warn(msg)

            return self._autocov_numba_x(fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step)

    @staticmethod
    def _autocov_numba_x(fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step):
        R_x = np.zeros((max_x_step,NCL2,NCL1-max_x_step))
        for ix0 in range(max_x_step):
            for ix in range(NCL1-max_x_step):
                R_x[ix0,:,ix] += (fluct1[:,:,ix]*fluct2[:,:,ix0+ix]).mean(axis=0)
        
        return R_x
    


    def plot_autocorr_line(self, comp, axis_vals,*args,**kwargs):
        fig, ax =  super().plot_autocorr_line(comp, axis_vals, *args, **kwargs)
        for a in ax:
            lines = a.get_lines()[-len(axis_vals):]
            for line,val in zip(lines,axis_vals):
                line.set_label(r"$x^*=%.3g$"%val)

        ncol = cplt.get_legend_ncols(len(ax[0].get_lines()))
        ax[0].clegend(vertical=False,ncol=ncol)
        ax[0].get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_spectra(self, comp, axis_vals,*args,**kwargs):
        
        fig, ax =  super().plot_spectra(comp, axis_vals, *args, **kwargs)
        for a in ax:
            lines = a.get_lines()[-len(axis_vals):]
            for line,val in zip(lines,axis_vals):
                line.set_label(r"$x^*=%.3g$"%val)

        ncol = cplt.get_legend_ncols(len(ax[0].get_lines()))
        ax[0].clegend(vertical=False,ncol=ncol)
        ax[0].get_gridspec().tight_layout(fig)
        return fig, ax
        
    def autocorr_contour_y(self,comp,axis_vals,*args,**kwargs):
        fig, ax = super().autocorr_contour_y(comp,axis_vals,*args,**kwargs)
        for a, val in zip(ax,axis_vals):
            a.axes.set_title(r"$x=%.3g$"%val,loc='right')
        return fig, ax

    def autocorr_contour_x(self,comp,*args,**kwargs):
        fig, ax =super().autocorr_contour_x(comp,*args,**kwargs)
        for a in ax:
            a.axes.set_xlabel(r"$x^*$")
        return fig, ax

    def spectrum_contour(self, comp,*args,**kwargs):
        fig, ax =  super().spectrum_contour(comp,*args,**kwargs)
        for a in ax:
            a.axes.set_xlabel(r"$x^*$")
        return fig, ax

class CHAPSim_autocov_tg(CHAPSim_autocov_base):
    _module = sys.modules[__name__]
    def _autocov_extract(self,comp1,comp2,path_to_folder='.',time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.Params['TEST']:
            times.sort(); times= times[-3:]
            
        self._meta_data = self._module.CHAPSim_meta(path_to_folder)
        self.comp=(comp1,comp2)
        self.NCL = self._meta_data.NCL
        self._avg_data = self._module._avg_tg_base_class(times,meta_data=self._meta_data,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path)

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]/2)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("\033[1;32 Variable max_z_sep must be less than half NCL3 in readdata file\n")
        if max_x_sep is None:
            max_x_sep=int(self._NCL[0]/2)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("\033[1;32 Variable max_x_sep must be less than half NCL3 in readdata file\n")
        self.shape_x = (max_x_sep,self.NCL[1],len(times))
        self.shape_z = (max_z_sep,self.NCL[1],len(times))
        for timing in times:
            fluct_data = self._module._fluct_tg_class(timing,self._avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            if 'R_z' not in locals():
                R_z, R_x = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_R_z, local_R_x = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                R_z = np.vstack([R_z,local_R_z])
                R_x = np.vstack([R_x,local_R_x])
            gc.collect()

        R_z = R_z.T.reshape(self.shape_z)
        R_x = R_x.T.reshape(self.shape_x)
        
        self.autocorrDF = cd.datastruct({'x':R_x,'z':R_z})
    
    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  'CHAPSim_autocov_tg'
        
        hdf_file = h5py.File(file_name,'r')
        self.shape_x = tuple(hdf_file[key].attrs["shape_x"][:])
        self.shape_z = tuple(hdf_file[key].attrs["shape_z"][:])
        self.comp = tuple(np.char.decode(hdf_file[key].attrs["comp"][:]))
        hdf_file.close()        

        self.autocorrDF = cd.datastruct.from_hdf(file_name,shapes=(self.shape_x,self.shape_z),key=base_name+'/autocorrDF')#pd.read_hdf(file_name,key=base_name+'/autocorrDF').data([shape_x,shape_z])
        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')
        self.NCL=self._meta_data.NCL
        self._avg_data = self._module._avg_tg_base_class.from_hdf(file_name,key+'/avg_data')
    
    def save_hdf(self, file_name, write_mode, key=None):
        if key is None:
            key = 'CHAPSim_autocov_tg'
        super().save_hdf(file_name, write_mode, key=key)
    
    @staticmethod
    def _autocov_calc(fluct_data,comp1,comp2,PhyTime,max_x_sep,max_z_sep):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        NCL=fluct_data.NCL    

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp1]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp2]
        print(PhyTime)
        R_x = CHAPSim_autocov_tg._autocov_calc_x(fluct_vals1,fluct_vals2,*NCL,max_x_sep)
        R_z = CHAPSim_autocov_tg._autocov_calc_z(fluct_vals1,fluct_vals2,*NCL,max_z_sep)

        R_z = np.mean(R_z,axis=2)/(NCL[2]-max_z_sep)
        R_x = np.mean(R_x,axis=2)/(NCL[2])
        
        return R_z.flatten(), R_x.flatten()

    # @staticmethod
    # @numba.njit(parallel=True,fastmath=True)
    # def _autocov_calc_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
    #     R_z = np.zeros((max_z_step,NCL2,NCL1))
    #     if max_z_step >0:
    #         for iz0 in numba.prange(max_z_step):
    #             for iz in numba.prange(NCL3-max_z_step):
    #                 R_z[iz0] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
    #     return R_z
    # @staticmethod
    # @numba.njit(parallel=True,fastmath=True)
    # def _autocov_calc_x(fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step):
        
    #     R_x = np.zeros((max_x_step,NCL2,NCL1-max_x_step))
    #     if max_x_step >0:
    #         for ix0 in numba.prange(max_x_step):
    #             for ix in numba.prange(NCL1-max_x_step):
    #                 for iz in numba.prange(NCL3):
    #                     R_x[ix0,:,ix] += fluct1[iz,:,ix]*fluct2[iz,:,ix0+ix]
    #     return R_x

    def plot_autocorr_line(self, comp, axis_vals,*args,**kwargs):
        fig, ax =  super().plot_autocorr_line(comp, axis_vals, *args, **kwargs)
        for a in ax:
            lines = a.get_lines()[-len(axis_vals):]
            for line,val in zip(lines,axis_vals):
                line.set_label(r"$t^*=%.3g$"%val)

        axes_items_num = len(ax[0].get_lines())
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax[0].clegend(vertical=False,ncol=ncol)# ,fontsize=15)
        ax[0].get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_spectra(self, comp, axis_vals,*args,**kwargs):
        fig, ax =  super().plot_spectra(comp, axis_vals, *args, **kwargs)
        for a in ax:
            lines = a.get_lines()[-len(axis_vals):]
            for line,val in zip(lines,axis_vals):
                line.set_label(r"$t^*=%.3g$"%val)

        axes_items_num = len(ax[0].get_lines())
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax[0].clegend(vertical=False,ncol=ncol)# ,fontsize=15)
        ax[0].get_gridspec().tight_layout(fig)
        return fig, ax

    def autocorr_contour_y(self,comp,axis_vals,*args,**kwargs):
        fig, ax = super().autocorr_contour_y(comp,axis_vals,*args,**kwargs)
        for a, val in zip(ax,axis_vals):
            a.axes.set_title(r"$t=%.3g$"%val,loc='right')
        return fig, ax    

    def autocorr_contour_x(self,*args,**kwargs):
        fig, ax =super().autocorr_contour_x(*args,**kwargs)
        for a in ax:
            a.axes.set_xlabel(r"$t^*$")
        
        return fig, ax

    def spectrum_contour(self, comp,*args,**kwargs):
        fig, ax =  super().spectrum_contour(comp,*args,**kwargs)
        for a in ax:
            a.axes.set_xlabel(r"$t^*$")
        
        return fig, ax