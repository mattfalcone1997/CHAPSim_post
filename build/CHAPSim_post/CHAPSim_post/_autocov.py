"""
# _autocov.py
Module for processing autocovariance, autocorrelation and spectra from
instantaneous results from CHAPSim DNS solver
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import numba
from scipy import fft

import sys
import warnings
import gc
import time

import CHAPSim_post as cp
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

class CHAPSim_autocov_base():
    _module = sys.modules[__name__]
    def __init__(self,*args,**kwargs):#self,comp1,comp2,max_x_sep=None,max_z_sep=None,path_to_folder='',time0='',abs_path=True):
        fromfile=kwargs.pop('fromfile',False)
        if not fromfile:
            self._meta_data, self.comp, self.NCL,\
            self._avg_data, self.autocorrDF, self.shape_x,\
            self.shape_z = self._autocov_extract(*args,**kwargs)
        else:
            self._meta_data, self.comp, self.NCL,\
            self._avg_data, self.autocorrDF, self.shape_x,\
            self.shape_z = self._hdf_extract(*args,**kwargs)

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.attrs['shape_x'] = np.array(self.shape_x)
        group.attrs['shape_z'] = np.array(self.shape_z)
        group.attrs['comp'] = np.array([np.string_(x) for x in self.comp])
        hdf_file.close()

        self._meta_data.save_hdf(file_name,'a',base_name+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',base_name+'/avg_data')
        self.autocorrDF.to_hdf(file_name,key=base_name+'/autocorrDF',mode='a')#,format='fixed',data_columns=True)

    def plot_autocorr_line(self,comp,axis_vals,y_vals,y_mode='half_channel',norm_xval=None,norm=True,fig='',ax='',**kwargs):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)


        if norm_xval is not None:
            if norm_xval ==0:
                norm_xval = np.amin(self._avg_data._return_xaxis())
            x_axis_vals=[norm_xval]*len(axis_vals)
        else:
            x_axis_vals=axis_vals

        coord = self._meta_data.CoordDF[comp][:shape[0]]
        if not hasattr(y_vals,'__iter__'):
            y_vals = [y_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,x_axis_vals,y_mode)
        Ruu = self.autocorrDF[comp]
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        if not fig:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [10,5*len(y_vals)]
                if len(y_vals) >1:
                    warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(y_vals),**kwargs)
        elif not ax:
            kwargs['subplot_kw'] = {'squeeze',False}
            ax = fig.subplots(len(y_vals),**kwargs)

        ax=ax.flatten()
        coord = self._meta_data.CoordDF[comp][:shape[0]]
        for j in range(len(y_vals)):
            for i in range(len(axis_index)):
                ax[j].cplot(coord,Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                #creating title label
                y_unit=r"y" if y_mode=='half_channel' \
                        else r"\delta_u" if y_mode=='disp_thickness' \
                        else r"\theta" if y_mode=='mom_thickness' \
                        else r"y^+" if norm_xval !=0 else r"y^{+0}"
 

                ax[j].set_title(r"$%s=%.3g$"%(y_unit,y_vals[j]),loc='left')
            ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)# ,fontsize=20)
            ax[j].set_xlabel(r"$%s/\delta$"%comp)# ,fontsize=20)
            
        

        return fig, ax
    def plot_spectra(self,comp,axis_vals,y_vals,y_mode='half_channel',norm_xval=None,fig='',ax='',**kwargs):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
        
            #raise TypeError("Variable `axis_vals' must be an int or iterable\n")

        if norm_xval is not None:
            if norm_xval ==0:
                x_axis_vals = [np.amin(self._avg_data._return_xaxis())]*len(axis_vals)
            else:
                x_axis_vals=[norm_xval]*len(axis_vals)
        else:
            x_axis_vals=axis_vals

        coord = self._meta_data.CoordDF[comp][:shape[0]]
        if not hasattr(y_vals,'__iter__'):
            y_vals = [y_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,x_axis_vals,y_mode)
        Ruu = self.autocorrDF[comp]#[:,axis_vals,:]

        if not fig:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [10,5*len(y_vals)]
                if len(y_vals) >1:
                    warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(y_vals),**kwargs)
        elif not ax:
            kwargs['subplot_kw'] = {'squeeze',False}
            ax = fig.subplots(len(y_vals),**kwargs)
        ax=ax.flatten()

        for j in range(len(y_vals)):
            for i in range(len(axis_index)):
                wavenumber_spectra = fft.rfft(Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                delta_comp = coord[1]-coord[0]
                Fs = (2.0*np.pi)/delta_comp
                comp_size= Ruu[:,y_index_axis_vals[i][j],axis_index[i]].size
                wavenumber_comp = 2*np.pi*fft.rfftfreq(comp_size,coord[1]-coord[0])
                y_unit=r"y" if y_mode=='half_channel' \
                        else r"\delta_u" if y_mode=='disp_thickness' \
                        else r"\theta" if y_mode=='mom_thickness' \
                        else r"y^+" if norm_xval !=0 else r"y^{+0}"
                ax[j].cplot(wavenumber_comp,2*np.abs(wavenumber_spectra))
                ax[j].set_title(r"$%s=%.3g$"%(y_unit,y_vals[j]),loc='left')
            string= (ord(self.comp[0])-ord('u')+1,ord(self.comp[1])-ord('u')+1,comp)
            ax[j].set_ylabel(r"$E_{%d%d}(\kappa_%s)$"%string)# ,fontsize=20)
            ax[j].set_xlabel(r"$\kappa_%s$"%comp)# ,fontsize=20)
        
        return fig, ax

    def autocorr_contour_y(self,comp,axis_vals,Y_plus=False,Y_plus_0=False,
                                Y_plus_max ='',norm=True,
                                show_positive=True,fig=None,ax=None):
        if comp not in ('x','z'):
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)

        shape = self.autocorrDF[comp].shape 
        Ruu = self.autocorrDF[comp][:,:,axis_index]
        print(shape,self.autocorrDF[comp].shape)


        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        if fig is None:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif ax is None:
            subplot_kw = {'squeeze':'False'}
            ax = fig.subplots(len(axis_vals),subplot_kw)
        ax=ax.flatten()
        # x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
        #             .values
        max_val = -np.float('inf'); min_val = np.float('inf')

        for i, axis_val in enumerate(axis_vals):
            y_coord = self._meta_data.CoordDF['y'].copy()
            coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
            if Y_plus:
                avg_time = self._avg_data.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
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
            ax[i] = ax[i].pcolormesh(X,Y,np.squeeze(Ruu[:,:,i]).T,cmap='jet',shading='auto')
            ax[i].axes.set_xlabel(r"$\Delta %s/\delta$" %comp)
            if Y_plus and Y_plus_0:
                ax[i].axes.set_ylabel(r"$Y^{+0}$")
            elif Y_plus and not Y_plus_0:
                ax[i].axes.set_ylabel(r"$Y^{+}$")
            else:
                ax[i].axes.set_ylabel(r"$y/\delta$")
            if Y_plus_max:
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

    def autocorr_contour_x(self,comp,axis_vals,axis_mode='half_channel',norm=True,fig='',ax=''):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals=[axis_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        
        Ruu_all = self.autocorrDF[comp]
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i]
        
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        if not fig:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif not ax:
            subplot_kw = {'squeeze':False}
            ax = fig.subplots(len(axis_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()
        
        x_axis =self._avg_data._return_xaxis()
        # y_coord = self._meta_data.CoordDF['y'].copy().dropna().values
        coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]

        ax_out=[]
        for i in range(len(axis_vals)):

            X,Y = np.meshgrid(x_axis,coord)
            ax1 = ax[i].pcolormesh(X,Y,Ruu[:,i],cmap='jet',shading='auto')            
            ax1.axes.set_ylabel(r"$\Delta %s/\delta$" %comp)
            title = r"$%s=%.3g$"%("y" if axis_mode=='half_channel' \
                        else r"\delta_u" if axis_mode=='disp_thickness' \
                        else r"\theta" if axis_mode=='mom_thickness' else r"y^+", axis_vals[i] )
            ax1.axes.set_title(title,loc='left')# ,fontsize=15,loc='left')
            fig.colorbar(ax1,ax=ax1.axes)
            ax_out.append(ax1)
        fig.tight_layout()
        
        return fig, np.array(ax_out)

    def spectrum_contour(self,comp,axis_vals,axis_mode='half_channel',fig='',ax=''):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")

        if not hasattr(axis_vals,'__iter__'):
            axis_vals=[axis_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        Ruu_all = self.autocorrDF[comp]#[:,axis_vals,:]
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i]

        if not fig:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif not ax:
            subplot_kw = {'squeeze':False}
            ax = fig.subplots(len(axis_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()
        # x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
        #         .values[:shape[2]]
        coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
        x_axis =self._avg_data._return_xaxis()

        ax_out=[]
        for i in range(len(axis_vals)):
            wavenumber_spectra = np.zeros((int(0.5*shape[0])+1,shape[2]))
            for j in range(shape[2]):
                wavenumber_spectra[:,j]=fft.rfft(Ruu[:,i,j])
            delta_comp = coord[1]-coord[0]
            Fs = (2.0*np.pi)/delta_comp
            comp_size= shape[0]
            wavenumber_comp = 2*np.pi*fft.rfftfreq(comp_size,coord[1]-coord[0])
            X, Y = np.meshgrid(x_axis,wavenumber_comp)
            ax1 = ax[i].pcolormesh(X,Y,np.abs(wavenumber_spectra),cmap='jet',shading='auto')
            ax1.axes.set_ylabel(r"$\kappa_%s$"%comp)
            title = r"$%s=%.3g$"%("y" if axis_mode=='half_channel' \
                        else r"\delta_u" if axis_mode=='disp_thickness' \
                        else r"\theta" if axis_mode=='mom_thickness' else "y^+", axis_vals[i] )
            ax1.axes.set_ylim([np.amin(wavenumber_comp[1:]),np.amax(wavenumber_comp)])
            ax1.axes.set_title(title)# ,fontsize=15,loc='left')
            fig.colorbar(ax1,ax=ax1.axes)
            ax_out.append(ax1)
        fig.tight_layout()
        return fig, np.array(ax_out)

    def __str__(self):
        return self.autocorrDF.__str__()

class CHAPSim_autocov_io(CHAPSim_autocov_base):
    _module = sys.modules[__name__]
    def _autocov_extract(self,comp1,comp2,path_to_folder='',time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        mem_debug = CT.debug_memory()
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if cp.Params['TEST']:
            times.sort(); times= times[-5:]
            
        meta_data = self._module._meta_class(path_to_folder)
        comp=(comp1,comp2)
        NCL = meta_data.NCL
        try:
            avg_data = self._module._avg_io_class(max(times),meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            avg_data = self._module._avg_io_class(max(times),meta_data,path_to_folder,time0)
        i=1

        if max_z_sep is None:
            max_z_sep=int(NCL[2]/2)
        elif max_z_sep>NCL[2]:
            raise ValueError("\033[1;32 Variable max_z_sep must be less than half NCL3 in readdata file\n")
        if max_x_sep is None:
            max_x_sep=int(NCL[0]/2)
        elif max_x_sep>NCL[0]:
            raise ValueError("\033[1;32 Variable max_x_sep must be less than half NCL3 in readdata file\n")
        shape_x = (max_x_sep,NCL[1],NCL[0]-max_x_sep)
        shape_z = (max_z_sep,NCL[1],NCL[0])
        for timing in times:
            
            fluct_data = self._module._fluct_io_class(timing,avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                R_x, R_z = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_R_x, local_R_z = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                if R_x.shape != local_R_x.shape or R_z.shape != local_R_z.shape:
                    msg = "There is a problem. the shapes of the local and averaged array are different"
                    raise ValueError(msg)
                R_x = R_x*coe3 + local_R_x*coe2
                R_z = R_z*coe3 + local_R_z*coe2
            # gc.collect()
            i += 1
        autocorrDF = cd.datastruct.from_dict({'x':R_x,'z':R_z})#.data([shape_x,shape_z])
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
   
    def _hdf_extract(self,file_name, group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov_io'
        hdf_file = h5py.File(file_name,'r')
        shape_x = tuple(hdf_file[base_name].attrs["shape_x"][:])
        shape_z = tuple(hdf_file[base_name].attrs["shape_z"][:])
        comp = tuple(np.char.decode(hdf_file[base_name].attrs["comp"][:]))
        hdf_file.close()       

        autocorrDF = cd.datastruct.from_hdf(file_name,shapes=(shape_x,shape_z),key=base_name+'/autocorrDF')#pd.read_hdf(file_name,key=base_name+'/autocorrDF').data([shape_x,shape_z])
        meta_data = self._module._meta_class.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = self._module._avg_io_class.from_hdf(file_name,base_name+'/avg_data')
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
    
    def save_hdf(self, file_name, write_mode, group_name=''):
        if not group_name:
            group_name = 'CHAPSim_autocov_io'
        super().save_hdf(file_name, write_mode, group_name=group_name)
    
    def _autocov_calc(self,fluct_data,comp1,comp2,PhyTime,max_x_sep,max_z_sep):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        NCL=fluct_data.NCL        

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp1]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp2]
        time1=time.time()
        R_x = self._autocov_calc_x(fluct_vals1,fluct_vals2,*NCL,max_x_sep)#.mean(axis=1)
        R_z = self._autocov_calc_z(fluct_vals1,fluct_vals2,*NCL,max_z_sep)
        # R_z = R_z/(NCL[2]-max_z_sep)
        print(time.time()-time1,flush=True)
        
        # R_x = R_x/(NCL[2])

        # R_z = R_z.reshape((max_z_sep*NCL[1]*NCL[0]))
        # Rz_DF = pd.DataFrame(R_z)
        # R_x = R_x.reshape((max_x_sep*NCL[1]*(NCL[0]-max_x_sep)))
        # Rx_DF = pd.DataFrame(R_x)
        # R_array=np.stack([R_x,R_z],axis=0)
        
        return R_x.copy(),R_z.copy()

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
                    autocov_calc_z(fluct1,fluct2,R_z,max_z_step)#,NCL1,NCL2,NCL3,max_z_step)
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
    @numba.njit(parallel=True,fastmath=True)
    def _autocov_numba_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        R_z = np.zeros((max_z_step,NCL2,NCL1))

        if max_z_step >0:
            for iz0 in numba.prange(max_z_step):
                for iz in numba.prange(NCL3-max_z_step):
                    R_z[iz0,:,:] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
        R_z /= (NCL3-max_z_step)
        return R_z



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
                    autocov_calc_x(fluct1,fluct2,R_x,max_x_step)#,NCL3,NCL2,NCL1,max_x_step)
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
    # @numba.jit()
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
                line.set_label(r"$x^*=%.3g$"%val)

        axes_items_num = len(ax[0].get_lines())
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax[0].clegend(vertical=False,ncol=ncol)# ,fontsize=15)
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
    def _autocov_extract(self,comp1,comp2,path_to_folder='',time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.Params['TEST']:
            times.sort(); times= times[-3:]
            
        meta_data = self._module.CHAPSim_meta(path_to_folder)
        comp=(comp1,comp2)
        NCL = meta_data.NCL
        avg_data = self._module._avg_tg_base_class(times,meta_data=meta_data,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path)

        if max_z_sep is None:
            max_z_sep=int(NCL[2]/2)
        elif max_z_sep>NCL[2]:
            raise ValueError("\033[1;32 Variable max_z_sep must be less than half NCL3 in readdata file\n")
        if max_x_sep is None:
            max_x_sep=int(NCL[0]/2)
        elif max_x_sep>NCL[0]:
            raise ValueError("\033[1;32 Variable max_x_sep must be less than half NCL3 in readdata file\n")
        shape_x = (max_x_sep,NCL[1],len(times))
        shape_z = (max_z_sep,NCL[1],len(times))
        for timing in times:
            fluct_data = self._module._fluct_tg_class(timing,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            if 'R_z' not in locals():
                R_z, R_x = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_R_z, local_R_x = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                R_z = np.vstack([R_z,local_R_z])
                R_x = np.vstack([R_x,local_R_x])
            gc.collect()

        R_z = R_z.T.reshape(shape_z)
        R_x = R_x.T.reshape(shape_x)
        
        autocorrDF = cd.datastruct({'x':R_x,'z':R_z})
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
    
    def _hdf_extract(self,file_name, group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov_tg'
        hdf_file = h5py.File(file_name,'r')
        shape_x = tuple(hdf_file[base_name].attrs["shape_x"][:])
        shape_z = tuple(hdf_file[base_name].attrs["shape_z"][:])
        comp = tuple(np.char.decode(hdf_file[base_name].attrs["comp"][:]))
        hdf_file.close()        

        autocorrDF = cd.datastruct.from_hdf(file_name,shapes=(shape_x,shape_z),key=base_name+'/autocorrDF')#pd.read_hdf(file_name,key=base_name+'/autocorrDF').data([shape_x,shape_z])
        meta_data = self._module._meta_class.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = self._module._avg_tg_base_class.from_hdf(file_name,base_name+'/avg_data')
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
    
    def save_hdf(self, file_name, write_mode, group_name=''):
        if not group_name:
            group_name = 'CHAPSim_autocov_tg'
        super().save_hdf(file_name, write_mode, group_name=group_name)
    
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

    @staticmethod
    @numba.njit(parallel=True,fastmath=True)
    def _autocov_calc_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        R_z = np.zeros((max_z_step,NCL2,NCL1))
        if max_z_step >0:
            for iz0 in numba.prange(max_z_step):
                for iz in numba.prange(NCL3-max_z_step):
                    R_z[iz0] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
        return R_z
    @staticmethod
    @numba.njit(parallel=True,fastmath=True)
    def _autocov_calc_x(fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step):
        
        R_x = np.zeros((max_x_step,NCL2,NCL1-max_x_step))
        if max_x_step >0:
            for ix0 in numba.prange(max_x_step):
                for ix in numba.prange(NCL1-max_x_step):
                    for iz in numba.prange(NCL3):
                        R_x[ix0,:,ix] += fluct1[iz,:,ix]*fluct2[iz,:,ix0+ix]
        return R_x

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