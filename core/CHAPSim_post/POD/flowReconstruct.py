import sys
import numpy as np
import h5py
import matplotlib as mpl

from . import POD
from CHAPSim_post.utils import indexing
import CHAPSim_post.dtypes as cd
import CHAPSim_post.plot as cplt

from CHAPSim_post import rcParams
from CHAPSim_post.post._fluct import CHAPSim_fluct_base
from CHAPSim_post.post._common import Common
from CHAPSim_post.utils import misc_utils

_fluct_class = POD._fluct_class
_avg_class = POD._avg_class

_POD2D_class = POD.POD2D
_POD3D_class = POD.POD3D

class flowReconstructBase():

    def __init__(self,*args,fromfile=False,**kwargs):

        if fromfile:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extractPOD(*args,**kwargs)

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)
        hdf_obj.create_dataset("POD_coeffs",data=np.array(self.POD_coeffs))

        POD_class = self._POD.__class__.__name__
        self._POD.save_hdf(file_name,'a',key+f"/{POD_class}")

        # self.POD_modesDF.to_hdf(file_name,key=key+'/POD_modesDF',mode='a')
        # self.meta_data.save_hdf(file_name,'a',key+'/meta_data')
        # self.avg_data.save_hdf(file_name,'a',key+'/avg_data')

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._POD_coeffs = hdf_obj["POD_coeffs"][:]

    @property
    def POD(self):
        return self._POD

    @property
    def POD_coeffs(self):
        return self._POD_coeffs

    @classmethod
    def from_hdf(cls,file_name,key=None):
        return cls(file_name,key,fromfile=True)

    def plot_coeffs(self,n_modes,fig=None,ax=None,**kwargs):
        if n_modes > self.POD_coeffs.size:
            msg = "The number of modes must be less than the number of available coefficients"
            raise ValueError(msg)
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs) 
        modes = [x+1 for x in range(n_modes)]
        ax.bar(modes,self.POD_coeffs[:n_modes])
        ax.set_xlabel(r"Modes")
        ax.set_ylabel(r"Coefficient $a$")

        return fig, ax

class flowReconstruct2D(flowReconstructBase,Common):

    def _extractPOD(self,PhyTime,plane,location,comp=None,path_to_folder='.',method='svd',low_memory=True,abs_path=True,time0=None,
                                y_mode='half-channel',nsnapshots=100,nmodes=10):

        if comp is None:
            comp = "uvw"

        self._POD = self._module._POD2D_class(comp,plane,location,path_to_folder,method,low_memory,abs_path,time0,y_mode,nsnapshots,nmodes)

        # self.POD_modesDF = POD_.POD_modesDF
        self.avg_data = self._POD.avg_data
        self._meta_data = self.avg_data._meta_data

        fluct_array,_ = self.POD._get_fluct_array(PhyTime,comp,path_to_folder,abs_path,self.avg_data,y_mode,plane,location)

        if rcParams['TEST'] and nmodes > 12:
            nmodes=12

        POD_coeffs=[]
        for i in range(nmodes):
            fluct = fluct_array.flatten()
            POD_array = self.POD.POD_modesDF.values[:,:,:,i].flatten()
            POD_coeffs.append(np.inner(fluct,POD_array))
        self._POD_coeffs = np.array(POD_coeffs)

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        super()._hdf_extract(file_name,key)

        self._POD = POD.POD2D.from_hdf(file_name,key+"/POD2D")
        
        self.avg_data = self._POD.avg_data
        self._meta_data = self.avg_data._meta_data
     



    def plot_contour(self,comp,modes,fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)
        coeffs = self.POD_coeffs[sorted(modes)]
        reconstruct_arrays = np.multiply(self.POD.POD_modesDF[comp][:,:,modes],coeffs)
        
        flow_reconstruct = np.sum(reconstruct_arrays,axis=-1)

        x_array = self.avg_data.CoordDF[self.POD._plane[0]]
        y_array = self.avg_data.CoordDF[self.POD._plane[1]]

        X, Z = np.meshgrid(x_array, y_array)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        ax1 = ax.pcolormesh(X,Z,flow_reconstruct,**pcolor_kw)
        ax1.axes.set_xlabel(r"$%s/\delta$" % self.POD._plane[0])
        ax1.axes.set_ylabel(r"$%s/\delta$" % self.POD._plane[1])

        ax1.axes.set_aspect('equal')

        cbar=fig.colorbar(ax1,ax=ax)
        cbar.set_label(r"$%s^\prime$"%comp)

        fig.tight_layout()

        return fig, ax1


class flowReconstruct3D(flowReconstructBase,Common):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        Common.__init__(self,self._meta_data)

    def _extractPOD(self,PhyTime,comp=None,path_to_folder='.',method='svd',low_memory=True,abs_path=True,
                    subdomain=None,time0=None,nsnapshots=100,nmodes=10):
        
        if comp is None:
            comp = "uvw"

        self._POD = self._module._POD3D_class(comp,path_to_folder,method,low_memory,abs_path,time0,subdomain,nsnapshots,nmodes)

        self.avg_data = self._POD.avg_data
        self._meta_data = self.avg_data._meta_data

        fluct_array, _ = self.POD._get_fluct_array(PhyTime,comp,path_to_folder,abs_path,self.avg_data,subdomain)
        
        if rcParams['TEST'] and nmodes > 7:
            nmodes=7

        POD_coeffs=[]
        for i in range(nmodes):
            fluct = fluct_array.flatten()
            POD_array = self.POD.POD_modesDF.values[:,:,:,:,i].flatten()

            POD_coeffs.append(np.inner(fluct,POD_array))
        self._POD_coeffs = np.array(POD_coeffs)


    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        super()._hdf_extract(file_name,key)

        self._POD = POD.POD3D.from_hdf(file_name,key+"/POD3D")
        
        self.avg_data = self.POD.avg_data
        self._meta_data = self.avg_data._meta_data

    # def save_hdf(self,file_name,write_mode,key=None):
    #     if key is None:
    #         key = self.__class__.__name__

    #     super().save_hdf(file_name,write_mode,key)

    #     self._POD.save_hdf(file_name,'a',key+"/POD3D")

    def _check_outer(self,ProcessDF,PhyTime):
        return PhyTime

    def _getFluctDF(self,modes):
        coeffs = self.POD_coeffs[modes]
        indices =self.POD.POD_modesDF.index

        flow_reconstruct = np.inner(self.POD.POD_modesDF.values[:,:,:,:,modes],coeffs)

        return cd.flowstruct3D(self._coorddata,flow_reconstruct,Domain=self.Domain,index=indices)

    def plot_contour(self,comp,modes,axis_vals,plane='xz',y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        FluctDF = self._getFluctDF(modes)
        
        plane = self.Domain.out_to_in(plane)
        axis_vals = misc_utils.check_list_vals(axis_vals)

        plane, coord = FluctDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = indexing.ycoords_from_coords(self.avg_data,axis_vals,mode=y_mode)[0]
            int_vals = indexing.ycoords_from_norm_coords(self.avg_data,axis_vals,mode=y_mode)[0]
        else:
            int_vals = axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)
            # int_vals = indexing.coord_index_calc(self.CoordDF,coord,axis_vals)
        
        x_size, z_size = FluctDF.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i,val in enumerate(int_vals):
            fig, ax1 = FluctDF.plot_contour(comp,plane,val,time=None,fig=fig,ax=ax[i],pcolor_kw=pcolor_kw)
            ax1.axes.set_xlabel(r"$%s/\delta$" % plane[0])
            ax1.axes.set_ylabel(r"$%s/\delta$" % plane[1])
            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            
            cbar=fig.colorbar(ax1,ax=ax[i])
            cbar.set_label(r"$%s^\prime$"%comp)

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax


    def plot_streaks(self,comp,modes,*args,**kwargs):
    
        fluctDF = self._getFluctDF(modes)



    def plot_fluct3D_xz(self,comp,modes,y_vals,y_mode='half-channel',x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        FluctDF = self._getFluctDF(modes)

        y_vals = misc_utils.check_list_vals(y_vals)
        y_int_vals  = indexing.ycoords_from_norm_coords(self.avg_data,y_vals,mode=y_mode)[0]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False
        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'},antialiased=True)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(y_int_vals),fig=fig,ax=ax,**kwargs)


        for i, val in enumerate(y_int_vals):
            fig, ax[i] = FluctDF.plot_surf(comp,'xz',val,time=None,x_split_pair=x_split_pair,fig=fig,ax=ax[i],surf_kw=surf_kw)
            ax[i].axes.set_ylabel(r'$x/\delta$')
            ax[i].axes.set_xlabel(r'$z/\delta$')
            ax[i].axes.set_zlabel(r'$%s^\prime$'%comp)
            ax[i].axes.invert_xaxis()

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax

        
        
    def plot_vector(self,plane,modes,axis_vals,y_mode='half_channel',spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
    
        FluctDF = self._getFluctDF(modes)        
        plane = self.Domain.out_to_in(plane)
        axis_vals = misc_utils.check_list_vals(axis_vals)

        plane, coord = FluctDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = indexing.ycoords_from_coords(self.avg_data,axis_vals,mode=y_mode)[0]
            int_vals = indexing.ycoords_from_norm_coords(self.avg_data,axis_vals,mode=y_mode)[0]
        else:
            int_vals = axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)

        x_size, z_size = FluctDF.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i, val in enumerate(int_vals):
            fig, ax[i] = FluctDF.plot_vector(plane,val,time=None,spacing=spacing,scaling=scaling,
                                                    fig=fig,ax=ax[i],quiver_kw=quiver_kw)
            ax[i].axes.set_xlabel(r"$%s/\delta$"%slice[0])
            ax[i].axes.set_ylabel(r"$%s/\delta$"%slice[1])
            ax[i].axes.set_title(r"$%s = %.2g$"%(title_symbol,axis_vals[i]),loc='right')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax