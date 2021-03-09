
import sys
import numpy as np
import h5py
import CHAPSim_post.post as cp
import CHAPSim_post.dtypes as cd
import CHAPSim_post.plot as cplt
import matplotlib as mpl

from CHAPSim_post import rcParams
from CHAPSim_post.utils import misc_utils,indexing
from CHAPSim_post.post._common import common3D, Common

_avg_class = cp.CHAPSim_AVG_io
_fluct_class = cp.CHAPSim_fluct_io

class _PODbase:

    def __init__(self,*args,fromfile=False,**kwargs):

        if fromfile:
            self._hdf_extract(*args,**kwargs)
        else:
            self._POD_extract(*args,**kwargs)

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = "POD"
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(key)
        group.create_dataset("eig_values",data=self._eig_values)
        hdf_file.close()
        self.POD_modesDF.to_hdf(file_name,key = key+"/POD_modesDF",mode='a')
        self.avg_data.save_hdf(file_name,'a',key+"/avg_data")

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = "POD"
        hdf_file = h5py.File(file_name,'r')
        self._eig_values = hdf_file[key+"/eig_values"][:]
        hdf_file.close()

        self.POD_modesDF = cd.datastruct.from_hdf(file_name,key=key+'/POD_modesDF')
        self.avg_data = self._module._avg_class.from_hdf(file_name,key=key+"/avg_data")
        self.CoordDF = self.avg_data.CoordDF
        self.meta_data = self.avg_data._meta_data

    @classmethod
    def from_hdf(cls,file_name,key=None):
        return cls(file_name,key,fromfile=True)

    def _getmodesSVD(self,X,nmodes):
        U, s, Vh = np.linalg.svd(X,full_matrices=False)

        return U[:,:nmodes], s**2

    def _performSVD(self,comp,path_to_folder,abs_path,avg_data,times,nmodes,*args):

        X, shape = self._get_X_matrix(comp,path_to_folder,abs_path,avg_data,times,*args)

        U, s = self._getmodesSVD(X,nmodes)
        return U.reshape((len(comp),*shape,nmodes)), s

    @property
    def EigenVals(self):
        return self._eig_values

    def _performSnapShots(self,comp,path_to_folder,abs_path,low_memory,avg_data,times,nmodes,*args):
        
        XtX = self._getTempCorrelation(comp,path_to_folder,abs_path,low_memory,avg_data,times,*args)
        eig_vals, RSingMat = np.linalg.eigh(XtX)
        
        sort_array = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_array]
        RSingMat = RSingMat[:,sort_array]

        PODmodes = self._getmodesSnapshot(eig_vals,RSingMat,low_memory,comp,path_to_folder,abs_path,avg_data,times,nmodes,*args)

        return PODmodes, eig_vals

    def _getmodesSnapshot(self,eig_vals,RSingMat,low_memory,comp,path_to_folder,abs_path,avg_data,times,nmodes,*args):
        if low_memory:
            POD_modes=None; shape = None
            for i, time in enumerate(times):
                fluct_array, shape = self._get_fluct_array(time,comp,path_to_folder,abs_path,avg_data,*args)
                if i ==0:
                    POD_modes = np.zeros((fluct_array.size,nmodes))
                for j in range(nmodes):
                    POD_modes[:,j] += fluct_array*RSingMat[i,j]/np.sqrt(eig_vals[j])
        else:
            X, shape = self._get_X_matrix(comp,path_to_folder,abs_path,avg_data,times,*args)
            POD_modes = np.matmul(X,RSingMat)
            for j in range(eig_vals.size):
                POD_modes[:,j] = POD_modes[:,j]/np.sqrt(eig_vals[j])

        return POD_modes[:,:nmodes].reshape(len(comp),*shape,nmodes)
    
    def _get_X_matrix(self,comp,path_to_folder,abs_path,avg_data,times,*args):
        X = []
        shape = None
        for time in times:
            X_i, shape = self._get_fluct_array(time,comp,path_to_folder,abs_path,avg_data,*args)

            X.append(X_i)

        return np.stack(X,axis=-1), shape

    def _getTempCorrelation(self,comp,path_to_folder,abs_path,low_memory,avg_data,times,*args):
        if low_memory:
            XtX = np.zeros((len(times),len(times)))
            
            for i,time1 in enumerate(times):
                X = []
                X_1,_ = self._get_fluct_array(time1,comp,path_to_folder,abs_path,avg_data,*args)
                X.append(np.inner(X_1,X_1))
                for j,time2 in enumerate(times[(i+1):]):
                    X_2,_ = self._get_fluct_array(time2,comp,path_to_folder,abs_path,avg_data,*args)
                    X.append(np.inner(X_1,X_2))

                X = np.array(X)

                XtX[i,i:] = X
                XtX[i:,i] = X
        else:
            X, _ = self._get_X_matrix(comp,path_to_folder,abs_path,avg_data,times,*args)
            XtX = np.matmul(X.T,X)

        assert np.allclose(XtX,XtX.T), "There is a problem, the temporal correlation is not symmetric"
        return XtX

    def plot_energy_levels(self,n_modes,fig=None, ax=None,**kwargs):

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        total_energy = np.sum(self.EigenVals)
        energy_prop = self.EigenVals[:n_modes]/total_energy
        modes = [x+1 for x in range(n_modes)]
        ax.bar(modes,energy_prop)
        ax.set_xlabel(r"Modes")
        ax.set_ylabel("Proportion of energy")

        return fig, ax

class POD2D(_PODbase,Common):
    def _POD_extract(self,comp,plane,location,path_to_folder='.',method='svd',low_memory=True,abs_path=True,time0=None,y_mode='half-channel',nsnapshots=100,nmodes=10):
        max_time = misc_utils.max_time_calc(path_to_folder,abs_path)
        self.avg_data = self._module._avg_class(max_time,path_to_folder=path_to_folder,
                                            abs_path=abs_path,time0=time0)
        times = misc_utils.time_extract(path_to_folder,abs_path)

        self._plane = plane

        if time0:
            times = list(filter(lambda x: x > time0, times))

        times = times[-nsnapshots:]

        if rcParams['TEST']:
            times = times[-12:]
        
        if nmodes > len(times):
            nmodes = len(times)

        if method.lower() == "svd":
            PODmodes, self._eig_values = self._performSVD(comp,path_to_folder,abs_path,self.avg_data,times,nmodes,y_mode,plane,location)
        elif method.lower() == "snapshots":
            PODmodes, self._eig_values = self._performSnapShots(comp,path_to_folder,abs_path,low_memory,self.avg_data,times,nmodes,y_mode,plane,location)
        else:
            msg = f"Method selected ({method}) is not valid"
            raise ValueError(msg)

        index = [[None]*len(comp), list(comp)]
        self.POD_modesDF = cd.datastruct(PODmodes,index=index)

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        super()._hdf_extract(file_name,key)

        hdf_file = h5py.File(file_name,'r')
        self._plane = hdf_file[key].attrs['plane'].decode('utf-8')
        hdf_file.close()
    
    def save_hdf(self, file_name, write_mode, key):
        if key is None:
            key = self.__class__.__name__

        super().save_hdf(file_name, write_mode, key=key)

        hdf_file = h5py.File(file_name,'a')
        hdf_file[key].attrs["plane"] = self._plane.encode('utf-8')
        hdf_file.close()

    def _get_fluct_array(self,time,comp,path_to_folder,abs_path,avg_data,y_mode,plane,location):
            fluct_data = self._module._fluct_class(time,avg_data,path_to_folder,abs_path)
            plane, coord, axis_index = indexing.contour_plane(plane,location,avg_data,y_mode,time)

            X_i = []
            shape= None
            for char in comp:
                fluct_array1 = indexing.contour_indexer(fluct_data.fluctDF[time,char],
                                                        axis_index,coord)
                shape = fluct_array1.shape
                X_i.append(fluct_array1.flatten())
            X_i = np.concatenate(X_i)
            return X_i, shape
    
    def plot_mode_contour(self,comp,modes,fig=None,ax=None,pcolor_kw=None,**kwargs):
        modes = misc_utils.check_list_vals(modes)
        ax_layout = len(modes)
        figsize=[10,3*len(modes)]
        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(ax_layout,fig=fig,ax=ax,**kwargs)

        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        x_coords = self.avg_data.CoordDF[self._plane[0]]
        y_coords = self.avg_data.CoordDF[self._plane[1]]

        X, Y = np.meshgrid(x_coords,y_coords)
        PODmodes = self.POD_modesDF[None,comp]

        coord = list('xyz')
        coord.remove(self._plane[0])
        coord.remove(self._plane[1])

        for i, mode in enumerate(modes):

            ax[i] = ax[i].pcolormesh(X,Y,PODmodes[:,:,mode],**pcolor_kw)
            
            ax[i].axes.set_xlabel(r"$%s/\delta$"%self._plane[0])
            ax[i].axes.set_ylabel(r"$%s/\delta$"%self._plane[1])
            ax[i].axes.set_title(r"Mode %d"%(mode+1),loc='left')
            ax[i].axes.set_title(r"$%s/\delta$"%coord[0],loc='right')

        return fig, ax

class POD3D(_PODbase,Common):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        Common.__init__(self,self.meta_data)
    def _POD_extract(self,comp,path_to_folder='.',method='svd',low_memory=True,abs_path=True,time0=None,nsnapshots=100,nmodes=10):
        max_time = misc_utils.max_time_calc(path_to_folder,abs_path)
        self.avg_data = self._module._avg_class(max_time,path_to_folder=path_to_folder,
                                            abs_path=abs_path,time0=time0)
        self.CoordDF = self.avg_data.CoordDF
        self.meta_data = self.avg_data._meta_data

        times = misc_utils.time_extract(path_to_folder,abs_path)

        if time0:
            times = list(filter(lambda x: x > time0, times))

        times = times[-nsnapshots:]

        if rcParams['TEST']:
            times = times[-7:]
            nmodes = 7

        if nmodes > len(times):
            nmodes = len(times)

        if method.lower() == "svd":
            PODmodes, self._eig_values = self._performSVD(comp,path_to_folder,abs_path,self.avg_data,times,nmodes)
        elif method.lower() == "snapshots":
            PODmodes, self._eig_values = self._performSnapShots(comp,path_to_folder,abs_path,low_memory,self.avg_data,times,nmodes)
        else:
            msg = f"Method selected ({method}) is not valid"
            raise ValueError(msg)

        index = [[None]*len(comp), list(comp)]
        self.POD_modesDF = cd.datastruct(PODmodes,index=index)

    
    def _get_fluct_array(self,time,comp,path_to_folder,abs_path,avg_data):
        shape = None
        fluct_data = self._module._fluct_class(time,avg_data,path_to_folder,abs_path)

        X_i = []
        for char in comp:
            fluct_array = fluct_data.fluctDF[time,char]
            shape = fluct_array.shape
            X_i.append(fluct_array.flatten())

        X_i = np.concatenate(X_i)
        return X_i, shape

    def _check_outer(self,ProcessDF,PhyTime):
        return PhyTime

    def plot_mode_contour(self,comp,modes,axis_val,plane='xz',y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        modes = misc_utils.check_list_vals(modes)
        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        axis_val = misc_utils.check_list_vals(axis_val)
        
        if len(axis_val) > 1:
            msg = "This routine can only process one slice in a single call"
            raise ValueError(msg)

        for i,mode in enumerate(modes):

            PODmodes = self.POD_modesDF.values[:,:,:,:,mode]
            POD_modeDF = cd.flowstruct3D(self.CoordDF,PODmodes,index=self.POD_modesDF.index)
            plane, coord = POD_modeDF.CoordDF.check_plane(plane)

            if coord == 'y':
                val = indexing.ycoords_from_coords(self.avg_data,axis_val,mode=y_mode)[0][0]
                int_val = indexing.ycoords_from_norm_coords(self.avg_data,axis_val,mode=y_mode)[0][0]   
            else:
                int_val = val = indexing.true_coords_from_coords(self.CoordDF,coord,axis_val)[0]

            if i == 0:
                x_size, z_size = POD_modeDF.get_unit_figsize(plane)
                figsize=[x_size,z_size*len(modes)]
                kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
                fig, ax = cplt.create_fig_ax_without_squeeze(len(modes),fig=fig,ax=ax,**kwargs)

            title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

            fig, ax[i] = POD_modeDF.plot_contour(comp,plane,int_val,time=None,fig=fig,ax=ax[i],pcolor_kw=pcolor_kw)
            ax[i].axes.set_xlabel(r"$%s/\delta$" % plane[0])
            ax[i].axes.set_ylabel(r"$%s/\delta$" % plane[1])
            ax[i].axes.set_title(r"$%s=%.2g$"%(title_symbol,val),loc='right')
            ax[i].axes.set_title(r"Mode %d"%(mode+1),loc='left')
            cbar=fig.colorbar(ax[i],ax=ax[i].axes)
            ax[i].axes.set_aspect('equal')

                    
        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
