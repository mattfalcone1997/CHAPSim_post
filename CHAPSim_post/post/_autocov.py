import numpy as np

import time
from abc import ABC, abstractmethod

import CHAPSim_post as cp
from CHAPSim_post.utils import parallel, misc_utils, indexing

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd
from scipy.signal import correlate
# from ._f90_ext_base import autocov_calc_z, autocov_calc_x

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg, CHAPSim_AVG_temp
_avg_io_class = CHAPSim_AVG_io
_avg_tg_class = CHAPSim_AVG_tg
_avg_temp_class = CHAPSim_AVG_temp

from ._fluct import CHAPSim_fluct_io,CHAPSim_fluct_tg, CHAPSim_fluct_temp
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg
_fluct_temp_class = CHAPSim_fluct_temp

from ._common import Common


from ._meta import CHAPSim_meta, coorddata
_meta_class = CHAPSim_meta

from CHAPSim_post._libs import autocorr_parallel
class _autocov_base(Common,ABC):
    def __init__(self,*args,from_hdf=False,**kwargs):

        if not from_hdf:
            self._autocov_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)

    @property
    def shape(self):
        raise NotImplementedError

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(from_hdf=True,*args,**kwargs)
    

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key =  self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        # hdf_obj.attrs['shape_x'] = np.array(self.shape_x)
        # hdf_obj.attrs['shape_z'] = np.array(self.shape_z)
        hdf_obj.attrs['comp'] = np.array([np.string_(x) for x in self.comp])

        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',key+'/avg_data')
        self.Rx_DF.to_hdf(file_name,key=key+'/Rx_DF',mode='a')#,format='fixed',data_columns=True)
        self.Rz_DF.to_hdf(file_name,key=key+'/Rz_DF',mode='a')

    @abstractmethod
    def _autocov_extract(self,*args,**kwargs):
        raise NotImplementedError

# class _autocov_developing(_autocov_base):
#     pass

class CHAPSim_autocov_io(_autocov_base):
    def _autocov_extract(self,comp,path_to_folder=".",time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.rcParams['TEST']:
            times.sort(); times= times[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder)
        self.comp=tuple(comp)

        self._avg_data = self._module._avg_io_class(max(times),path_to_folder,time0,abs_path)
        
        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")

        for i,timing in enumerate(times):
            time1 = time.time()
            fluct_data = self._module._fluct_io_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)

            coe3 = i/(i+1)
            coe2 = 1/(i+1)

            time2 = time.time()

            if i==0:
                R_x, R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
            else:
                local_R_x, local_R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
                if R_x.shape != local_R_x.shape or R_z.shape != local_R_z.shape:
                    msg = "There is a problem. the shapes of the local and averaged array are different"
                    raise ValueError(msg)
                R_x = R_x*coe3 + local_R_x*coe2
                R_z = R_z*coe3 + local_R_z*coe2
            print(f"{i+1}/{len(times)}", timing, time.time() - time2,time2-time1,flush=True)

        if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
            vy_count = comp.count('v')

            R_x = 0.5*(R_x + R_x[:,::-1]*(-1)**vy_count )
            R_z = 0.5*(R_z + R_z[:,::-1]*(-1)**vy_count )

        coorddata_z, coorddata_x, = self._get_coorddata(R_x,R_z)
        self.Rx_DF = cd.FlowStructND(coorddata_x,
                                    {(None,'x'):R_x},
                                    data_layout = ['delta x','y','x'],
                                    polar_plane=None,
                                    wall_normal_line='y')
        self.Rz_DF = cd.FlowStructND(coorddata_z,
                                     {(None,'z'):R_z},
                                    data_layout = ['delta z','y','x'],
                                    polar_plane=None,
                                    wall_normal_line='y')

    def _get_coorddata(self,R_x,R_z):

        x_coords_z = self.CoordDF['x']
        x_coords_x = self.CoordDF['x'][:R_x.shape[-1]]

        sep_coords_z = self.CoordDF['z'][:R_z.shape[0]]
        sep_coords_x = self.CoordDF['x'][:R_x.shape[0]]

        y_coords = self.CoordDF['y']

        x_coords_z_nd = self.Coord_ND_DF['x']
        x_coords_x_nd = self.Coord_ND_DF['x'][:R_x.shape[-1]+1]

        sep_coords_z_nd = self.Coord_ND_DF['z'][:R_z.shape[0]+1]
        sep_coords_x_nd = self.Coord_ND_DF['x'][:R_x.shape[0]+1]

        y_coord_nd = self.Coord_ND_DF['y']

        coord_z = cd.coordstruct({'delta z' : sep_coords_z, 
                                    'y' : y_coords,
                                    'x' : x_coords_z})

        coord_z_nd = cd.coordstruct({'delta z' : sep_coords_z_nd, 
                            'y' : y_coord_nd,
                            'x' : x_coords_z_nd})

        coord_x = cd.coordstruct({'delta x' : sep_coords_x, 
                            'y' : y_coords,
                            'x' : x_coords_x})

        coord_x_nd = cd.coordstruct({'delta x' : sep_coords_x_nd, 
                            'y' : y_coord_nd,
                            'x' : x_coords_x_nd})
        
        coordstruct_x = coorddata.from_coordstructs(self.Domain,
                                                    coord_x,
                                                    coord_x_nd)

        coordstruct_z = coorddata.from_coordstructs(self.Domain,
                                                    coord_z,
                                                    coord_z_nd)

        return coordstruct_z, coordstruct_x



    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.comp = tuple(np.char.decode(hdf_obj.attrs["comp"][:]))

        self.Rx_DF = cd.FlowStructND.from_hdf(file_name,key=key+'/Rx_DF')
        self.Rz_DF = cd.FlowStructND.from_hdf(file_name,key=key+'/Rz_DF')
        self._meta_data = self._module._meta_class.from_hdf(file_name,key=key+'/meta_data')
        self._avg_data = self._module._avg_io_class.from_hdf(file_name,key=key+'/avg_data')

    def _autocov_calc(self,fluct_data,comp,PhyTime,max_x_sep,max_z_sep):
        PhyTime = fluct_data.check_PhyTime(PhyTime)

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp[0]]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp[1]]

        R_x = autocorr_parallel.autocov_calc_io_x(fluct_vals1,fluct_vals2,max_x_sep)#.mean(axis=1)
        R_z = autocorr_parallel.autocov_calc_io_z(fluct_vals1,fluct_vals2,max_z_sep)
        
        return R_x, R_z

    def plot_line(self,comp,x_vals,y_vals,y_mode='half_channel',norm=True,line_kw=None,fig=None,ax=None,**kwargs):
        if not comp in ('x','z'):
            msg = f"comp must be the string 'x' or 'z' not {comp} "
            raise ValueError(msg)

        y_vals = misc_utils.check_list_vals(y_vals)
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        
        if comp == 'x':
            Ruu_DF = self.Rx_DF.copy()
        else:
            Ruu_DF = self.Rz_DF.copy()

        line_kw = cplt.update_line_kw(line_kw)

        if norm:
            Ruu_0=Ruu_DF[None,comp][0]
            Ruu_DF[None,comp]/=Ruu_0

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5*len(y_vals)])
        fig, ax, single_output = cplt.create_fig_ax_without_squeeze(len(y_vals),fig=fig,ax=ax,**kwargs)
        
        y_vals = self._avg_data.ycoords_from_coords(y_vals,mode=y_mode)[0]
        y_vals_int = self._avg_data.ycoords_from_norm_coords(y_vals,mode=y_mode)[0]

        title_symbol = misc_utils.get_title_symbol('y',y_mode,local=False)
        labels = [self.Domain.create_label(r"$x = %.3g$"%x) for x in x_vals]

        for j,y in enumerate(y_vals_int):
            for i, x in enumerate(x_vals):
                fig, ax[j] = Ruu_DF.slice[:,y,x].plot_line(comp,
                                                            label=labels[i],
                                                            fig=fig,
                                                            ax=ax[j],
                                                            line_kw=line_kw)


                ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)
                xlabel = self.Domain.create_label(r"$\Delta %s$"%comp)
                ax[j].set_xlabel(xlabel)
                ax[j].set_title(r"$%s=%.3g$"%(title_symbol,y_vals[j]),loc='right')
        
        if single_output:
            return fig, ax[0]
        else:
            return fig, ax
    
    def plot_contour_zy(self,comp,axis_vals,norm=True,show_positive=True,contour_kw=None,fig=None,ax=None,**kwargs):
        if not comp in ('x','z'):
            msg = f"comp must be the string 'x' or 'z' not {comp} "
            raise ValueError(msg)

        axis_vals = misc_utils.check_list_vals(axis_vals)
        if comp == 'x':
            Ruu_DF = self.Rx_DF[None,[comp]].copy()
        else:
            Ruu_DF = self.Rz_DF[None,[comp]].copy()
            
        contour_kw = cplt.update_contour_kw(contour_kw)
        
        if norm:
            Ruu_0=Ruu_DF[None,comp][0]
            Ruu_DF[None,comp]/=Ruu_0
        

            
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,4*len(axis_vals)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        xlabel = self.Domain.create_label(r"$\Delta %s/\delta$" %comp)
        ylabel = self.Domain.create_label(r"$y$")
        
        min_val = np.inf
        max_val = -np.inf
        
        for i, val in enumerate(axis_vals):
            
            Ruu_slice = Ruu_DF.slice[:,:,val]
            
            if not show_positive:
                Ruu_slice[None,comp] = np.ma.masked_array(Ruu_slice[None,comp])
                Ruu_slice[None,comp].mask = Ruu_slice[None,comp] > 0
                
            fig, ax[i] = Ruu_slice.plot_contour(comp,
                                                None,
                                                fig=fig,
                                                ax=ax[i],
                                                contour_kw=contour_kw)
            lmin, lmax = ax[i].get_clim()
            
            min_val = min(min_val,lmin)
            max_val = max(max_val,lmax)
            
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)
            

        for a in ax:
            a.set_clim([min_val,max_val])
            fig.colorbar(a,ax=a.axes)
            fig.tight_layout()
            
        return fig, ax
class CHAPSim_autocov_tg(_autocov_base):
    _tgpost = True
    def _autocov_extract(self,comp,path_to_folder=".",time0=None,ntimes=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if ntimes is not None:
            times = sorted(times)[-ntimes:]

        if cp.rcParams['TEST']:
            times= sorted(times)[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder,tgpost=True)
        self.comp=tuple(comp)

        self._avg_data = self._module._avg_tg_class(max(times),path_to_folder,time0,abs_path)

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")

        for i,timing in enumerate(times):
            time1 = time.time()
            fluct_data = self._module._fluct_tg_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)

            coe3 = i/(i+1)
            coe2 = 1/(i+1)
            time2 = time.time()
            if i==0:
                R_x, R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
            else:
                local_R_x, local_R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)

                R_x = R_x*coe3 + local_R_x*coe2
                R_z = R_z*coe3 + local_R_z*coe2

            print(f"{i+1}/{len(times)}", timing, time.time() - time2,time2-time1,flush=True)

        if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
            vy_count = comp.count('v')

            R_x = 0.5*(R_x + R_x[:,::-1]*pow(-1,vy_count))
            R_z = 0.5*(R_z + R_z[:,::-1]*pow(-1,vy_count))
        
        coorddata_z, coorddata_x = self._get_coorddata(R_x,R_z)
        self.Rx_DF = cd.FlowStructND(coorddata_x,
                                    {(None,'x'):R_x},
                                    data_layout = ['delta x','y'],
                                    polar_plane=None,
                                    wall_normal_line='y')

        self.Rz_DF = cd.FlowStructND(coorddata_z,{(None,'z'):R_z},
                                    data_layout = ['delta z','y'],
                                    polar_plane=None,
                                    wall_normal_line='y')

    def _get_coorddata(self,R_x,R_z):

        sep_coords_z = self.CoordDF['z'][:R_z.shape[0]]
        sep_coords_x = self.CoordDF['x'][:R_x.shape[0]]

        y_coords = self.CoordDF['y']

        sep_coords_z_nd = self.Coord_ND_DF['z'][:R_z.shape[0]+1]
        sep_coords_x_nd = self.Coord_ND_DF['x'][:R_x.shape[0]+1]

        y_coord_nd = self.Coord_ND_DF['y']

        coord_z = cd.coordstruct({'delta z' : sep_coords_z, 
                                    'y' : y_coords})

        coord_z_nd = cd.coordstruct({'delta z' : sep_coords_z_nd, 
                            'y' : y_coord_nd})

        coord_x = cd.coordstruct({'delta x' : sep_coords_x, 
                            'y' : y_coords})

        coord_x_nd = cd.coordstruct({'delta x' : sep_coords_x_nd, 
                            'y' : y_coord_nd})
        
        coorddata_x = coorddata.from_coordstructs(self.Domain,
                                                    coord_x,
                                                    coord_x_nd)

        coorddata_z = coorddata.from_coordstructs(self.Domain,
                                                    coord_z,
                                                    coord_z_nd)

        return coorddata_z, coorddata_x

    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.comp = tuple(np.char.decode(hdf_obj.attrs["comp"][:]))

        self.Rx_DF = cd.FlowStructND.from_hdf(file_name,key=key+'/Rx_DF')
        self.Rz_DF = cd.FlowStructND.from_hdf(file_name,key=key+'/Rz_DF')
        self._meta_data = self._module._meta_class.from_hdf(file_name,key=key+'/meta_data')
        self._avg_data = self._module._avg_tg_class.from_hdf(file_name,key=key+'/avg_data')

    def _autocov_calc(self,fluct_data,comp,PhyTime,max_x_sep,max_z_sep):
        PhyTime = fluct_data.check_PhyTime(PhyTime)

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp[0]]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp[1]]

        # R_x = self.autocov_calc_tg_x_np(fluct_vals1,fluct_vals2)#.mean(axis=1)
        # R_z = self.autocov_calc_tg_z_np(fluct_vals1,fluct_vals2)

        R_x = autocorr_parallel.autocov_calc_tg_x(fluct_vals1,fluct_vals2,max_x_sep)#.mean(axis=1)
        R_z = autocorr_parallel.autocov_calc_tg_z(fluct_vals1,fluct_vals2,max_z_sep)
        
        return R_x, R_z

    def plot_line(self,comp,y_vals,y_mode='half_channel',norm=True,line_kw=None,fig=None,ax=None,**kwargs):
        if not comp in ('x','z'):
            msg = f"comp must be the string 'x' or 'z' not {comp} "
            raise ValueError(msg)

        y_vals = misc_utils.check_list_vals(y_vals)
        if comp == 'x':
            Ruu_DF = self.Rx_DF.copy()
        else:
            Ruu_DF = self.Rz_DF.copy()

        line_kw = cplt.update_line_kw(line_kw)

        if norm:
            Ruu_0=Ruu_DF[None,comp][0]
            Ruu_DF[None,comp]/=Ruu_0
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5*len(y_vals)])
        fig, ax,single_output = cplt.create_fig_ax_without_squeeze(len(y_vals),fig=fig,ax=ax,**kwargs)
        
        y_vals = self._avg_data.ycoords_from_coords(y_vals,mode=y_mode)[0]
        y_vals_int = self._avg_data.ycoords_from_norm_coords(y_vals,mode=y_mode)[0]


        title_symbol = misc_utils.get_title_symbol('y',y_mode,local=False)
        for j,y in enumerate(y_vals_int):
            fig, ax[j] = Ruu_DF.slice[:,y].plot_line(comp,
                                                        fig=fig,
                                                        ax=ax[j],
                                                        line_kw=line_kw)
            ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)
            ax[j].set_xlabel(r"$\Delta %s/\delta$"%comp)
            ax[j].set_title(r"$%s=%.3g$"%(title_symbol,y_vals[j]),loc='right')

        if single_output:
            return fig, ax[0]
        else:
            return fig, ax
    
            
class CHAPSim_autocov_temp(CHAPSim_autocov_tg):
    def _autocov_extract(self,comp,path_to_folder=".",time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.rcParams['TEST']:
            times= sorted(times)[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder,tgpost=True)
        self.comp=tuple(comp)

        self._avg_data = self._module._avg_temp_class(path_to_folder,time0,abs_path,PhyTimes=times)

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")
        R_x = np.zeros((len(times),max_x_sep,self.NCL[1]))
        R_z = np.zeros((len(times),max_z_sep,self.NCL[1]))

        for i,timing in enumerate(times):
            time1 = time.time()
            fluct_data = self._module._fluct_temp_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)
            time2 = time.time()
            R_x_single_t, R_z_single_t = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)

            R_x[i,:,:] = R_x_single_t
            R_z[i,:,:] = R_z_single_t

            print(i, timing, time.time() - time2,time2-time1,flush=True)

        if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
            vy_count = comp.count('v')

            R_x = 0.5*(R_x + R_x[:,:,::-1]*(-1)**vy_count )
            R_z = 0.5*(R_z + R_z[:,:,::-1]*(-1)**vy_count )

        coorddata_z, coorddata_x, = self._get_coorddata(R_x,R_z)
        index_x = [times,['x']*len(times)]
        index_z = [times,['z']*len(times)]
        
        self.Rx_DF = cd.FlowStructND_time(coorddata_x,
                                          R_x,
                                          index = index_x,
                                          data_layout = ['delta x','y'],
                                          polar_plane=None,
                                          wall_normal_line='y')
        
        self.Rz_DF = cd.FlowStructND_time(coorddata_z,
                                          R_z,
                                          index = index_z,
                                          data_layout = ['delta z','y'],
                                          polar_plane=None,
                                          wall_normal_line='y')

    def _get_coorddata(self, R_x, R_z):
        

        sep_coords_z = self.CoordDF['z'][:R_z.shape[1]]
        sep_coords_x = self.CoordDF['x'][:R_x.shape[1]]

        y_coords = self.CoordDF['y']

        sep_coords_z_nd = self.Coord_ND_DF['z'][:R_z.shape[1]+1]
        sep_coords_x_nd = self.Coord_ND_DF['x'][:R_x.shape[1]+1]

        y_coord_nd = self.Coord_ND_DF['y']

        coord_z = cd.coordstruct({'delta z' : sep_coords_z, 
                                    'y' : y_coords})

        coord_z_nd = cd.coordstruct({'delta z' : sep_coords_z_nd, 
                            'y' : y_coord_nd})

        coord_x = cd.coordstruct({'delta x' : sep_coords_x, 
                            'y' : y_coords})

        coord_x_nd = cd.coordstruct({'delta x' : sep_coords_x_nd, 
                            'y' : y_coord_nd})
        
        coordstruct_x = coorddata.from_coordstructs(self.Domain,
                                                    coord_x,
                                                    coord_x_nd)

        coordstruct_z = coorddata.from_coordstructs(self.Domain,
                                                    coord_z,
                                                    coord_z_nd)
        return coordstruct_z, coordstruct_x

    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.comp = tuple(np.char.decode(hdf_obj.attrs["comp"][:]))

        self.Rx_DF = cd.FlowStructND.from_hdf(file_name,key=key+'/Rx_DF')
        self.Rz_DF = cd.FlowStructND.from_hdf(file_name,key=key+'/Rz_DF')
        self._meta_data = self._module._meta_class.from_hdf(file_name,key=key+'/meta_data')
        self._avg_data = self._module._avg_temp_class.from_hdf(file_name,key=key+'/avg_data')

    def plot_line(self,comp,times,y_vals,y_mode='half_channel',norm=True,line_kw=None,fig=None,ax=None,**kwargs):
        if not comp in ('x','z'):
            msg = f"comp must be the string 'x' or 'z' not {comp} "
            raise ValueError(msg)

        times = misc_utils.check_list_vals(times)
        y_vals = misc_utils.check_list_vals(y_vals)

        if comp == 'x':
            Ruu_DF = self.Rx_DF.copy()
        else:
            Ruu_DF = self.Rz_DF.copy()

        line_kw = cplt.update_line_kw(line_kw)
        if norm:
            for timing in Ruu_DF.times:
                Ruu_0=Ruu_DF[timing,comp][0]
                Ruu_DF[timing,comp]/=Ruu_0

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5*len(y_vals)])
        fig, ax, single_output = cplt.create_fig_ax_without_squeeze(len(y_vals),fig=fig,ax=ax,**kwargs)
        
        y_vals = self._avg_data.ycoords_from_coords(y_vals,mode=y_mode)[0]
        y_vals_int = self._avg_data.ycoords_from_norm_coords(y_vals,mode=y_mode)[0]

        title_symbol = misc_utils.get_title_symbol('y',y_mode,local=False)
        labels = [self.Domain.create_label(r"$t = %.3g$"%x) for x in times]

        for j,y in enumerate(y_vals_int):
            for i, time in enumerate(times):
                fig, ax[j] = Ruu_DF.slice[:,y].plot_line(comp,
                                                         time = time,
                                                         label=labels[i],
                                                         fig=fig,
                                                         ax=ax[j],
                                                         line_kw=line_kw)


                ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)
                xlabel = self.Domain.create_label(r"$\Delta %s/\delta$"%comp)
                ax[j].set_xlabel(xlabel)
                ax[j].set_title(r"$%s=%.3g$"%(title_symbol,y_vals[j]),loc='right')

        return fig, ax