from CHAPSim_post.post._common import Common
from CHAPSim_post import utils, rcParams
import CHAPSim_post.post as cp
import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd
import numpy as np
import pyfftw
from abc import ABC, abstractmethod
from pyfftw.interfaces import numpy_fft
from time import perf_counter

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
pyfftw.interfaces.cache.enable()
        
def _compute_spectra_tg(fluct1, fluct2, fft_axis,mean_axis):
             
    fluct_hat1 =numpy_fft.rfft(fluct1,axis=fft_axis)
    
    if fluct2 is None:
        return (fluct_hat1*fluct_hat1.conj()).mean(mean_axis)
    else:
        fluct_hat2 = numpy_fft.rfft(fluct2,axis=fft_axis)
        return (fluct_hat1*fluct_hat2.conj()).mean(mean_axis)

def _compute_spectra_io(fluct1, fluct2, fft_axis):
    fluct_hat1 =numpy_fft.rfft(fluct1,axis=fft_axis)
    
    if fluct2 is None:
        return fluct_hat1*fluct_hat1.conj()
    else:
        fluct_hat2 = numpy_fft.rfft(fluct2,axis=fft_axis)
        return fluct_hat1*fluct_hat2.conj()

def _test_symmetry(array,axis):
    indexer = [slice(None)]*len(array.shape)
    indexer[axis] = slice(None,None,-1)
    
    if np.array_equal(array, array[indexer]):
        msg = "Symmetry averaging has failed"
        raise RuntimeError(msg)
    elif rcParams['TEST']:
        msg = "Symmetry average passed"
        print(msg)
class _Spectra_base(Common):
    def __init__(self,*args,from_hdf=False,**kwargs):
        if from_hdf:
            self._hdf_extract(*args,**kwargs)
        else:
            self._spectra_extract(*args,**kwargs)
    
    @abstractmethod
    def save_hdf(self,filename,mode,key=None):
        if key is None:
            key = self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(filename,mode,key=key)
        self._avg_data.save_hdf(filename,'a',key=key + '/avg_data')
        hdf_obj.attrs['comp'] = self._comp
        
        
    @classmethod
    def from_hdf(cls,filename,key=None):
        return cls(filename,key=key,from_hdf=True)
        
    @abstractmethod
    def _hdf_extract(self,filename,key=None):
        if key is None:
            key = self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(filename,'r',key=key)
        self._comp = hdf_obj.attrs['comp']
    
    @property
    def _meta_data(self):
        return self._avg_data._meta_data
        
    @property
    def shape(self):
        return self.E_zDF.shape

_avg_io_class = cp.CHAPSim_AVG_io
_avg_tg_class = cp.CHAPSim_AVG_tg
_avg_temp_class = cp.CHAPSim_AVG_temp

_fluct_io_class = cp.CHAPSim_fluct_io
_fluct_temp_class = cp.CHAPSim_fluct_temp

class Spectra1D_io(_Spectra_base):
    def _spectra_extract(self,comp, path_to_folder,time0=None):
             
        times = utils.time_extract(path_to_folder)
        if time0 is not None:
            times = list(filter(lambda x: x > time0,times))
        
        if rcParams['TEST']:
            times = times[-10:]
        
        self._avg_data = self._module._avg_io_class(times[-1],path_to_folder,time0=time0)
        self._comp = comp        

        for i, time in enumerate(times):
            time1 = perf_counter()
            fluct1, fluct2 = self._fluct_calc(time,path_to_folder)
            
            time2 = perf_counter()
            
            coe1 = i/(i+1)
            coe2 = 1/(i+1)
            
            pre_spectra_z = _compute_spectra_io(fluct1, fluct2, 0)
            
            if i == 0:
                spectra_z = np.zeros_like(pre_spectra_z)

            spectra_z = coe1*spectra_z + coe2*pre_spectra_z
            print(f"Time step {i+1} or {len(times)}. Extraction: "
                    f"{time2 - time1}s. Calculation {perf_counter() - time2}")
                    
        if cp.rcParams['SymmetryAVG'] and self.Domain.is_channel:
            v_count = self._comp.count('v')
            sign = (-1)**(v_count)
            
            spectra_z = 0.5*(spectra_z + sign*spectra_z[:,::-1])
            _test_symmetry(spectra_z,axis=1)
        z_array = self._avg_data.Coord_ND_DF['z']
        k_z = 2. * np.pi/(z_array[-1])*np.arange(1,spectra_z.shape[0]+1)
        
        coorddata = self._coorddata.copy()
        
        coorddata.coord_staggered = None
        coorddata.coord_centered['k_z'] = k_z
        del coorddata.coord_centered['z']
        
        self.E_zDF = cd.FlowStructND(coorddata,
                                    {(times[-1], comp) : spectra_z},
                                    data_layout = ['k_z', 'y', 'x'],
                                    wall_normal_line = 'y',
                                    polar_plane=None)
                                    
    def save_hdf(self,filename,mode,key=None):
        if key is None:
            key = self.__class__.__name__
            
        super().save_hdf(filename,mode,key=key)
          
        self.E_zDF.to_hdf(filename,'a',key=key+'/E_zDF')
    
    def _hdf_extract(self,filename,key=None):
        if key is None:
            key = self.__class__.__name__
                
        super()._hdf_extract(filename,key=key)
        
        self._avg_data = self._module._avg_io_class.from_hdf(filename,key=key+'/avg_data')
        self.E_zDF = cd.FlowStructND.from_hdf(filename,key=key+'/E_zDF')
        
    @abstractmethod                                                   
    def _fluct_calc(self,time,path_to_folder):
        pass

    def plot_spectra_contour(self,x_loc,time=None,wavelength=False,contour_kw=None,fig=None,ax=None):
                
        contour_kw = cplt.update_contour_kw(contour_kw)
        if 'plot_func' not in contour_kw:
            contour_kw['plot_func'] = 'contourf'
            
        time = self.E_zDF.check_times(time)
            
        fstruct = self.E_zDF.slice[:,:,x_loc]
        
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        return fstruct.plot_contour(self._comp,
                                    time=time,
                                    transform_xdata=x_transform,
                                    contour_kw=contour_kw,
                                    fig=fig,
                                    ax=ax)

class Spectra1D_tg(_Spectra_base, ABC):   
    def _spectra_extract(self,comp, path_to_folder,time0=None):
             
        times = utils.time_extract(path_to_folder)
        if time0 is not None:
            times = list(filter(lambda x: x > time0,times))
        
        if rcParams['TEST']:
            times = times[-10:]
        
        self._avg_data = self._module._avg_tg_class(times[-1],path_to_folder,time0=time0)
        self._comp = comp        

        for i, time in enumerate(times):
            time1 = perf_counter()
            fluct1, fluct2 = self._fluct_calc(time,path_to_folder)
            
            time2 = perf_counter()
            
            coe1 = i/(i+1)
            coe2 = 1/(i+1)
            
            pre_spectra_z = _compute_spectra_tg(fluct1, fluct2, 0,2)
            pre_spectra_x = _compute_spectra_tg(fluct1, fluct2, 2,0)
            
            if i == 0:
                spectra_z = np.zeros_like(pre_spectra_z)
                spectra_x = np.zeros_like(pre_spectra_x)

            spectra_z = coe1*spectra_z + coe2*pre_spectra_z
            spectra_x = coe1*spectra_x + coe2*pre_spectra_x
            
            print(f"Time step {i+1} or {len(times)}. Extraction: "
                    f"{time2 - time1}s. Calculation {perf_counter() - time2}")
                    
        if cp.rcParams['SymmetryAVG'] and self.Domain.is_channel:
            v_count = self._comp.count('v')
            sign = (-1)**(v_count)
            
            spectra_z = 0.5*(spectra_z + sign*spectra_z[:,::-1])
            spectra_x = 0.5*(spectra_x + sign*spectra_x[:,::-1])
            _test_symmetry(spectra_z,axis=1)
            _test_symmetry(spectra_x,axis=1)
        z_array = self.Coord_ND_DF['z']
        k_z = 2. * np.pi/(z_array[-1])*np.arange(1,spectra_z.shape[1]+1)
        
        
        coorddata_z = self._coorddata.copy()
        coorddata_z.coord_staggered = None
        coorddata_z.coord_centered['k_z'] = k_z
        del coorddata_z.coord_centered['z']
        del coorddata_z.coord_centered['x']
        
        self.E_zDF = cd.FlowStructND(coorddata_z,
                                    {(times[-1],comp) : spectra_z},
                                    data_layout = ['k_z', 'y'],
                                    wall_normal_line = 'y',
                                    polar_plane=None)
                                    
        x_array = self.Coord_ND_DF['x']
        k_x = 2. * np.pi/(x_array[-1])*np.arange(1,spectra_x.shape[1]+1)
        coorddata_x = self._coorddata.copy()
        
        coorddata_x = self._coorddata.copy()
        coorddata_x.coord_staggered = None
        coorddata_x.coord_centered['k_x'] = k_x
        del coorddata_x.coord_centered['x']
        del coorddata_x.coord_centered['z']
        
        self.E_xDF = cd.FlowStructND(coorddata_x,
                                    {(times[-1],comp) : spectra_x},
                                    data_layout = ['k_x', 'y'],
                                    wall_normal_line = 'y',
                                    polar_plane=None)
    def save_hdf(self,filename,mode,key=None):
        if key is None:
            key = self.__class__.__name__
            
        super().save_hdf(filename,mode,key=key)
          
        self.E_zDF.to_hdf(filename,'a',key=key+'/E_zDF')
        self.E_xDF.to_hdf(filename,'a',key=key+'/E_xDF')
        
    def _hdf_extract(self,filename,key=None):
        if key is None:
            key = self.__class__.__name__
                
        super()._hdf_extract(filename,key=key)
        
        self._avg_data = self._module._avg_temp_class.from_hdf(filename,key=key+'/avg_data')
                
        self.E_zDF = cd.FlowStructND.from_hdf(filename,key=key+'/E_zDF')
        self.E_xDF = cd.FlowStructND.from_hdf(filename,key=key+'/E_xDF')            
                                
class Spectra1D_temp(_Spectra_base, ABC):   
    def _spectra_extract(self,comp, path_to_folder,time0=None):
             
        times = utils.time_extract(path_to_folder)
        if time0 is not None:
            times = list(filter(lambda x: x > time0,times))
        
        if rcParams['TEST']:
            times = times[-10:]

        self._avg_data = self._module._avg_temp_class(path_to_folder,time0=time0,PhyTimes=times)
        self._comp = comp    
        
        spectra_z = []
        spectra_x = []
        for i, time in enumerate(times):
            time1 = perf_counter()
            fluct1, fluct2 = self._fluct_calc(time,path_to_folder)
            
            time2 = perf_counter()
            spectra_z.append(_compute_spectra_tg(fluct1, fluct2, 0, 2))
            spectra_x.append(_compute_spectra_tg(fluct1, fluct2, 2, 0).T)
            
            print(f"Time step {i+1} or {len(times)}. Extraction: "
                    f"{time2 - time1}s. Calculation {perf_counter() - time2}")
        spectra_z = np.stack(spectra_z,axis=0)
        spectra_x = np.stack(spectra_x,axis=0)
        
        if cp.rcParams['SymmetryAVG'] and self.Domain.is_channel:
            v_count = self._comp.count('v')
            sign = (-1)**(v_count)
            
            spectra_z = 0.5*(spectra_z + sign*spectra_z[:,::-1])
            spectra_x = 0.5*(spectra_x + sign*spectra_x[:,::-1])
            
            _test_symmetry(spectra_z,axis=1)
            _test_symmetry(spectra_x,axis=1)
            
        
        z_array = self.Coord_ND_DF['z']
        k_z = 2. * np.pi/(z_array[-1])*np.arange(1,spectra_z.shape[1]+1)
        
        
        coorddata_z = self._coorddata.copy()
        coorddata_z.coord_staggered = None
        coorddata_z.coord_centered['k_z'] = k_z
        d_z = np.diff(z_array)[0]

        del coorddata_z.coord_centered['z']
        del coorddata_z.coord_centered['x']
        
        self.E_zDF = cd.FlowStructND_time(coorddata_z,
                                    spectra_z*d_z*d_z,
                                    index=[times,[comp]*len(times)],
                                    data_layout = ['k_z', 'y'],
                                    wall_normal_line = 'y',
                                    polar_plane=None)
                                    
        x_array = self.Coord_ND_DF['x']
        k_x = 2. * np.pi/(x_array[-1])*np.arange(1,spectra_x.shape[1]+1)
        coorddata_x = self._coorddata.copy()
        
        coorddata_x = self._coorddata.copy()
        coorddata_x.coord_staggered = None
        coorddata_x.coord_centered['k_x'] = k_x
        d_x = np.diff(z_array)[0]


        del coorddata_x.coord_centered['x']
        del coorddata_x.coord_centered['z']
        
        self.E_xDF = cd.FlowStructND_time(coorddata_x,
                                    spectra_x*d_x*d_x,
                                    index=[times,[comp]*len(times)],
                                    data_layout = ['k_x', 'y'],
                                    wall_normal_line = 'y',
                                    polar_plane=None)
                                    
    def save_hdf(self,filename,mode,key=None):
        if key is None:
            key = self.__class__.__name__
            
        super().save_hdf(filename,mode,key=key)
          
        self.E_zDF.to_hdf(filename,'a',key=key+'/E_zDF')
        self.E_xDF.to_hdf(filename,'a',key=key+'/E_xDF')
        
    def _hdf_extract(self,filename,key=None):
        if key is None:
            key = self.__class__.__name__
                
        super()._hdf_extract(filename,key=key)
        
        self._avg_data = self._module._avg_temp_class.from_hdf(filename,key=key+'/avg_data')
                
        self.E_zDF = cd.FlowStructND_time.from_hdf(filename,key=key+'/E_zDF')
        self.E_xDF = cd.FlowStructND_time.from_hdf(filename,key=key+'/E_xDF')
                    
    @abstractmethod
    def _fluct_calc(self,time,path_to_folder):
        pass
        
    def plot_spectra_contour(self,direction,time=None,wavelength=False,contour_kw=None,fig=None,ax=None):
        if direction not in ['x','z']:
            msg = f"direction must be x or z not {direction}"
            raise ValueError(msg)
        
        contour_kw = cplt.update_contour_kw(contour_kw)
        if 'plot_func' not in contour_kw:
            contour_kw['plot_func'] = 'contourf'
            
        time = self.E_zDF.check_times(time)
            
        fstruct = self.E_zDF if direction =='z' else self.E_xDF
        
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        return fstruct.plot_contour(self._comp,
                                    time=time,
                                    rotate=True,
                                    transform_xdata=x_transform,
                                    contour_kw=contour_kw,
                                    fig=fig,
                                    ax=ax)
                                    
    def plot_spectra_line_t(self,direction,y_val,times,wavelength=False,y_mode='half-channel', line_kw=None,fig=None,ax=None):
        if direction not in ['x','z']:
            msg = f"direction must be x or z not {direction}"
            raise ValueError(msg)
            
        times = utils.check_list_vals(times)
        
        labels = [ "$t^*=%.1f$"%time for time in times ]
        
        x_transform = (lambda x: 2*np.pi/x) if wavelength else None
        
        fstruct = self.E_zDF.slice[:,y_val] if direction =='z' else self.E_xDF.slice[:,y_val]
        
        for i, time in enumerate(times):
            fig, ax = fstruct.plot_line(self._comp,
                                        time=time,
                                        label=labels[i],
                                        transform_xdata=x_transform,
                                        fig=fig,
                                        ax=ax,
                                        line_kw=line_kw)
                                        
        return fig, ax
        
            
        

       
class velocitySpectra1D_io(Spectra1D_io):            
    def _fluct_calc(self,time,path_to_folder):
        fluct_data = self._module._fluct_io_class(time,path_to_folder=path_to_folder,avg_data=self._avg_data)
        
        comp1, comp2 = self._comp
        if comp1 == comp2:
            return fluct_data.fluctDF[time,comp1], None
        else:
            return fluct_data.fluctDF[time,comp1], fluct_data.fluctDF[time,comp2]
                    
class velocitySpectra1D_temp(Spectra1D_temp):
    def _fluct_calc(self,time,path_to_folder):
        fluct_data = self._module._fluct_temp_class(time,path_to_folder=path_to_folder,avg_data=self._avg_data)
        
        comp1, comp2 = self._comp
        if comp1 == comp2:
            return fluct_data.fluctDF[time,comp1], None
        else:
            return fluct_data.fluctDF[time,comp1], fluct_data.fluctDF[time,comp2]        
