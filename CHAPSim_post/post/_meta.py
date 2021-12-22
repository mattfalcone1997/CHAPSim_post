"""
# _meta.py
Class containing the meta data and coordinate arrays for the package
"""

import numpy as np
import os
import warnings
import gc
import copy
import types
import pandas as pd
from pyvista import StructuredGrid

import CHAPSim_post.dtypes as cd

from CHAPSim_post.utils import misc_utils, gradient
from CHAPSim_post import styleParams, plot as cplt

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as simps
    from scipy.integrate import cumulative_trapezoid as cumtrapz 
else:
    from scipy.integrate import simps 
    from scipy.integrate import cumtrapz

class CHAPSim_meta():
    def __init__(self,*args,from_file=False,**kwargs):

        if from_file:
            kwargs.pop('tgpost',None)
            self._hdf_extract(*args,**kwargs)
        else:
            self.__extract_meta(*args,**kwargs)

    def __extract_meta(self,path_to_folder='.',abs_path=True,tgpost=False):
        self.metaDF = self._readdata_extract(path_to_folder,abs_path)
        ioflg = self.metaDF['iDomain'] in [2,3]
        tgflg = self.metaDF['iDomain'] == 1
        # if ioflg and tgpost:
        #     self._output_info_tg = OutputFileStore_tg(path_to_folder)
        # elif ioflg and not tgpost:
        #     self._output_info_tg = OutputFileStore_io(path_to_folder)

        # if tgflg:
        #     self._output_info_tg = OutputFileStore_tg(path_to_folder)
        
        iCase = self.metaDF['iCase']
        self._coorddata = coorddata(iCase,self.metaDF,path_to_folder,abs_path,tgpost,ioflg)     

        # self.path_to_folder = path_to_folder
        # self._abs_path = abs_path

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(from_file=True,*args,**kwargs)

    @property
    def NCL(self):
        return self._coorddata.NCL

    @property
    def coord_data(self):
        return self._coorddata

    @property
    def CoordDF(self):
        return self._coorddata.centered

    @property
    def Coord_ND_DF(self):
        return self._coorddata.staggered

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = 'CHAPSim_meta'
            
        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.metaDF = cd.metastruct.from_hdf(file_name,key=key+'/metaDF')#pd.read_hdf(file_name,key=base_name+'/metaDF')

        self._coorddata = coorddata.from_hdf(file_name,key=key+"/coorddata")        
        #h5py.File(file_name,'r')



        # self.path_to_folder = hdf_file[key].attrs['path_to_folder'].decode('utf-8')
        # self._abs_path = bool(hdf_file[key].attrs['abs_path'])
        # hdf_file.close()
    
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)
        # group.attrs["path_to_folder"] = self.path_to_folder.encode('utf-8')
        # group.attrs["abs_path"] = int(self._abs_path)
        # group.create_dataset("NCL",data=self.NCL)

        self.metaDF.to_hdf(file_name,key=key+'/metaDF',mode='a')
        self._coorddata.to_hdf(file_name,key=key+'/coorddata',mode='a')

    def _readdata_extract(self,path_to_folder,abs_path):
        
        if not abs_path:
            readdata_file = os.path.abspath(os.path.join(path_to_folder,'readdata.ini'))
        else:
           readdata_file = os.path.join(path_to_folder,'readdata.ini')

        with open(readdata_file) as file_object:
            lines = (line.rstrip() for line in file_object)
            lines = list(line for line in lines if line.count('=')>0)
        
        meta_list=[]
        key_list=[]
        for line in lines:
            if line[0] in ['#',';',' ']:
                continue
            line_split = line.split(';')[0]
            key = line_split.split('=')[0]
            vals= line_split.split('=')[1]

            key_list.append(key)
            if len(vals.split(',')) ==1:
                try:
                    args = float(vals.split(',')[0])
                except ValueError:
                    args = vals.split(',')[0]
            else:
                args = [float(x) for x in vals.split(',')]
            meta_list.append(args)

        meta_DF = cd.metastruct(meta_list,index=key_list)
        return meta_DF
    
    def _coord_extract(self,path_to_folder,abs_path,tgpost,ioflg):
        if os.path.isdir(os.path.join(path_to_folder,'0_log_monitors')):
            return self._coord_extract_new(path_to_folder,abs_path,tgpost,ioflg)
        else:
            return self._coord_extract_old(path_to_folder,abs_path,tgpost,ioflg)

    def _coord_extract_new(self,path_to_folder,abs_path,tgpost,ioflg):
        full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')

        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(full_path,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(full_path,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(full_path,'CHK_COORD_ZND.dat')
    #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        x_coord=np.loadtxt(file,comments='#',usecols=1)
        
        if tgpost and not ioflg:
            XND = x_coord[:-1]
        else:
            index = int(self.metaDF['NCL1_tg']) + 1 
            if (tgpost and ioflg):
                XND = x_coord[:index]
                XND -= XND[0]
            elif self.metaDF['iCase'] == 5:
                index = int(self.metaDF['NCL1_io']) + 1
                XND = x_coord[:index]
            else:
                XND = x_coord[index:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,usecols=1,skiprows=1)
        index = int(self.metaDF['NCL2']) + 1 
        YCC=y_coord[index:]
        YND = y_coord[:index]
        y_size = YCC.size
        file.close()
        #============================================================
    
        file=open(z_coord_file,'rb')
        ZND=np.loadtxt(file,comments='#',usecols=1)

        #============================================================
        XCC, ZCC = self._coord_interp(XND,ZND)

        z_size = ZCC.size
        x_size = XCC.size
        y_size = YCC.size
        file.close()

        CoordDF = cd.datastruct({'x':XCC,'y':YCC,'z':ZCC})
        Coord_ND_DF = cd.datastruct({'x':XND,'y':YND,'z':ZND})
        NCL = [x_size, y_size, z_size]
        return CoordDF, Coord_ND_DF, NCL


    def _coord_extract_old(self,path_to_folder,abs_path,tgpost,ioflg):
        """ Function to extract the coordinates from their .dat file """
        
        full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')

        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(full_path,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(full_path,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(full_path,'CHK_COORD_ZND.dat')
        #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        x_coord=np.loadtxt(file,comments='#')
        x_size=  int(x_coord[0])
        x_coord=np.delete(x_coord,0)
        
        if tgpost and not ioflg:
            XND = x_coord[:-1]
        else:
            for i in range(x_size):
                if x_coord[i] == 0.0:
                    index=i
                    break
            if tgpost and ioflg:
                XND = x_coord[:index+1]
                XND -= XND[0]
            else:
                XND = x_coord[index+1:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,comments='#',usecols=1)
        y_size = int(y_coord[0])
        for i in range(y_coord.size):
            if y_coord[i] == 1.0:
                index=i
                break
        
        # YCC=np.delete(y_coord,np.arange(index+1))
        YND = y_coord[:index+1]
        YCC = y_coord[(index+1):]
        y_size = YCC.size
        file.close()
        #============================================================
    
        file=open(z_coord_file,'rb')
        z_coord=np.loadtxt(file,comments='#')
        z_size = int(z_coord[0])
        ZND=np.delete(z_coord,0)
        #============================================================
        XCC, ZCC = self._coord_interp(XND,ZND)

        z_size = ZCC.size
        x_size = XCC.size
        y_size = YCC.size
        file.close()

        CoordDF = cd.datastruct({'x':XCC,'y':YCC,'z':ZCC})
        Coord_ND_DF = cd.datastruct({'x':XND,'y':YND,'z':ZND})
        NCL = [x_size, y_size, z_size]
        return CoordDF, Coord_ND_DF, NCL
        
    def _coord_interp(self,XND, ZND):
        """ Interpolate the coordinates to give their cell centre values """
        XCC=np.zeros(XND.size-1)
        for i in range(XCC.size):
            XCC[i] = 0.5*(XND[i+1] + XND[i])
    
        ZCC=np.zeros(ZND.size-1)
        for i in range(ZCC.size):
            ZCC[i] = 0.5*(ZND[i+1] + ZND[i])
    
        return XCC, ZCC
    
    def copy(self):
        return copy.deepcopy(self)


class OutputFileStore_base:
    def __init__(self,*args,from_file=False,**kwargs):
        if from_file:
            self._hdf_extract(*args,**kwargs)
        else:
            self._out_extract(*args,**kwargs)

    def _out_extract(self,path_to_folder):

        if not '0_log_monitors' in os.listdir(path_to_folder):
            path = path_to_folder
            files = [os.path.join(path,x) for x in os.listdir(path_to_folder)\
                 if self._monitor_old in x]
            new = False
        else:
            path = os.path.join(path_to_folder,'0_log_monitors')
            files = [os.path.join(path,x) for x in os.listdir(path) if self._monitor_new in x]
            new = True

        dframe_list = []
        for file in sorted(files):
            dframe = self.get_dataframe(file,new=new)
            # print(dframe)
            if dframe is None or dframe.empty:
                continue

            dframe_list.append(dframe)

        self.output_data = pd.concat(dframe_list).drop_duplicates()

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        self.output_data = pd.read_hdf(file_name,key=key+"/output_data",mode='r')

    def get_dataframe(self,file_name,new=True):
        if os.path.splitext(file_name)[-1] =='.swp':
            return None

        f = open(file_name,'r')
        lines = [ line for line in f.readlines() if line[0] != '#']
        data = []
        for line in lines:
            split_line = line.split(' ')
            while '' in split_line:
                split_line.remove('')
            for i,elem in enumerate(split_line):
                split_line[i] = elem.replace('\n','')

            try:
                data.append([float(elem) for elem in split_line])
            except ValueError:
                pass
        if not data:
            return None

        dframe = pd.DataFrame( np.array(data))
        columns = self.get_col_names(file_name,new)[:dframe.shape[1]]
        index = [int(x) for x in dframe[0]]

        dframe.columns = columns
        dframe.index=index
        # print(dframe)
        return dframe
        

    def get_col_names(self,file_name,new=True):
        f = open(file_name,'rb')
        def _line_check(line):
            if os.path.splitext(file_name)[-1] =='.swp':
                return False
            line = line.decode('ascii')
            return '#' in line and 'MYID' not in line

        lines = [line.decode('ascii') for line in f.readlines() if _line_check(line)]

        if not lines:
            return []

        if new:
            assert len(lines) == 1, "Just checking"
            columns =  lines[0].replace('#','').replace('\n','').replace('"','').split(',')
        else:

            columns = lines[-1].replace('#','').replace('"','').split(',')
        return columns
    def to_hdf(self,file_name,mode='a',key=None):
        if key is None:
            key = self.__class__.__name__

        self.output_data.to_hdf(file_name,key=key+"/output_data")

    @classmethod
    def from_hdf(cls,file_name,key=None):
        return cls(file_name,from_file=True,key=key)
                
class OutputFileStore_tg(OutputFileStore_base):
    _monitor_new = "monitor.tg"
    _monitor_old = "history.periodicxz."

class OutputFileStore_io(OutputFileStore_base):
    _monitor_new = "monitor.io"
    _monitor_old = "OUTPUT_"

class DomainHandler(cd.GeomHandler):
    @staticmethod
    def get_FlowGeometry(metaDF):
        iCase = metaDF['iCase']
        if iCase in [1,4]:
            geom_type = cd.CHANNEL
        elif iCase in [2,3]:
            geom_type = cd.PIPE
        elif iCase == 5:
            geom_type = cd.BLAYER
        else:
            msg = "CHAPSim case type invalid"
            raise ValueError(msg)
        
        return geom_type
    
    
    def _alter_item(self,char,to_out=True):
        convert_dict = styleParams.cart_to_polar if to_out else styleParams.polar_to_cart
        if self.coord_sys == 'cylind' and char in convert_dict.keys():
            return convert_dict[char]
        else:
            return char
    
    def get_symbol(self,comp):
        if comp not in styleParams.cart_to_polar.keys():
            msg = "Comp must be either x, y, or z"
            raise ValueError(msg)

        if self.is_polar:
            return styleParams.CoordLabel_pipe(comp)
        else:
            return styleParams.CoordLabel_channel(comp)

    def _get_coord_symbol(self,comp):
        if comp not in ['x','y','z']:
            return comp
        if self.is_polar:
            comp = styleParams.cart_to_polar[comp]
        if self.is_polar:
            return styleParams.CoordLabel_pipe(comp)
        else:
            return styleParams.CoordLabel_channel(comp)

    def _get_velocity_symbol(self,comp):
        if comp not in ['u','v','w']:
            return comp
        if self.is_polar:
            return styleParams.cart_to_polar[comp]
        else:
            return comp

    def create_label(self, text):
        text_list = list(text)
        for i in range(len(text_list)):
            text_list[i] = self._get_coord_symbol(text_list[i])
            text_list[i] = self._get_velocity_symbol(text_list[i])

        return "".join(text_list)

    def styleYdata(self,data):
        if self.is_polar:
            return styleParams.ystyle_channel.transformy(data)
        else:
            return styleParams.ystyle_pipe.transformy(data)

    def in_to_out(self,key):
        return "".join([self._alter_item(x,True) for x in key])

    def to_internal(self,key):
        if 'theta' in key:
            key = key.split('theta')
            for i, _ in enumerate(key):
                if not key[i]:
                    key[i] ='theta'

        return "".join([self._alter_item(x,False) for x in key]) 

    def Scalar_grad_io(self,coordDF,flow_array):

        if flow_array.ndim == 2:
            grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
        elif flow_array.ndim == 3:
            grad_vector = np.zeros((3,flow_array.shape[0],flow_array.shape[1],
                                flow_array.shape[2]))
        else:
            msg = "This function can only be used on 2 or 3 dimensional arrays"
            raise ValueError(msg)

        grad_vector[0] = self.Grad_calc(coordDF,flow_array,'x')
        grad_vector[1] = self.Grad_calc(coordDF,flow_array,'y')
                
        if flow_array.ndim == 3:
            grad_vector[2] = self.Grad_calc(coordDF,flow_array,'z')

            factor_out = 1/coordDF['y'] if self.is_polar else 1.0
            grad_vector[2] = np.multiply(grad_vector[2],factor_out)

        return grad_vector

    def Vector_div_io(self,coordDF,vector_array):
        if vector_array.ndim not in (3,4):
            msg = "The number of dimension of the vector array must be 3 or 4"
            raise ValueError(msg)

        grad_vector = np.zeros_like(vector_array)
        grad_vector[0] = self.Grad_calc(coordDF,vector_array[0],'x')
        
        factor_in = coordDF['y'] if self.is_polar else 1.0
        factor_out = 1/coordDF['y'] if self.is_polar else 1.0

        grad_vector[1] = self.Grad_calc(coordDF,np.multiply(vector_array[1],factor_in),'y')
        grad_vector[1] = np.multiply(grad_vector[1],factor_out)
        
        if vector_array.ndim == 4:
            grad_vector[2] = self.Grad_calc(coordDF,
                                    np.multiply(vector_array[2],factor_in),'z')
            grad_vector[2] = np.multiply(grad_vector[2],factor_out)

        div_scalar = np.sum(grad_vector,axis=0)
        return div_scalar

    def Scalar_laplacian(self,coordDF,flow_array):
        grad_vector = self.Scalar_grad_io(coordDF,flow_array)
        lap_scalar = self.Vector_div_io(coordDF,grad_vector)
        return lap_scalar

    def Scalar_laplacian_tg(self,coordDF,flow_array):
        factor_in = coordDF['y'] if self.is_polar else 1.0
        factor_out = 1/coordDF['y'] if self.is_polar else 1.0
        dflow_dy = np.multiply(self.Grad_calc(coordDF,flow_array,'y'),
                                factor_in)
        lap_scalar = np.multiply(self.Grad_calc(coordDF,dflow_dy,'y'),
                                factor_out)
        return lap_scalar

    def IntegrateWallNormal_cumulat(self,CoordDF,flow_array):
        y_coords = CoordDF['y']
        if flow_array.ndim == 2:
            axis = 0
        elif flow_array.ndim == 3:
            axis = 1
        else:
            msg = r"This function can only be applied to arrays with dimension 2 and 3"
            raise ValueError(msg)

        if self.is_polar:
            int_array = cumtrapz(flow_array,y_coords, axis=axis,initial=0)
        else:
            middle_index = int(y_coords.size*0.5)
            lower_coords = y_coords[:middle_index][::-1]
            upper_coords = y_coords[middle_index:]

            lower_data = flow_array[:middle_index][::-1] if axis == 0 else flow_array[:,:middle_index][:,::-1]
            upper_data = flow_array[middle_index:] if axis == 0 else flow_array[:,middle_index:]

            int_lower_data = cumtrapz(lower_data,lower_coords,axis=axis,initial=0)
            int_upper_data = cumtrapz(upper_data,upper_coords,axis=axis,initial=0)

            int_array = np.zeros_like(flow_array)

            if axis == 0:
                int_array[:middle_index] = int_lower_data[::-1]
                int_array[middle_index:] = int_upper_data
            else:
                int_array[:,:middle_index] = int_lower_data[::-1]
                int_array[:,middle_index:] = int_upper_data

        return int_array

    def IntegrateWallNormal_total(self,CoordDF,flow_array):
        y_coords = CoordDF['y']
        if flow_array.ndim == 2:
            axis = 0
        elif flow_array.ndim == 3:
            axis = 1
        else:
            msg = r"This function can only be applied to arrays with dimension 2 and 3"
            raise ValueError(msg)

        return simps(flow_array,y_coords, axis=axis)

    def Integrate_cumult(self,CoordDF,flow_array):
        channel = not self.is_polar
        return gradient.cumIntegrate_y(CoordDF,flow_array,channel=channel)

    def Integrate_tot(self,CoordDF,flow_array):
        channel = not self.is_polar
        return gradient.totIntegrate_y(CoordDF,flow_array,channel=channel)

class coorddata(cd.AxisData):
    _domain_handler_class = DomainHandler
    
    def __init__(self,*args,from_file=False,from_coordstruct=False,**kwargs):
        if from_file or from_coordstruct:
            super().__init__(*args,from_file=from_file,**kwargs)
        else:
            self._coord_extract(*args,**kwargs)
            self._check_integrity()
         



    # def create_subdomain(self,*args,**kwargs):
    #     out_coorddata = self.copy()
    #     out_coorddata.coord_centered = out_coorddata.coord_centered.create_subdomain(*args,**kwargs)
    #     out_coorddata.coord_staggered = out_coorddata.coord_staggered.create_subdomain(*args,**kwargs)

    #     return out_coorddata

    # def __getattr__(self,attr):
    #     # print(attr,flush=True)
    #     # raise Exception
    #     msg = "The coorddata cannot use the attribute %s"%attr
    #     if attr == 'coord_centered' or attr == 'coord_staggered':
    #         raise AttributeError(msg)
    
    #     obj_c = getattr(self.coord_centered,attr)
    #     if self.contains_staggered:
    #         obj_s = getattr(self.coord_staggered,attr)

    #     def _apply_method(*args,**kwargs):
    #         out_c = obj_c(*args,**kwargs)
    #         if self.contains_staggered:
    #             out_s = obj_s(*args,**kwargs)

    #         if not all(isinstance(x,cd.coordstruct) for x in [out_c,out_s]):
    #             raise AttributeError(msg)

    #         out_coorddata = self.copy()

    #         out_coorddata.coord_centered = out_c
    #         out_coorddata.coord_staggered = None
    #         return out_coorddata

    #     if all(isinstance(x,types.MethodType) for x in [obj_c,obj_s]):
    #         return _apply_method
    #     else:
    #         raise AttributeError(msg)


    # def _copy_extract(self,other_coorddata):
    #     self.coord_centered = other_coorddata.centered.copy()
        
    #     if other_coorddata.contains_staggered:           
    #         self.coord_staggered = other_coorddata.staggered.copy()
    #     else:
    #         self.coord_staggered = None
            
    #     self._domain_handler = other_coorddata._domain_handler

    def _coord_extract(self,iCase,metaDF,path_to_folder,abs_path,tgpost,ioflg):
        if os.path.isdir(os.path.join(path_to_folder,'0_log_monitors')):
            coord_dict, coord_nd_dict =  self._coord_extract_new(metaDF,path_to_folder,abs_path,tgpost,ioflg)
        else:
            coord_dict, coord_nd_dict = self._coord_extract_old(path_to_folder,abs_path,tgpost,ioflg)

        self.coord_centered = cd.coordstruct(coord_dict)
        self.coord_staggered = cd.coordstruct(coord_nd_dict)

        geom_type = self._domain_handler_class.get_FlowGeometry(metaDF)
        self._domain_handler = self._domain_handler_class(geom_type)


    @classmethod
    def from_coordstructs(cls,Domain,coord,coord_nd):
        return cls(Domain,coord,coord_nd,from_coordstruct=True)
             
    
    # def to_hdf(self,filename,mode,key=None):
    #     if key is None:
    #         key = self.__class__.__name__

    #     self.coord_centered.to_hdf(filename,key=key+"/coord_centered",mode=mode)
        
    #     if self.contains_staggered:
    #         self.coord_staggered.to_hdf(filename,key=key+"/coord_staggered",mode=mode)

    #     hdf_obj = cd.hdfHandler(filename,mode='r',key=key)
    #     cart_mode = False if self._domain_handler.is_polar else True
    #     hdf_obj.attrs['cart_mode'] = cart_mode

        
    
    # def _legacy_mode(self,filename, key):

    #     key_split = key.split("/")
    #     parent_key = os.path.join(*key_split[:-1])
    #     end_key = key_split[-1]

    #     hdf_obj = cd.hdfHandler(filename,mode='r',key=parent_key)

    #     lower_keys = ["coord_centered","coord_staggered"]

    #     if not hdf_obj.check_key(end_key):
    #         return True
        
    #     lower_keys_present = all([lkey in hdf_obj[end_key].keys() for lkey in lower_keys])
    #     if not lower_keys_present:
    #         return True

    #     return False

    # def _create_staggered_legacy(self,coord):
    #     XCC = coord['x']
    #     YCC = coord['y']
    #     ZCC = coord['z']

    #     XND = np.zeros(XCC.size+1) 
    #     YND = np.zeros(YCC.size+1)
    #     ZND = np.zeros(ZCC.size+1)

    #     XND[0] = 0.0
    #     YND[0] = 0 if self._domain_handler.is_polar else -1.0
    #     ZND[0] = 0.0

    #     for i in  range(1,XND.size):
    #         XND[i] = XND[i-1] + 2*(XCC[i-1]-XND[i-1])
        
    #     for i in  range(1,YND.size):
    #         YND[i] = YND[i-1] + 2*(YCC[i-1]-YND[i-1])

    #     for i in  range(1,ZND.size):
    #         ZND[i] = ZND[i-1] + 2*(ZCC[i-1]-ZND[i-1])

    #     return cd.coordstruct({'x':XND,'y':YND,'z':ZND})

    # @property
    # def Mode(self):
    #     return self._mode

    # @Mode.setter
    # def Mode(self,new_mode):
    #     if new_mode not in self._modes_available:
    #         msg = "Mode selected must be in %s"% self._modes_available
    #         raise ValueError(msg)

    #     self._mode = new_mode

    @property
    def staggered(self):
        return self.coord_staggered

    @property
    def centered(self):
        return self.coord_centered
    
    @property
    def NCL(self):
        x_size = self.coord_centered['x'].size
        y_size = self.coord_centered['y'].size
        z_size = self.coord_centered['z'].size
        return (x_size,y_size,z_size)

    def create_vtkStructuredGrid(self,staggered = True):
        if staggered:
            if not self.contains_staggered:
                msg = "The staggered data cannot be None if this options is set"
                raise ValueError(msg)

            x_coords = self.coord_staggered['x']
            y_coords = self.coord_staggered['y']
            z_coords = self.coord_staggered['z']
        else:
            x_coords = self.coord_centered['x']
            y_coords = self.coord_centered['y']
            z_coords = self.coord_centered['z']

        Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)

        grid = StructuredGrid(X,Z,Y)
        return grid

    @property
    def contains_staggered(self):
        return not self.coord_staggered is None

    def _coord_extract_new(self,metaDF,path_to_folder,abs_path,tgpost,ioflg):
        full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')

        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(full_path,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(full_path,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(full_path,'CHK_COORD_ZND.dat')
    #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        x_coord=np.loadtxt(file,comments='#',usecols=1)
        
        if tgpost and not ioflg:
            XND = x_coord[:-1]
        else:
            index = int(metaDF['NCL1_tg']) + 1 
            if (tgpost and ioflg):
                XND = x_coord[:index]
                XND -= XND[0]
            elif metaDF['iCase'] == 5:
                index = int(metaDF['NCL1_io']) + 1
                XND = x_coord[:index]
            else:
                XND = x_coord[index:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,usecols=1,skiprows=1)
        index = int(metaDF['NCL2']) + 1 
        YCC=y_coord[index:]
        YND = y_coord[:index]
        y_size = YCC.size
        file.close()
        #============================================================
    
        file=open(z_coord_file,'rb')
        ZND=np.loadtxt(file,comments='#',usecols=1)

        #============================================================
        XCC, ZCC = self._coord_interp(XND,ZND)

        z_size = ZCC.size
        x_size = XCC.size
        y_size = YCC.size
        file.close()

        coord_dict = {'x':XCC,'y':YCC,'z':ZCC}
        coord_nd_dict = {'x':XND,'y':YND,'z':ZND}
        return coord_dict, coord_nd_dict


    def _coord_extract_old(self,path_to_folder,abs_path,tgpost,ioflg):
        """ Function to extract the coordinates from their .dat file """
        
        full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')

        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(full_path,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(full_path,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(full_path,'CHK_COORD_ZND.dat')
        #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        x_coord=np.loadtxt(file,comments='#')
        x_size=  int(x_coord[0])
        x_coord=np.delete(x_coord,0)
        
        if tgpost and not ioflg:
            XND = x_coord[:-1]
        else:
            for i in range(x_size):
                if x_coord[i] == 0.0:
                    index=i
                    break
            if tgpost and ioflg:
                XND = x_coord[:index+1]
                XND -= XND[0]
            else:
                XND = x_coord[index+1:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,comments='#',usecols=1)
        y_size = int(y_coord[0])
        for i in range(y_coord.size):
            if y_coord[i] == 1.0:
                index=i
                break
        
        # YCC=np.delete(y_coord,np.arange(index+1))
        YND = y_coord[:index+1]
        YCC = y_coord[(index+1):]
        y_size = YCC.size
        file.close()
        #============================================================
    
        file=open(z_coord_file,'rb')
        z_coord=np.loadtxt(file,comments='#')
        z_size = int(z_coord[0])
        ZND=np.delete(z_coord,0)
        #============================================================
        XCC, ZCC = self._coord_interp(XND,ZND)

        z_size = ZCC.size
        x_size = XCC.size
        y_size = YCC.size
        file.close()

        coord_dict = {'x':XCC,'y':YCC,'z':ZCC}
        coord_nd_dict = {'x':XND,'y':YND,'z':ZND}
        return coord_dict, coord_nd_dict
        
    def _coord_interp(self,XND, ZND):
        """ Interpolate the coordinates to give their cell centre values """
        XCC=np.zeros(XND.size-1)
        for i in range(XCC.size):
            XCC[i] = 0.5*(XND[i+1] + XND[i])
    
        ZCC=np.zeros(ZND.size-1)
        for i in range(ZCC.size):
            ZCC[i] = 0.5*(ZND[i+1] + ZND[i])
    
        return XCC, ZCC


    def copy(self):
        return copy.deepcopy(self)

    # def __deepcopy__(self,memo):
    #     return self.copy()

    def __eq__(self,other_obj):
        if not isinstance(other_obj,self.__class__):
            msg = "This operation can only be done on other objects of this type"
            raise TypeError(msg)

        if self._domain_handler.is_polar != other_obj._domain_handler.is_polar:
            return False

        if self.coord_centered != other_obj.coord_centered:
            return False

        if self.coord_staggered != other_obj.coord_staggered:
            return False

        return True

    def __ne__(self,other_obj):
        return not self.__eq__(other_obj)

    # @property
    # def Coord_ND_DF(self):
    #     XCC = self.CoordDF['x']
    #     YCC = self.CoordDF['y']
    #     ZCC = self.CoordDF['z']

    #     XND = np.zeros(XCC.size+1) 
    #     YND = np.zeros(YCC.size+1)
    #     ZND = np.zeros(ZCC.size+1)

    #     XND[0] = 0.0
    #     YND[0] = -1.0 if self.metaDF['iCase'] == 1 else 0
    #     ZND[0] = 0.0

    #     for i in  range(1,XND.size):
    #         XND[i] = XND[i-1] + 2*XCC[i-1]-XND[i-1]
        
    #     for i in  range(1,YND.size):
    #         YND[i] = YND[i-1] + 2*YCC[i-1]-YND[i-1]

    #     for i in  range(1,ZND.size):
    #         ZND[i] = ZND[i-1] + 2*ZCC[i-1]-ZND[i-1]

    #     return cd.datastruct({'x':XND,'y':YND,'z':ZND})
    