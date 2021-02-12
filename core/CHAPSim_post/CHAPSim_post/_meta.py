"""
# _meta.py
Class containing the meta data and coordinate arrays for the package
"""

import numpy as np
import h5py

import sys
import os
import warnings
import gc
import copy

import CHAPSim_post.CHAPSim_Tools as CT
import CHAPSim_post.CHAPSim_dtypes as cd

from CHAPSim_post.utils import misc_utils

class CHAPSim_meta():
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        copy = kwargs.pop('copy',False)
        if copy and fromfile:
            raise ValueError("You cannot create instance of CHAPSim_meta"+\
                                        " by copy and file simultaneously")
        if fromfile:
            kwargs.pop('tgpost',None)
            self._hdf_extract(*args,**kwargs)
        else:
            self.__extract_meta(*args,**kwargs)

    def __extract_meta(self,path_to_folder='.',abs_path=True,tgpost=False):
        self.metaDF = self._readdata_extract(path_to_folder,abs_path)
        ioflg = self.metaDF['iDomain'] in [2,3]
        self.CoordDF, self.NCL = self._coord_extract(path_to_folder,abs_path,tgpost,ioflg)
        
        self.path_to_folder = path_to_folder
        self._abs_path = abs_path

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = 'CHAPSim_meta'
        
        self.CoordDF = cd.datastruct.from_hdf(file_name,key=key+'/CoordDF')#pd.read_hdf(file_name,key=base_name+'/CoordDF').coord()
        self.metaDF = cd.metastruct.from_hdf(file_name,key=key+'/metaDF')#pd.read_hdf(file_name,key=base_name+'/metaDF')
        
        hdf_file = h5py.File(file_name,'r')
        self.NCL = hdf_file[key+'/NCL'][:]
        self.path_to_folder = hdf_file[key].attrs['path_to_folder'].decode('utf-8')
        self._abs_path = bool(hdf_file[key].attrs['abs_path'])
        hdf_file.close()
   
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = 'CHAPSim_meta'

        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(key)
        group.attrs["path_to_folder"] = self.path_to_folder.encode('utf-8')
        group.attrs["abs_path"] = int(self._abs_path)
        group.create_dataset("NCL",data=self.NCL)
        hdf_file.close()

        self.metaDF.to_hdf(file_name,key=key+'/metaDF',mode='a')
        self.CoordDF.to_hdf(file_name,key=key+'/CoordDF',mode='a')

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
            if tgpost and ioflg:
                XND = x_coord[:index]
                XND -= XND[0]
            else:
                XND = x_coord[index:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,usecols=1,skiprows=1)
        index = int(self.metaDF['NCL1_tg']) + 1 
        YCC=y_coord[index:]
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
        NCL = [x_size, y_size, z_size]
        return CoordDF, NCL


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
        YCC=np.delete(y_coord,np.arange(index+1))
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
        NCL = [x_size, y_size, z_size]
        return CoordDF, NCL
        
    def _coord_interp(self,XND, ZND):
        """ Interpolate the coordinates to give their cell centre values """
        XCC=np.zeros(XND.size-1)
        for i in range(XCC.size):
            XCC[i] = 0.5*(XND[i+1] + XND[i])
    
        ZCC=np.zeros(ZND.size-1)
        for i in range(ZCC.size):
            ZCC[i] = 0.5*(ZND[i+1] + ZND[i])
    
        return XCC, ZCC

    @property
    def Coord_ND_DF(self):
        XCC = self.CoordDF['x']
        YCC = self.CoordDF['y']
        ZCC = self.CoordDF['z']

        XND = np.zeros(XCC.size+1) 
        YND = np.zeros(YCC.size+1)
        ZND = np.zeros(ZCC.size+1)

        XND[0] = 0.0
        YND[0] = -1.0 if self.metaDF['iCase'] == 1 else 0
        ZND[0] = 0.0

        for i in  range(1,XND.size):
            XND[i] = XND[i-1] + 2*XCC[i-1]-XND[i-1]
        
        for i in  range(1,YND.size):
            YND[i] = YND[i-1] + 2*YCC[i-1]-YND[i-1]

        for i in  range(1,ZND.size):
            ZND[i] = ZND[i-1] + 2*ZCC[i-1]-ZND[i-1]

        return cd.datastruct({'x':XND,'y':YND,'z':ZND})
    