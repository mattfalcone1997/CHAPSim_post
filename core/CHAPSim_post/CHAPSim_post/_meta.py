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

from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd

class CHAPSim_meta():
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        copy = kwargs.pop('copy',False)
        if copy and fromfile:
            raise ValueError("You cannot create instance of CHAPSim_meta"+\
                                        " by copy and file simultaneously")
        if fromfile:
            self.CoordDF, self.NCL, self.Coord_ND_DF,\
            self.metaDF, self.path_to_folder,\
            self._abs_path = self._hdf_extract(*args,**kwargs)
        elif copy:
            self.CoordDF, self.NCL, self.Coord_ND_DF,\
            self.metaDF, self.path_to_folder,\
            self._abs_path = self._copy_extract(*args,**kwargs)
        else:
            self.CoordDF, self.NCL, self.Coord_ND_DF,\
            self.metaDF, self.path_to_folder,\
            self._abs_path = self.__extract_meta(*args,**kwargs)

    def __extract_meta(self,path_to_folder='',abs_path=True,tgpost=False):
        metaDF = self._readdata_extract(path_to_folder,abs_path)
        ioflg = metaDF['NCL1_tg_io'][1] > 2
        CoordDF, NCL = self._coord_extract(path_to_folder,abs_path,tgpost,ioflg)
        Coord_ND_DF = self.Coord_ND_extract(path_to_folder,NCL,abs_path,tgpost,ioflg)
        
        path_to_folder = path_to_folder
        abs_path = abs_path
        # moving_wall = self.__moving_wall_setup(NCL,path_to_folder,abs_path,metaDF,tgpost)
        return CoordDF, NCL, Coord_ND_DF, metaDF, path_to_folder, abs_path

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    @classmethod
    def copy(cls,meta_data):
        return cls(meta_data,copy=True)

    def _hdf_extract(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_meta'
        
        CoordDF = cd.datastruct.from_hdf(file_name,key=base_name+'/CoordDF')#pd.read_hdf(file_name,key=base_name+'/CoordDF').coord()
        Coord_ND_DF = cd.datastruct.from_hdf(file_name,key=base_name+'/Coord_ND_DF')#pd.read_hdf(file_name,key=base_name+'/Coord_ND_DF').coord()
        metaDF = cd.metastruct.from_hdf(file_name,key=base_name+'/metaDF')#pd.read_hdf(file_name,key=base_name+'/metaDF')
        
        hdf_file = h5py.File(file_name,'r')
        NCL = hdf_file[base_name+'/NCL'][:]
        path_to_folder = hdf_file[base_name].attrs['path_to_folder'].decode('utf-8')
        abs_path = bool(hdf_file[base_name].attrs['abs_path'])
        # moving_wall = hdf_file[base_name+'/moving_wall'][:]
        hdf_file.close()
        return CoordDF, NCL, Coord_ND_DF, metaDF, path_to_folder, abs_path
   
    def _copy_extract(self,meta_data):
        CoordDF = meta_data.CoordDF
        NCL = meta_data.NCL
        Coord_ND_DF = meta_data.Coord_ND_DF
        metaDF = meta_data.metaDF
        path_to_folder = meta_data.path_to_folder
        abs_path = meta_data._abs_path

        return CoordDF, NCL, Coord_ND_DF, metaDF, path_to_folder, abs_path

    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_meta'

        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.attrs["path_to_folder"] = self.path_to_folder.encode('utf-8')
        group.attrs["abs_path"] = int(self._abs_path)
        group.create_dataset("NCL",data=self.NCL)
        # group.create_dataset("moving_wall",data=self.__moving_wall)
        hdf_file.close()
        self.metaDF.to_hdf(file_name,key=base_name+'/metaDF',mode='a')#,format='fixed',data_columns=True)
        self.CoordDF.to_hdf(file_name,key=base_name+'/CoordDF',mode='a')#,format='fixed',data_columns=True)
        self.Coord_ND_DF.to_hdf(file_name,key=base_name+'/Coord_ND_DF',mode='a')#,format='fixed',data_columns=True)

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
        """ Function to extract the coordinates from their .dat file """
    
        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(path_to_folder,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(path_to_folder,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(path_to_folder,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(path_to_folder,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(path_to_folder,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(path_to_folder,'CHK_COORD_ZND.dat')
        #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        #print(x_coord_file)
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
                XND = x_coord[index+1:]#np.delete(x_coord,np.arange(index+1))
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
        CoordDF = cd.datastruct.from_dict({'x':XCC,'y':YCC,'z':ZCC})
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

    def return_edge_data(self):
        XCC = self.CoordDF['x'].dropna().values
        YCC = self.CoordDF['y'].dropna().values
        ZCC = self.CoordDF['z'].dropna().values

        XND = np.zeros(XCC.size+1) 
        YND = np.zeros(YCC.size+1)
        ZND = np.zeros(ZCC.size+1)

        XND[0] = 0.0
        YND[0] = -1.0
        ZND[0] = 0.0

        for i in  range(1,XND.size):
            XND[i] = 2*XCC[i-1]-XND[i-1]
        
        for i in  range(1,YND.size):
            YND[i] = 2*YCC[i-1]-YND[i-1]

        for i in  range(1,ZND.size):
            ZND[i] = 2*ZCC[i-1]-ZND[i-1]

        return XND, YND, ZND
        
    def Coord_ND_extract(self,path_to_folder,NCL,abs_path,tgpost,ioflg):
        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(path_to_folder,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(path_to_folder,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(path_to_folder,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(path_to_folder,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(path_to_folder,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(path_to_folder,'CHK_COORD_ZND.dat')
        #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        #print(x_coord_file)
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
                XND = x_coord[index+1:]#np.delete(x_coord,np.arange(index+1))
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        YND=np.loadtxt(file,comments='#',usecols=1)
        YND=YND[:NCL[1]+1]
        
        file.close()
        #y_size = YCC.size
        #============================================================
    
        file=open(z_coord_file,'rb')
        z_coord=np.loadtxt(file,comments='#')
        file.close()
        ZND=np.delete(z_coord,0)
        CoordDF = cd.datastruct.from_dict({'x':XND,'y':YND,'z':ZND})
        return CoordDF