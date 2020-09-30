'''
# CHAPSim_post - Postprocessing library for CHAPSim Solver
Postprocessing python scripts developed to be run on desktops and             
workstations. This has been tested on iceberg. Primarily developed for      
postprocessing the instantaneous and averaged fields of individual            
timesteps . Methods that require extensive use of the filesystem            
particularly creating averages using the instantaneous data from CHAPSim    
are advised to use the parallel module CHAPSim_parallel. Based principally  
on the Pandas DataFrame class                                               

## Structures and classes                                          #  
# = CHAPSim_Inst => Creates DataFrame for the instantaneous flow fields       #  
#                   velocity vector and pressure                              #  
#                => methods for producing contour and vector plots            #
# = CHAPSim_AVG => Creates DataFrames for all the average varables            #      
#                  produced by the CHAPSim Solver                             #  
#               => methods for producing contour plots for u_rms and u_mean   #
#         = CHAPSim_perturb => Child of CHAPSim_AVG, a module postprocessing  #
#                              in terms of perturbing velocity fields         #
# alongside an array of different line plot options                           #  
# = CHAPSim_autocov => Deprecated along with child classes - may be upgraded  #
#                      to include multi-threading                             #
# = CHAPSim_spectra(2) => Deprecated along with child classes                 #  
# = CHAPSim_meta=> Calculates meta data including parameters from the         #
#                  readdata file, the mesh size and a coordinate DataFrame    #
#                  => ***TO BE COMPLETED                                      #  
# = CHAPSim_budget => Calculates the budget terms of the components of the    #
#                     Reynolds stress equation                                #
#          = CHAPSim_<uu>_budget => Child of CHAPSim_budget calculating       #
#                                   specific components.                      #
#          = CHAPSim_k_budget => Child of CHAPSim_budget calculating the      #
#                                budget of the TKE equation                   #
#                                                                             #
# = CHAPSim_mom_balance => Class to calculate the budget of the momentum      #
#                          equation                                           #
#                       => Method for plotting the local imbalance to check   #
#                          solution veracity                                  #
# = CHAPSim_lambda2 => Class to calculate lambda2 to find vortex cores        #
#                   => **TO BE COMPLETED
#=============================================================================# 
#================ STATUS - In development and testing ========================#
# 
# -> testing on heft2 workstation required to enhance module integrity
# -> Complete peturbing velocity
# -> Commenting required
# -> Optimisations and removal of deprecated classes
#=============================================================================#
'''
# MODULE IMPORTS


from CHAPSim_parallel import CHAPSim_Quad_Anal, CHAPSim_autocov
import numpy as np
import pandas as pd; from pandas.errors import PerformanceWarning 
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate, fft
import os
import itertools
import time
import warnings
import sys

import CHAPSim_Tools as CT
import CHAPSim_plot as cplt
import CHAPSim_post_base as cbase
#import plotly.graph_objects as go

#import loop_accel as la
import numba
import h5py
#on iceberg HPC vtkmodules not found allows it to work as this resource isn't need on iceberg
try: 
    import pyvista as pv
except ImportError:
    warnings.warn("\033[1;33module `pyvista' has missing modules will not work correctly", stacklevel=2)

module = sys.modules[__name__]
module.TEST = False

class CHAPSim_Inst():
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        if not fromfile:
            self._meta_data, self.CoordDF, self.NCL,\
            self.InstDF,self.shape = self._inst_extract(*args,**kwargs)
        else:
            self._meta_data, self.CoordDF, self.NCL,\
            self.InstDF,self.shape = self._hdf_extract(*args,**kwargs)

    def _inst_extract(self,time,meta_data='',path_to_folder='',abs_path = True,tgpost=False):
        if not meta_data:
            meta_data = CHAPSim_meta(path_to_folder,abs_path,tgpost)
        CoordDF = meta_data.CoordDF
        NCL = meta_data.NCL

        #Give capacity for both float and lists
        if isinstance(time,float): 
            InstDF = self.__flow_extract(time,path_to_folder,abs_path,tgpost)
        elif hasattr(time,'__iter__'):
            for PhyTime in time:
                if not hasattr(self, 'InstDF'):
                    InstDF = self.__flow_extract(PhyTime,path_to_folder,abs_path,tgpost)
                else: #Variable already exists
                    local_DF = self.__flow_extract(PhyTime,path_to_folder,abs_path,tgpost)
                    concat_DF = [self.InstDF,local_DF]
                    InstDF = pd.concat(concat_DF)
        else:
            raise TypeError("\033[1;32 `time' must be either float or list")
        shape = (NCL[2],NCL[1],NCL[0])
        return meta_data,CoordDF,NCL,InstDF, shape

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)
    
    def _hdf_extract(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Inst'
        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        CoordDF=meta_data.CoordDF
        NCL=meta_data.NCL

        InstDF = pd.read_hdf(file_name,base_name+'/InstDF')
        shape = (NCL[2],NCL[1],NCL[0])

        return meta_data, CoordDF, NCL, InstDF, shape

    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Inst'
        self._meta_data.save_hdf(file_name,write_mode,group_name=base_name+'/meta_data')
        self.InstDF.to_hdf(file_name,key=base_name+'/InstDF',mode='a',format='fixed',data_columns=True)

    def __flow_extract(self,Time_input,path_to_folder,abs_path,tgpost):
        """ Extract velocity and pressure from the instantanous files """
        instant = "%0.9E" % Time_input
        file_folder = "1_instant_D"
        if tgpost:
            file_string = "DNS_perixz_INSTANT_T" + instant
        else:
            file_string = "DNS_perioz_INSTANT_T" + instant
        veloVector = ["_U","_V","_W","_P"]
        file_ext = ".D"
        
        file_list=[]
        for velo in veloVector:
            if not abs_path:
                file_list.append(os.path.abspath(os.path.join(path_to_folder, \
                             file_folder, file_string + velo + file_ext)))
            else:
                file_list.append(os.path.join(path_to_folder, \
                             file_folder, file_string + velo + file_ext))
        #A list of all the relevant files for this timestep                           
        open_list =[]
        #opening all the relevant files and placing them in a list
        for file in file_list:
            file_temp = open(file,'rb')
            open_list.append(file_temp)
        #allocating arrays
        int_info=np.zeros((4,4))
        r_info = np.zeros((4,3))

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

        flow_info=np.zeros((4,dummy_size))

        i=0
        #Extracting flow information
        for file in open_list:
            flow_info[i]=np.fromfile(file,dtype='float64',count=dummy_size)
            i+=1
        #Reshaping and interpolating flow data so that it is centred
        flow_info=flow_info.reshape((4,NCL3,NCL2,NCL1))
        # if tgpost:
        #     flow_info1=flow_info
        #     flow_info1 = flow_info1.reshape((4,dummy_size))
        # else:
        flow_info1 = self.__velo_interp(flow_info,NCL3,NCL2,NCL1)
        flow_info1 = flow_info1.reshape((4,dummy_size-NCL3*NCL2))
        flow_info=flow_info.reshape((4,dummy_size))
        Phy_string = '%.9g' % PhyTime
        # creating dataframe index
        index=pd.MultiIndex.from_arrays([[Phy_string,Phy_string,Phy_string,Phy_string],['u','v','w','P']])
        # creating dataframe so that data can be easily accessible elsewhere
        Instant_DF = pd.DataFrame(flow_info1,index=index)
        
        return Instant_DF
    def __velo_interp(self,flow_info,NCL3, NCL2, NCL1):
        """ Convert the velocity info so that it is defined at the cell centre """
        #This doesn't interpolate pressure as it is already located at the cell centre
        #interpolation reduces u extent, therefore to maintain size, V, W reduced by 1
        flow_interp = np.zeros((4,NCL3,NCL2,NCL1-1))
        for i in range(NCL1-1): #U velocity
            flow_interp[0,:,:,i] = 0.5*(flow_info[0,:,:,i] + flow_info[0,:,:,i+1])
        for i in range(NCL2): #V velocity
            if i != NCL2-1:
                flow_interp[1,:,i,:] = 0.5*(flow_info[1,:,i,:-1] + flow_info[1,:,i+1,:-1])
            else: #Interpolate with the top wall
                flow_interp[1,:,i,:] = 0.5*(flow_info[1,:,i,:-1] + flow_info[1,:,0,:-1])
        for i in range(NCL3): #W velocity
            if i != NCL3-1:
                flow_interp[2,i,:,:] = 0.5*(flow_info[2,i,:,:-1] + flow_info[2,i+1,:,:-1])
            else: #interpolation with first cell due to z periodicity BC
                flow_interp[2,i,:,:] = 0.5*(flow_info[2,i,:,:-1] + flow_info[2,0,:,:-1])
        flow_interp[3,:,:,:] = flow_info[3,:,:,:-1] #Removing final pressure value 
        return flow_interp
    def inst_contour(self,axis1,axis2,axis3_value,flow_field,PhyTime,fig='',ax='',**kwargs):
        """Function to output velocity contour plot on a particular plane"""
        #======================================================================
        # axis1 and axis2 represents the axes that will be shown in the plot
        # axis3_value is the cell value which will be shown
        # velo field represents the u,v,w or magnitude that will be ploted
        #======================================================================
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.9g}".format(PhyTime)
       
        if axis1 == axis2:
            raise ValueError("\033[1;32 Axes cannot be the same")
  
            
            
        axes = ['x','y','z']
        if axis1 not in axes or axis2 not in axes:
            raise ValueError("\033[1;32 axis values must be x,y or z")
        if axis1 != 'x' and axis1 != 'z':
            axis_temp = axis2
            axis2 = axis1
            axis1 = axis_temp
        elif axis1 == 'z' and axis2 == 'x':
            axis_temp = axis2
            axis2 = axis1
            axis1 = axis_temp
        
        axis1_coords = self.CoordDF[axis1].dropna().values
        axis2_coords = self.CoordDF[axis2].dropna().values
        if flow_field == 'u' or flow_field =='v' or flow_field =='w' or flow_field =='P':
            local_velo = self.InstDF.loc[PhyTime,flow_field].values
        elif flow_field == 'mag':
            index = pd.MultiIndex.from_arrays([[PhyTime,PhyTime,\
                                                PhyTime],['u','v','w']])
            local_velo = np.sqrt(np.square(self.InstDF.loc[index]).sum(axis=0)).values
        else:
            raise ValueError("\033[1;32 Not a valid argument")
        local_velo = local_velo.reshape(self.NCL[2],self.NCL[1],self.NCL[0])
        
        axis1_mesh, axis2_mesh = np.meshgrid(axis1_coords,axis2_coords)
        
        if axis1 =='x' and axis2 == 'y':
            velo_post = local_velo[axis3_value,:,:]
        elif axis1 =='x' and axis2 == 'z':
            velo_post = local_velo[:,axis3_value,:]
        elif axis1 =='z' and axis2 == 'y':
            velo_post = local_velo[:,:,axis3_value]
            velo_post = velo_post.T
        else:
            raise Exception
        
        if not fig:
            if 'figsize' not in kwargs.keys:
                kwargs['figsize']=figsize=[10,5]
            fig,ax = plt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
            
        ax1 = ax.pcolormesh(axis1_mesh,axis2_mesh,velo_post,cmap='jet')
        ax = ax1.axes
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
        cbar=fig.colorbar(ax1,ax=ax)
        if flow_field=='mag':
            cbar_label=r"$\vert U\vert$"
        else:
            cbar_label=r"$%s$"%flow_field.upper()
        cbar.set_label(cbar_label)# ,fontsize=12)
        ax.set_xlabel(r"$%s/\delta$" % axis1)# ,fontsize=18)
        ax.set_ylabel(r"$%s/\delta$" % axis2)# ,fontsize=16)
        #ax.axes().set_aspect('equal')
        #plt.colorbar(orientation='horizontal',shrink=0.5,pad=0.2)
        
        return fig, ax1
    def velo_vector(self,axis1,axis2,axis3_value,PhyTime,axis1_spacing,\
                    axis2_spacing, fig='', ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(1,1,1)
        fig, ax = _vector_plot(self.CoordDF,self.InstDF.loc[PhyTime],self.NCL,\
                               fig,ax,axis1_spacing,axis2_spacing,axis1,axis2,axis3_value)
        
        ax.set_title("Instantaneous velocity vector plot")
        return fig, ax
    
    def lambda2_calc(self,PhyTime,x_start_index=None,x_end_index=None,y_index=None):
        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
        
        arr_len_y = y_index if y_index is not None else self.NCL[1]
        
        if x_start_index is None:
            x_start_index=0 
        if x_end_index is None:
            x_end_index = self.NCL[0]
        arr_len_x = x_end_index - x_start_index

        strain_rate = np.zeros((self.NCL[2],arr_len_y,arr_len_x,3,3))
        rot_rate =  np.zeros((self.NCL[2],arr_len_y,arr_len_x,3,3))
        i=0
        for velo1,coord1 in zip(velo_list,coord_list):
            j=0
            for velo2,coord2 in zip(velo_list,coord_list):
                velo_field1 = self.InstDF.loc[PhyTime,velo1].values\
                    .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))[:,:arr_len_y,x_start_index:x_end_index]
                velo_field2 = self.InstDF.loc[PhyTime,velo2].values\
                    .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))[:,:arr_len_y,x_start_index:x_end_index]
                strain_rate[:,:,:,i,j] = 0.5*(CT.Grad_calc(self.CoordDF,velo_field1,coord2,False) \
                                        + CT.Grad_calc(self.CoordDF,velo_field2,coord1,False))
                rot_rate[:,:,:,i,j] = 0.5*(CT.Grad_calc(self.CoordDF,velo_field1,coord2,False) \
                                        - CT.Grad_calc(self.CoordDF,velo_field2,coord1,False))
                j+=1
            i+=1
        del velo_field1 ; del velo_field2
        S2_Omega2 = strain_rate**2 + rot_rate**2
        del strain_rate ; del rot_rate
        #eigs = np.vectorize(np.linalg.eig,otypes=[float],signature='(m,n,o,p,q)->(o,p,q)')
        S2_Omega2_eigvals, e_vecs = np.linalg.eigh(S2_Omega2)
        del e_vecs; del S2_Omega2
        #lambda2 = np.zeros_like(velo_field1)
        # eigs = np.zeros(3)
        
        lambda2 = np.sort(S2_Omega2_eigvals,axis=3)[:,:,:,1]
        
        # for i in range(inst_data.NCL[2]):
        #     for j in range(inst_data.NCL[1]):
        #         for k in range(inst_data.NCL[0]):
        #             eigs = np.sort(S2_Omega2_eigvals[i,j,k,:])
        #             lambda2[i,j,k] = eigs[1]
        return lambda2
    def plot_lambda2(self,vals_list,x_split_list='',PhyTime='',ylim='',Y_plus=True,avg_data='',colors='',fig=None,ax=None,**kwargs):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.InstDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.InstDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.InstDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        
        
        if not hasattr(vals_list,'__iter__'):
            vals_list = [vals_list]
        X = self._meta_data.CoordDF['x'].dropna().values
        Y = self._meta_data.CoordDF['y'].dropna().values
        Z = self._meta_data.CoordDF['z'].dropna().values
        if ylim:
            if Y_plus:
                y_index= CT.Y_plus_index_calc(avg_data,self.CoordDF,ylim)
            else:
                y_index=CT.coord_index_calc(self.CoordDF,'y',ylim)
            Y=Y[:y_index]
            # lambda2=lambda2[:,:y_index,:]
        if not x_split_list:
            x_split_list = [np.amin(X),np.amax(X)]
        
        if fig is None:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [9,5.5*(len(x_split_list)-1)]
            fig = cplt.mCHAPSimFigure(visible='off',**kwargs)
        else:
            if not isinstance(fig, cplt.matlabFigure):
                raise TypeError("fig must be of type %s not %s"\
                                %(cplt.matlabFigure,type(fig)))
        if ax is None:
            ax = fig.subplots(len(x_split_list)-1,squeeze=False)
        else:
            if not isinstance(ax, cplt.matlabAxes) and not isinstance(ax,np.ndarray):
                raise TypeError("fig must be of type %s not %s"\
                                %(cplt.matlabAxes,type(ax)))
        for j in range(len(x_split_list)-1):
            x_start = CT.coord_index_calc(self.CoordDF,'x',x_split_list[j])
            x_end = CT.coord_index_calc(self.CoordDF,'x',x_split_list[j+1])
            lambda2 = self.lambda2_calc(PhyTime,x_start,x_end)
            if ylim:
                lambda2=lambda2[:,:y_index,:]
            for val,i in zip(vals_list,range(len(vals_list))):
                
                color = colors[i%len(colors)] if colors else ''
                patch = ax[j].plot_isosurface(Y,Z,X[x_start:x_end],lambda2,val,color)
                # ax[j].add_lighting()
                # patch.set_color(colors[i%len(colors)])            
        # Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)
        
        # grid = pv.StructuredGrid(X,Z,Y)
        # grid.cell_arrays['lambda_2']=lambda2.flatten()
        # pgrid=grid.cell_data_to_point_data()
        # contour=pgrid.contour(isosurfaces=1,scalars='lambda_2',preference='point',rng=(lambda_min,lambda_max))
        # #grid.set_active_scalars('lambda_2')
        # if not plotter:
        #     plotter = pv.BackgroundPlotter(notebook=False)
        #     plotter.set_background('white')
        #     plotter.show_bounds(color='black')
            
        
        # #plotter.add_mesh(grid)
        # #filter_grid = grid.threshold((lambda_min,lambda_max),'lambda_2')
        
        
        # #plotter.add_axes(color='black')
        # plotter.add_mesh(contour,interpolate_before_map=True,cmap=cmap)
        # plotter.remove_scalar_bar()
        
        
        return fig, ax

    def vorticity_calc(self):
        pass
    def plot_entrophy(self):
        pass
    def __str__(self):
        return self.InstDF.__str__()
    def __iadd__(self,inst_data):
        assert self.CoordDF.equals(inst_data.CoordDF), "CHAPSim_Inst are not from the same case"
        assert self.NCL == inst_data.NCL, "CHAPSim_Inst are not from the same case"

        self.InstDF = pd.concat([self.InstDF,inst_data.InstDF])
        return self

class CHAPSim_Inst_io(CHAPSim_Inst):
    tgpost = False
    def _inst_extract(self,*args,**kwargs):
        kwargs['tgpost'] = self.tgpost
        return super()._inst_extract(*args,**kwargs)

class CHAPSim_Inst_tg(CHAPSim_Inst):
    tgpost = True
    def _inst_extract(self,*args,**kwargs):
        kwargs['tgpost'] = self.tgpost
        meta_data,CoordDF,NCL,InstDF, shape = super()._inst_extract(*args,**kwargs)
        
        NCL1_io = meta_data.metaDF.loc['HX_tg_io'].values[1]
        ioflowflg = True if NCL1_io > 2 else False
        
        if ioflowflg:
            NCL[0] -= 1
        shape = (NCL[2],NCL[1],NCL[0])

        return meta_data,CoordDF,NCL,InstDF, shape

    # def _hdf_extract(self,*args,**kwargs):
    #     meta_data, CoordDF, NCL, InstDF, shape = super()._hdf_extract(*args,**kwargs)

    #     NCL1_io = meta_data.metaDF.loc['HX_tg_io'].values[1]
    #     ioflowflg = True if NCL1_io > 2 else False
        
    #     if ioflowflg:
    #         NCL[0] -= 1

    #     shape = (NCL[2],NCL[1],NCL[0])
    #     return meta_data, CoordDF, NCL, InstDF, shape
        


class CHAPSim_AVG_io(cbase.CHAPSim_AVG_base):
    module = sys.modules[__name__]
    tgpost = False
    def _extract_avg(self,time,meta_data='',path_to_folder='',time0='',abs_path=True):
        
        if not meta_data:
            meta_data = CHAPSim_meta(path_to_folder,abs_path,False)
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL = meta_data.NCL
       
        if isinstance(time,float):
            DF_list = self._AVG_extract(time,time0,path_to_folder,abs_path)
        elif hasattr(time,'__iter__'):
            for PhyTime in time:
                if 'DF_list' not in locals():
                    DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                else:
                    DF_temp=[]
                    local_DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                    for DF, local_DF in zip(DF_list, local_DF_list):
                        concat_DF = [DF, local_DF]
                        
                        DF_temp.append(pd.concat(concat_DF))
                    DF_list=DF_temp
                    
        else:
            raise TypeError("\033[1;32 `time' can only be a float or a list")
        
        DF_list=self._Reverse_decomp(*DF_list)

        times = list(set([x[0] for x in DF_list[0].index]))
        shape = (NCL[1],NCL[0])

        return_list = [meta_data, CoordDF, metaDF, NCL,shape,times, *DF_list]
        return itertools.chain(return_list)
    
    def _hdf_extract(self,file_name,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_io'
        
        parent_list = list(super()._hdf_extract(file_name,group_name))

        meta_data = CHAPSim_meta.from_hdf(file_name,group_name+'/meta_data')
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL=meta_data.NCL
        
        hdf_file = h5py.File(file_name,'r')
        times = list(set([x[0] for x in parent_list[0].index]))
        shape = (NCL[1],NCL[0])
        hdf_file.close()

        return_list = [meta_data,CoordDF,metaDF,NCL,shape,times]

        return itertools.chain(return_list + parent_list)
        
    def _AVG_extract(self,Time_input,time0,path_to_folder,abs_path):
        if time0:
            instant = "%0.9E" % time0
            
            file_string = "DNS_perioz_AVERAGD_T" + instant + "_FLOW.D"
            file_folder = "2_averagd_D"
            if not abs_path:
                file_path = os.path.abspath(os.path.join(path_to_folder, \
                                         file_folder, file_string))
            else:
                file_path = os.path.join(path_to_folder, \
                                         file_folder, file_string)
                    
            file = open(file_path,'rb')
            
            int_info = np.zeros(4)
            r_info = np.zeros(3)
            int_info = np.fromfile(file,dtype='int32',count=4)    
            
            NCL1 = int_info[0]
            NCL2 = int_info[1]
            NSTATIS0 = int_info[3]
            dummy_size = NCL1*NCL2*50*21
            r_info = np.fromfile(file,dtype='float64',count=3)
            
            PhyTime = r_info[0]
            AVG_info0 = np.zeros(dummy_size)
            AVG_info0 = np.fromfile(file,dtype='float64',count=dummy_size)
        
        
        instant = "%0.9E" % Time_input
        
        file_string = "DNS_perioz_AVERAGD_T" + instant + "_FLOW.D"

        file_folder = "2_averagd_D"
        if not abs_path:
            file_path = os.path.abspath(os.path.join(path_to_folder, \
                                     file_folder, file_string))
        else:
            file_path = os.path.join(path_to_folder, \
                                     file_folder, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)
        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL1 = int_info[0]
        NCL2 = int_info[1]
        NSTATIS1 = int_info[3]
        
        dummy_size = NCL1*NCL2*50*21
        r_info = np.fromfile(file,dtype='float64',count=3)
        
        PhyTime = r_info[0]
        #REN = r_info[1]
        #DT = r_info[2]
        AVG_info = np.zeros(dummy_size)
        AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)
        if time0:
            AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)
        AVG_info = AVG_info.reshape(21,50,NCL2,NCL1)
            
        #Velo_AVG = np.zeros((3,NCL2,NCL1))
        Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        Pr_Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        DUDX2_tensor = np.zeros((81,NCL2,NCL1))
        flow_AVG = AVG_info[0,:4,:,:]
        
        PU_vector = AVG_info[2,:3,:,:]
        UU_tensor = AVG_info[3,:6,:,:]
        UUU_tensor = AVG_info[5,:10,:,:]
        
        for i in range(3):
            for j in range(3):
                Velo_grad_tensor[i*3+j,:,:] = AVG_info[6+j,i,:,:]
                Pr_Velo_grad_tensor[i*3+j,:,:] = AVG_info[9+j,i,:,:]
        for i in range(9):
            for j in range(9):
                DUDX2_tensor[i*9+j] = AVG_info[12+j,i,:,:] 
            
        #======================================================================
        flow_AVG = flow_AVG.reshape((4,NCL2*NCL1))
        
        PU_vector = PU_vector.reshape((3,NCL1*NCL2))
        UU_tensor = UU_tensor.reshape((6,NCL1*NCL2))
        UUU_tensor = UUU_tensor.reshape((10,NCL1*NCL2))
        Velo_grad_tensor = Velo_grad_tensor.reshape((9,NCL1*NCL2))
        Pr_Velo_grad_tensor = Pr_Velo_grad_tensor.reshape((9,NCL1*NCL2))
        DUDX2_tensor = DUDX2_tensor.reshape((81,NCL1*NCL2))
        #======================================================================
        #Set up of pandas dataframes
        Phy_string = '%.9g' % PhyTime
        flow_index = [[Phy_string,Phy_string,Phy_string,Phy_string],\
                      ['u','v','w','P']]
        vector_index = [[Phy_string,Phy_string,Phy_string],['u','v','w']]
        sym_2_tensor_index = [[Phy_string,Phy_string,Phy_string,Phy_string,\
                             Phy_string,Phy_string],['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[Phy_string,Phy_string,Phy_string,Phy_string,\
                               Phy_string,Phy_string,Phy_string,Phy_string,\
                               Phy_string,Phy_string],['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[Phy_string,Phy_string,Phy_string,Phy_string,\
                               Phy_string,Phy_string,Phy_string,Phy_string,\
                               Phy_string],['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']
        Phy_string_list=[]
        comp_string_list =[]
        for i in range(81):
            Phy_string_list.append(Phy_string)
            
        for du1 in du_list:
            for dx1 in dx_list:
                for du2 in du_list:
                    for dx2 in dx_list:
                        comp_string = du1 + dx1 + du2 + dx2
                        comp_string_list.append(comp_string)
         
        tensor_4_index = [Phy_string_list,comp_string_list]
        flow_AVGDF = pd.DataFrame(flow_AVG,pd.MultiIndex.from_arrays(flow_index))
        
        PU_vectorDF = pd.DataFrame(PU_vector,index=pd.MultiIndex.from_arrays(vector_index))
        UU_tensorDF = pd.DataFrame(UU_tensor,index=pd.MultiIndex.from_arrays(sym_2_tensor_index))
        UUU_tensorDF = pd.DataFrame(UUU_tensor,index=pd.MultiIndex.from_arrays(sym_3_tensor_index))
        Velo_grad_tensorDF = pd.DataFrame(Velo_grad_tensor,index=pd.MultiIndex.from_arrays(tensor_2_index))
        PR_Velo_grad_tensorDF = pd.DataFrame(Pr_Velo_grad_tensor,index=pd.MultiIndex.from_arrays(tensor_2_index))
        DUDX2_tensorDF = pd.DataFrame(DUDX2_tensor,index=pd.MultiIndex.from_arrays(tensor_4_index))

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]
    
    def save_hdf(self,file_name,write_mode,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_io'
        super().save_hdf(file_name,write_mode,group_name)

    def check_times(self,PhyTime):
        if isinstance(PhyTime,float) or isinstance(PhyTime,int):
            PhyTime = "%.9g"%PhyTime
        elif not isinstance(PhyTime,str):
            raise TypeError("PhyTime is the wrong type")

        if len(self.get_times()) == 1:
            avg_time = self.get_times()[0]
            if PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            if not PhyTime in self.get_times():
                raise ValueError("The time given is not in this present in class")
        return PhyTime

    def _return_index(self,x_val):
        return CT.coord_index_calc(self.CoordDF,'x',x_val)

    def _return_xaxis(self):
        return self.CoordDF['x'].dropna().values

    def int_thickness_calc(self, PhyTime=''):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        return super()._int_thickness_calc(PhyTime)

    def wall_unit_calc(self,PhyTime=''):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        return self._wall_unit_calc(PhyTime)

    def plot_shape_factor(self, PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = self._plot_shape_factor(PhyTime,fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x'].dropna().values
        line = ax.get_lines()[-1]
        line.set_xdata(x_coords)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_mom_thickness(self, PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        fig, ax = super().plot_mom_thickness(PhyTime,fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x'].dropna().values
        line = ax.get_lines()[-1]
        line.set_xdata(x_coords)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_disp_thickness(self, PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_disp_thickness(PhyTime,fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x'].dropna().values
        line = ax.get_lines()[-1]
        line.set_xdata(x_coords)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds(self,comp1,comp2,x_vals,PhyTime='',norm=None,Y_plus=True,fig='',ax='',**kwargs):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        

        fig, ax = super().plot_Reynolds(comp1,comp2,x_vals,PhyTime,
                                        norm=norm,Y_plus=Y_plus,
                                        fig=fig,ax=ax,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$"%float(x))

        axes_items_num = len(lines)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.relim()
        ax.autoscale_view()

        return fig, ax        

    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_Reynolds_x(comp1,comp2,y_vals_list,Y_plus=True,
                                            PhyTime=PhyTime,fig=fig,ax=ax,**kwargs)
        
        line_no = 1 if y_vals_list == 'max' else len(y_vals_list)
        lines = ax.get_lines()[-line_no:]
        x_coord = self.CoordDF['x'].dropna().values
        for line in lines:
            line.set_xdata(x_coord)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
    def bulk_velo_calc(self,PhyTime=''):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        return self._bulk_velo_calc(PhyTime)

    def plot_bulk_velocity(self,PhyTime='',fig='',ax='',**kwargs):
        
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_bulk_velocity(PhyTime,fig,ax,**kwargs)
        line = ax.get_lines()[-1]
        x_coord = self.CoordDF['x'].dropna().values
        line.set_xdata(np.array(x_coord))
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def tau_calc(self,PhyTime=''):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        return self._tau_calc(PhyTime)

    def plot_skin_friction(self,PhyTime='',fig='',ax='',**kwargs):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        

        fig, ax = super().plot_skin_friction(PhyTime,fig,ax,**kwargs)
        line = ax.get_lines()[-1]
        x_coord = self.CoordDF['x'].dropna().values
        line.set_xdata(np.array(x_coord))
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,x_vals,PhyTime='',Y_plus=True,Y_plus_max=100,fig='',ax='',**kwargs):
        
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_eddy_visc(x_vals,PhyTime,Y_plus,Y_plus_max,fig,ax,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$" % float(x))

        axes_items_num = len(x_vals)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def avg_line_plot(self,x_vals,comp,PhyTime='',fig='',ax='',*args,**kwargs):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super().avg_line_plot(x_vals,PhyTime,comp,fig='',ax='',**kwargs)
        line_no=len(x_vals)
        lines = ax.get_lines()[-line_no:]
        for line, x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$"% x)

        axes_items_num = len(x_vals)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
    def plot_near_wall(self,x_vals,PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super().plot_near_wall(x_vals,PhyTime,fig=fig,ax=ax,**kwargs)
        line_no=len(x_vals)
        lines = ax.get_lines()[-line_no-1:-1]
        for line, x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$"% x)

        axes_items_num = len(x_vals)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
class CHAPSim_AVG_tg_base(cbase.CHAPSim_AVG_base):
    module = sys.modules[__name__]
    tgpost = True
    def _extract_avg(self,PhyTimes,*args,meta_data='',path_to_folder='',time0='',abs_path=True,permissive=False):

        if isinstance(path_to_folder,list):
            folder_path=path_to_folder[0]
            if not permissive:
                for path in path_to_folder:
                    PhyTimes = list(set(PhyTimes).intersection(time_extract(path,abs_path)))
        else:
            folder_path=path_to_folder
            PhyTimes = list(set(PhyTimes).intersection(time_extract(path_to_folder,abs_path)))
        PhyTimes=sorted(PhyTimes)

        if not meta_data:
            meta_data = CHAPSim_meta(folder_path,abs_path,tgpost=True)
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL = meta_data.NCL

        if isinstance(PhyTimes,float):
            times = ['%.9g' % PhyTimes]
            DF_list = self._AVG_extract(PhyTimes,folder_path,abs_path,metaDF,time0)
        elif hasattr(PhyTimes,'__iter__'):
            times = ['%.9g' % time for time in PhyTimes]
            for PhyTime in PhyTimes:
                if 'DF_list' not in locals():
                    DF_list = self._AVG_extract(PhyTime,folder_path,abs_path,metaDF,time0)
                else:
                    DF_temp=[]
                    local_DF_list = self._AVG_extract(PhyTime,folder_path,abs_path,metaDF,time0)
                    for DF, local_DF in zip(DF_list, local_DF_list):
                        concat_DF = [DF, local_DF]
                        DF_temp.append(pd.concat(concat_DF,axis=1))
                    DF_list=DF_temp
            

        else:
            raise TypeError("\033[1;32 `PhyTimes' can only be a float or a list") 
        
        old_shape = (len(times),NCL[1])
        flat_shape = (NCL[1]*len(times))
        for DF in DF_list:
            DF.columns = range(flat_shape)
            for index, row in DF.iterrows():
                DF.loc[index] = pd.Series(row.values.reshape(old_shape).T.reshape(flat_shape))


        DF_list=self._Reverse_decomp(*DF_list)

        return_list = [meta_data, CoordDF, metaDF, NCL,old_shape[::-1],times,*DF_list]
        if isinstance(path_to_folder,list):
            i=2
            for path in path_to_folder[1:]:                
                AVG_list = list(CHAPSim_AVG_tg_base._extract_avg(self,PhyTimes,meta_data=meta_data,
                                path_to_folder=path,time0=time0,abs_path=abs_path,permissive=False))
                return_list = self._ensemble_average(return_list,AVG_list,i,permissive)
                i+=1

        # self._Reverse_decomp(*DF_list)
        return itertools.chain(return_list)

    def _ensemble_average(self,return_list, AVG_list,number,permissive=False):
        if not permissive:
            coe2 = (number-1)/number ; coe3 = 1/number
            assert return_list[1].equals(AVG_list[1]), "Coordinates are not the same"
            assert return_list[3] == AVG_list[3], "Mesh is not the same"
            assert return_list[5] == AVG_list[5], "Times must be same for non permissive ensemble averaging"
            for i in range(6,13):
                index = return_list[i].index
                array = coe2*return_list[i].values + coe3*AVG_list[i].values
                return_list[i] = pd.DataFrame(array,index = index)
        else:
            raise NotImplementedError

        return return_list

    def Perform_ensemble(self):
        raise NotImplementedError
    def _extract_file(self,PhyTime,path_to_folder,abs_path):
        instant = "%0.9E" % PhyTime
        
        file_string = "DNS_perixz_AVERAGD_T" + instant + "_FLOW.D"
        
        file_folder = "2_averagd_D"
        if not abs_path:
            file_path = os.path.abspath(os.path.join(path_to_folder, \
                                        file_folder, file_string))
        else:
            file_path = os.path.join(path_to_folder, \
                                        file_folder, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)
        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL2 = int_info[0]
        NSZ = int_info[1]
        ITERG = int_info[2]
        NSTATIS = int_info[3]
        dummy_size = NCL2*NSZ
        r_info = np.fromfile(file,dtype='float64',count=3)
        
        PhyTime = r_info[0]
        AVG_info = np.zeros(dummy_size)
        AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)

        AVG_info = AVG_info.reshape(NSZ,NCL2)

        return AVG_info, NSTATIS
    def _AVG_extract(self,PhyTime,path_to_folder,abs_path,metaDF,time0):

        factor = metaDF.loc['NCL1_tg_io'].values[0]*metaDF.loc['NCL3'].values[0]
        AVG_info, NSTATIS1 = self._extract_file(PhyTime,path_to_folder,abs_path)
        ioflowflg = True if metaDF.loc['NCL1_tg_io'].values[1]>2 else False
        if ioflowflg and time0:
            AVG_info0, NSTATIS0 = self._extract_file(time0,path_to_folder,abs_path)
            AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)
        # print(AVG_info[51])
        flow_AVG = AVG_info[:4]
        PU_vector = AVG_info[4:7]
        UU_tensor = AVG_info[7:13]
        UUU_tensor = AVG_info[13:23]
        Velo_grad_tensor = AVG_info[23:32]
        Pr_Velo_grad_tensor = AVG_info[32:41]
        DUDX2_tensor = AVG_info[41:]*factor

        Phy_string = '%.9g' % PhyTime
        flow_index = [[None]*4,['u','v','w','P']]
        vector_index = [[None]*3,['u','v','w']]
        sym_2_tensor_index = [[None]*6,['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[None]*10,['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[None]*9,['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']
        Phy_string_list = [None]*81
        comp_string_index =[]            
        for du1 in du_list:
            for dx1 in dx_list:
                for du2 in du_list:
                    for dx2 in dx_list:
                        comp_string = du1 + dx1 + du2 + dx2
                        comp_string_index.append(comp_string)
        tensor_4_index=[Phy_string_list,comp_string_index]

        flow_AVGDF = pd.DataFrame(flow_AVG,index=flow_index)
        PU_vectorDF = pd.DataFrame(PU_vector,index=vector_index)
        UU_tensorDF = pd.DataFrame(UU_tensor,index=sym_2_tensor_index)
        UUU_tensorDF = pd.DataFrame(UUU_tensor,index=sym_3_tensor_index)
        Velo_grad_tensorDF = pd.DataFrame(Velo_grad_tensor,index=tensor_2_index)
        PR_Velo_grad_tensorDF = pd.DataFrame(Pr_Velo_grad_tensor,index=tensor_2_index)
        DUDX2_tensorDF = pd.DataFrame(DUDX2_tensor,index=tensor_4_index)

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]
    
    def _hdf_extract(self,file_name,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_tg'
        base_name = group_name

        parent_list = list(super()._hdf_extract(file_name,group_name))

        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL=meta_data.NCL
        
        hdf_file = h5py.File(file_name,'r')
        shape = tuple(hdf_file[base_name].attrs["shape"][:])
        times = list(np.char.decode(hdf_file[base_name].attrs["times"][:]))
        hdf_file.close()

        return_list = [meta_data,CoordDF,metaDF,NCL,shape,times]

        return itertools.chain(return_list + parent_list)

    def save_hdf(self,file_name,write_mode,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_tg'
        super().save_hdf(file_name,write_mode,group_name)
        base_name=group_name if group_name else 'CHAPSim_AVG'
        hdf_file = h5py.File(file_name,'a')
        group = hdf_file[base_name]
        group.attrs['shape'] = np.array(self.shape)
        group.attrs['times'] = np.array([np.string_(x) for x in self.get_times()])
        hdf_file.close()


    def _return_index(self,PhyTime):
        if not isinstance(PhyTime,str):
            PhyTime = "{:.9g}".format(PhyTime)

        if PhyTime not in self.get_times():
            raise ValueError("time %s must be in times"% PhyTime)
        for i in range(len(self.get_times())):
            if PhyTime==self.get_times()[i]:
                return i

    def _return_xaxis(self):
        return np.array([float(time) for time in self.get_times()])

    def wall_unit_calc(self):
        return self._wall_unit_calc(float("NaN"))

    def int_thickness_calc(self):
        PhyTime = float("NaN")
        return super()._int_thickness_calc(PhyTime)

    def plot_shape_factor(self,fig='',ax='',**kwargs):
        PhyTime = float("NaN")
        fig, ax = self._plot_shape_factor(fig=fig,ax=ax,**kwargs)
        times = np.array([float(x) for x in self.get_times()])
        line=ax.get_lines()[-1]
        line.set_xdata(times)
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds(self,comp1,comp2,PhyTime,norm=None,Y_plus=True,fig='',ax='',**kwargs):

        fig, ax = super().plot_Reynolds(comp1,comp2,PhyTime,float("NaN"),
                                        norm=norm,Y_plus=Y_plus,
                                        fig=fig,ax=ax,**kwargs)
        lines = ax.get_lines()[-len(PhyTime):]
        for line,time in zip(lines,PhyTime):
            line.set_label(r"$t^*=%.3g$"%float(time))
        axes_items_num = len(lines)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,fig='',ax='',**kwargs):
        fig, ax = super().plot_Reynolds_x(comp1,comp2,y_vals_list,Y_plus=True,
                                            PhyTime=float("NaN"),fig=fig,ax=ax,**kwargs)
        
        line_no = 1 if y_vals_list == 'max' else len(y_vals_list)
        lines = ax.get_lines()[-line_no:]
        times = np.array([float(x) for x in self.get_times()])
        for line in lines:
            line.set_xdata(times)
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax    

    def bulk_velo_calc(self):
        return super()._bulk_velo_calc(float('NaN'))

    def plot_bulk_velocity(self,fig='',ax='',**kwargs):
        fig, ax = super().plot_bulk_velocity(float('NaN'),fig,ax,**kwargs)
        line = ax.get_lines()[-1]
        times = [float(x) for x in self.get_times()]
        line.set_xdata(np.array(times))
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def tau_calc(self):
        return self._tau_calc(float("NaN"))

    def plot_skin_friction(self,fig='',ax='',**kwargs):
        fig, ax = super().plot_skin_friction(float("NaN"),fig=fig,ax=ax,**kwargs)
        line = ax.get_lines()[-1]
        times = [float(x) for x in self.get_times()]
        line.set_xdata(np.array(times))
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,times,Y_plus=True,Y_plus_max=100,fig='',ax='',**kwargs):
        fig, ax = super().plot_eddy_visc(times,float("NaN"),Y_plus,Y_plus_max,fig,ax,**kwargs)
        lines = ax.get_lines()[-len(times):]
        try:
            for line, time in zip(lines,times):
                line.set_label(r"$t^*=%.3g$" % float(time))
        except TypeError:
            lines.set_label(r"$t^*=%.3g$" % float(times))

        axes_items_num = len(times)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)

        return fig, ax

    def avg_line_plot(self,times,*args,fig='',ax='',**kwargs):
        
        fig, ax = super().avg_line_plot(times,float("NaN"),*args,fig=fig,ax=ax,**kwargs)

        lines = ax.get_lines()[-len(times):]
        for line, time in zip(lines,times):
            line.set_label(r"$t^*=%.3f$"% time)
            
        axes_items_num = len(times)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        
        return fig, ax

    def plot_near_wall(self,times,fig='',ax='',**kwargs):
        fig, ax = super().plot_near_wall(times,float('nan'),fig=fig,ax=ax,**kwargs)
        line_no=len(times)
        lines = ax.get_lines()[-line_no:]
        for line, time in zip(lines,times):
            line.set_label(r"$t^*=%.3g$"% time)

        axes_items_num = len(times)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
        
class CHAPSim_AVG_tg(CHAPSim_AVG_tg_base):
    def _extract_avg(self,path_to_folder='',time0='',abs_path=True,*args,**kwargs):
        if isinstance(path_to_folder,list):
            times = time_extract(path_to_folder[0],abs_path)
        else:
            times = time_extract(path_to_folder,abs_path)
        if time0:
            times = list(filter(lambda x: x > time0, times))
        return super()._extract_avg(times,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path,*args,**kwargs)

class CHAPSim_AVG():
    def __new__(cls,*args,tgpost=False,**kwargs):
        if tgpost:
            return CHAPSim_AVG_tg(*args,**kwargs)
        else:
            return CHAPSim_AVG_io(*args,**kwargs)

    @classmethod
    def from_hdf(cls,*args,tgpost=False,**kwargs):
        if tgpost:
            return cls(tgpost=tgpost,fromfile=True,*args,**kwargs)
        else:
            return cls(fromfile=True,*args,**kwargs)

class CHAPSim_fluct_io(cbase.CHAPSim_fluct_base):
    tgpost = False
    def __init__(self,time_inst_data_list,avg_data='',path_to_folder='',abs_path=True,*args,**kwargs):
        if not avg_data:
            time = CT.max_time_calc(path_to_folder,abs_path)
            avg_data = CHAPSim_AVG_io(time,path_to_folder=path_to_folder,abs_path=abs_path,*args,**kwargs)
        if not hasattr(time_inst_data_list,'__iter__'):
            time_inst_data_list = [time_inst_data_list]
        for time_inst_data in time_inst_data_list:
            if isinstance(time_inst_data,CHAPSim_Inst):
                if 'inst_data' not in locals():
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if 'inst_data' not in locals():
                    inst_data = CHAPSim_Inst(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=False)
                else:
                    inst_data += CHAPSim_Inst(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=False)

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)

        self.avg_data = avg_data
        self.meta_data = avg_data._meta_data
        self.NCL = self.meta_data.NCL
        self.CoordDF = self.meta_data.CoordDF
        self.shape = inst_data.shape

    def _fluct_calc(self,inst_data,avg_data):
        inst_shape = inst_data.shape
        avg_shape = avg_data.shape
        fluct_array = np.zeros((4,*inst_shape))
        j=0
        for (index, inst), (index_avg, avg) in zip(inst_data.InstDF.iterrows(),avg_data.flow_AVGDF.iterrows()):
            inst = inst.values.reshape(inst_data.shape)
            avg = avg.values.reshpe(avg_data.shape)
            for i in range(inst_shape[0]):
                fluct_array[j,i] = inst[i] - avg
            j+=1
        return fluct_array.reshape((4,np.prod(inst_shape)))
        
    def _fluctDF_calc(self, inst_data, avg_data):
        inst_times = list(set([x[0] for x in inst_data.InstDF.index]))
        u_comp = [x[1] for x in inst_data.InstDF.index]
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape))
        avg_time = list(set([x[0] for x in avg_data.flow_AVGDF.index]))
        
        assert len(avg_time) == 1, "In this context therecan only be one time in avg_data"
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape))
        j=0
        avg_index = avg_data.flow_AVGDF.index 
        
        for time in inst_times:
            for comp,index in zip(u_comp,avg_index):
                avg_values = avg_data.flow_AVGDF.loc[index].values.reshape(avg_data.shape)
                inst_values = inst_data.InstDF.loc[time,comp].values.reshape(inst_data.shape)

                for i in range(inst_data.shape[0]):
                    fluct[j,i] = inst_values[i] -avg_values
                j+=1
        fluct = fluct.reshape((len(inst_data.InstDF.index),np.prod(inst_data.shape)))
        return pd.DataFrame(fluct,index=inst_data.InstDF.index)

class CHAPSim_fluct_tg(cbase.CHAPSim_fluct_base):
    tgpost = True
    def __init__(self,time_inst_data_list,avg_data='',path_to_folder='',abs_path=True,*args,**kwargs):
        if not hasattr(time_inst_data_list,'__iter__'):
            time_inst_data_list = [time_inst_data_list]
        for time_inst_data in time_inst_data_list:
            if isinstance(time_inst_data,CHAPSim_Inst):
                if 'inst_data' not in locals():
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if 'inst_data' not in locals():
                    inst_data = CHAPSim_Inst(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=True)
                else:
                    inst_data += CHAPSim_Inst(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=True)
        
        if not avg_data:
            times = [float(x[0]) for x in inst_data.InstDF.index]
            avg_data = CHAPSim_AVG_tg_base(times,path_to_folder=path_to_folder,abs_path=abs_path)

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)
        self.avg_data = avg_data
        self.meta_data = avg_data._meta_data
        self.NCL = self.meta_data.NCL
        self.CoordDF = self.meta_data.CoordDF
        self.shape = inst_data.shape
    
    def _fluctDF_calc(self, inst_data, avg_data):
        avg_times = avg_data.get_times()
        inst_times = list(set([x[0] for x in inst_data.InstDF.index]))
        u_comp = [x[1] for x in avg_data.flow_AVGDF.index]
        indices = inst_data.InstDF.index
        fluct = np.zeros((len(indices),*inst_data.shape))
        if len(avg_times) == 1:
            j=0
            for time in inst_times:
                avg_index=avg_data._return_index(time)
                for comp in u_comp:
                    avg_values = avg_data.flow_AVGDF.loc[float('nan'),comp].values.reshape(avg_data.shape)
                    inst_values = inst_data.InstDF.loc[time,comp].values.reshape(inst_data.shape)

                    for i in range(inst_data.shape[0]):
                        for k in range(inst_data.shape[2]):
                            fluct[j,i,:,k] = inst_values[i,:,k] -avg_values[:,avg_index]
                    j+=1
        elif all(time in avg_times for time in inst_times):
            for index, j in zip(indices,range(len(indices))):
                avg_index=avg_data._return_index(index[0])
                avg_values = avg_data.flow_AVGDF.loc[float('nan'),index[1]].values.reshape(avg_data.shape)
                inst_values = inst_data.InstDF.loc[index].values.reshape(inst_data.shape)
                for i in range(inst_data.shape[0]):
                    for k in range(inst_data.shape[2]):
                        fluct[j,i,:,k] = inst_values[i,:,k] -avg_values[:,avg_index]
        else:
            raise ValueError("avg_data must either be length 1 or the same length as inst_data")
        
        return pd.DataFrame(fluct.reshape(len(indices),np.prod(inst_data.shape)),index=inst_data.InstDF.index)

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
        ioflg = metaDF.loc['NCL1_tg_io'].values[1] > 2
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

        CoordDF = pd.read_hdf(file_name,key=base_name+'/CoordDF')
        Coord_ND_DF = pd.read_hdf(file_name,key=base_name+'/Coord_ND_DF')
        metaDF = pd.read_hdf(file_name,key=base_name+'/metaDF')
        
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
        self.metaDF.to_hdf(file_name,key=base_name+'/metaDF',mode='a',format='fixed',data_columns=True)
        self.CoordDF.to_hdf(file_name,key=base_name+'/CoordDF',mode='a',format='fixed',data_columns=True)
        self.Coord_ND_DF.to_hdf(file_name,key=base_name+'/Coord_ND_DF',mode='a',format='fixed',data_columns=True)

    def _readdata_extract(self,path_to_folder,abs_path):
        
        if not abs_path:
            readdata_file = os.path.abspath(os.path.join(path_to_folder,'readdata.ini'))
        else:
           readdata_file = os.path.join(path_to_folder,'readdata.ini')
        with open(readdata_file) as file_object:
            lines = (line.rstrip() for line in file_object)
            lines = list(line for line in lines if line)
            meta_list=[]
            meta_out = np.zeros((len(lines),2))
            i=0
        for line in lines:
            meta_list.append(line.split(None,1)[0][:-1])
            try:
                
                if line.split()[2] == ';':
                    meta_out[i][0] = line.split()[1]
                    meta_out[i][1] = float('NaN')
                else:
                    meta_out[i][0] = line.split()[1][:-1]
                    meta_out[i][1] = line.split()[2]
            except IndexError:
                meta_out[i] = float('NaN')
            except ValueError:
                meta_out[i] = float('NaN')
            i += 1
        meta_DF = pd.DataFrame(meta_out,index=meta_list)
        meta_DF = meta_DF.dropna(how='all')
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
        X_series = pd.Series(XCC)
        Y_series = pd.Series(YCC)
        Z_series = pd.Series(ZCC)
        CoordDF = pd.DataFrame({'x':X_series,'y':Y_series,'z':Z_series})
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

        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        YND=np.loadtxt(file,comments='#',usecols=1)
        YND=YND[:NCL[1]+1]
        
        #y_size = YCC.size
        #============================================================
    
        file=open(z_coord_file,'rb')
        z_coord=np.loadtxt(file,comments='#')
        
        ZND=np.delete(z_coord,0)
        X_series = pd.Series(XND)
        Y_series = pd.Series(YND)
        Z_series = pd.Series(ZND)
        CoordDF = pd.DataFrame({'x':X_series,'y':Y_series,'z':Z_series})
        return CoordDF

class CHAPSim_budget_io(cbase.CHAPSim_budget_base):
    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape(self.avg_data.shape)
        U_mean = self.avg_data.flow_AVGDF.loc[PhyTime,'u']\
                        .values.reshape(self.avg_data.shape)
        V_mean = self.avg_data.flow_AVGDF.loc[PhyTime,'v']\
                        .values.reshape(self.avg_data.shape)
        uu_dx = Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -(U_mean*uu_dx + V_mean*uu_dy)
        return advection.flatten()

    def _turb_transport(self,PhyTime,comp1,comp2):
        uu_comp = comp1+comp2
        uu_comp1 = uu_comp+'u'
        uu_comp2 = uu_comp+'v'

        
        if ord(uu_comp1[0]) > ord(uu_comp1[1]):
            uu_comp1 = uu_comp1[:2][::-1] + uu_comp1[2]
        if ord(uu_comp1[0]) > ord(uu_comp1[2]):
            uu_comp1 = uu_comp1[::-1]
        if ord(uu_comp1[1]) > ord(uu_comp1[2]):
            uu_comp1 = uu_comp1[0] + uu_comp1[1:][::-1]
            
        if ord(uu_comp2[0]) > ord(uu_comp2[1]):
            uu_comp2 = uu_comp2[:2][::-1] + uu_comp2[2]
        if ord(uu_comp2[0]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[::-1]
        if ord(uu_comp2[1]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[0] + uu_comp2[1:][::-1]

        u1u2u = self.avg_data.UUU_tensorDF.loc[PhyTime,uu_comp1]\
                        .values.reshape(self.avg_data.shape)
        u1u2v = self.avg_data.UUU_tensorDF.loc[PhyTime,uu_comp2]\
                        .values.reshape(self.avg_data.shape)

        u1u2u_dx = Grad_calc(self.avg_data.CoordDF,u1u2u,'x')
        u1u2v_dy = Grad_calc(self.avg_data.CoordDF,u1u2v,'y')

        turb_transport = -(u1u2u_dx + u1u2v_dy)
        return turb_transport.flatten()

    def _pressure_strain(self,PhyTime,comp1,comp2):
        u1u2 = comp1 + chr(ord(comp2)-ord('u')+ord('x'))
        u2u1 = comp2 + chr(ord(comp1)-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.avg_data.PR_Velo_grad_tensorDF.loc[PhyTime,u1u2]\
                        .values.reshape(self.avg_data.shape)
        pdu2dx1 = self.avg_data.PR_Velo_grad_tensorDF.loc[PhyTime,u2u1]\
                        .values.reshape(self.avg_data.shape)
        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain.flatten()

    def _pressure_diffusion(self,PhyTime,comp1,comp2):
        if comp1 == 'u' and comp2 =='u':
            diff1 = diff2 = 'x'
        elif comp1 == 'v' and comp2 =='v':
            diff1 = diff2 = 'y'
        elif comp1 =='u' and comp2 =='v':
            diff1 = 'y'
            diff2 = 'x'
        elif comp1 == 'w' and comp2 == 'w':
            pressure_diff = np.zeros(self.avg_data.shape)
            return pressure_diff.flatten()
        else:
            raise ValueError

        pu1 = self.avg_data.PU_vectorDF.loc[PhyTime,comp1]\
                        .values.reshape(self.avg_data.shape)
        pu2 = self.avg_data.PU_vectorDF.loc[PhyTime,comp2]\
                        .values.reshape(self.avg_data.shape)

        rho_star = 1.0
        pu1_grad = Grad_calc(self.avg_data.CoordDF,pu1,diff1)
        pu2_grad = Grad_calc(self.avg_data.CoordDF,pu2,diff2)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff.flatten()

    def _viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        u1u2 = self.avg_data.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape(self.avg_data.shape)

        REN = self.avg_data._metaDF.loc['REN'].values[0]
        viscous_diff = (1/REN)*_scalar_laplacian(self.avg_data.CoordDF,u1u2)
        return viscous_diff.flatten()

    def _production_extract(self,PhyTime,comp1,comp2):
        U1U_comp = comp1 + 'u'
        U2U_comp = comp2 + 'u'
        U1V_comp = comp1 + 'v'
        U2V_comp = comp2 + 'v'
        
        uu_comp_list = [U1U_comp, U2U_comp,U1V_comp, U2V_comp]
        for i in range(len(uu_comp_list)):
            if ord(uu_comp_list[i][0]) > ord(uu_comp_list[i][1]):
                uu_comp_list[i] = uu_comp_list[i][::-1]
                
        U1U_comp, U2U_comp,U1V_comp, U2V_comp = itertools.chain(uu_comp_list)
        u1u = self.avg_data.UU_tensorDF.loc[PhyTime,U1U_comp]\
                        .values.reshape(self.avg_data.shape)
        u2u = self.avg_data.UU_tensorDF.loc[PhyTime,U2U_comp]\
                        .values.reshape(self.avg_data.shape)
        u1v = self.avg_data.UU_tensorDF.loc[PhyTime,U1V_comp]\
                        .values.reshape(self.avg_data.shape)
        u2v = self.avg_data.UU_tensorDF.loc[PhyTime,U2V_comp]\
                        .values.reshape(self.avg_data.shape)

        U1x_comp = comp1 + 'x'
        U2x_comp = comp2 + 'x'
        U1y_comp = comp1 + 'y'
        U2y_comp = comp2 + 'y'
        
        du1dx = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U1x_comp]\
                        .values.reshape(self.avg_data.shape)
        du2dx = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U2x_comp]\
                        .values.reshape(self.avg_data.shape)
        du1dy = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U1y_comp]\
                        .values.reshape(self.avg_data.shape)
        du2dy = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U2y_comp]\
                        .values.reshape(self.avg_data.shape)

        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production.flatten()
    def _dissipation_extract(self,PhyTime,comp1,comp2):
        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        dU1dx_comp = comp1 + 'u'
        dU2dx_comp = comp2 + 'u'
        dU1dy_comp = comp1 + 'v'
        dU2dy_comp = comp2 + 'v'
        
        du1dxdu2dx = self.avg_data.DUDX2_tensorDF.loc[PhyTime,dU1dxdU2dx_comp]\
                        .values.reshape(self.avg_data.shape)
        du1dydu2dy = self.avg_data.DUDX2_tensorDF.loc[PhyTime,dU1dydU2dy_comp]\
                        .values.reshape(self.avg_data.shape)

        REN = self.avg_data._metaDF.loc['REN'].values[0]
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation.flatten()

    def budget_plot(self, x_list,PhyTime='',wall_units=True, fig='', ax ='',**kwargs):
        if len(set([x[0] for x in self.avg_data.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.avg_data.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super()._budget_plot(PhyTime, x_list,wall_units=wall_units, fig=fig, ax =ax,**kwargs)
        for a,x in zip(ax,x_list):
            a.set_title(r"$x^*=%.2f$"%x,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,comp=None, PhyTime='', wall_units=True, fig='', ax='', **kwargs):
        if len(set([x[0] for x in self.avg_data.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.avg_data.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super()._plot_integral_budget(comp,PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)

        return fig, ax
    def plot_budget_x(self,comp=None,y_vals_list='max',Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.avg_data.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.avg_data.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)

        
        if comp ==None:
            comp_list = [x[1] for x in self.budgetDF.index]
            comp_len = len(comp_list)
            for comp in comp_list:
                fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        else:
            comp_len = 1
            fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_budget_tg(cbase.CHAPSim_budget_base):
    def __init__(self,*args,**kwargs):
        if 'PhyTime' in kwargs.keys():
            raise KeyError("PhyTime cannot be used in tg class\n")
        super().__init__(*args,PhyTime=float('nan'),**kwargs)

    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape(self.avg_data.shape)
        U_mean = self.avg_data.flow_AVGDF.loc[PhyTime,'u']\
                        .values.reshape(self.avg_data.shape)
        V_mean = self.avg_data.flow_AVGDF.loc[PhyTime,'v']\
                        .values.reshape(self.avg_data.shape)
        # uu_dx = Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = CT.Grad_calc_tg(self.avg_data.CoordDF,uu)

        advection = -V_mean*uu_dy
        return advection.flatten()

    def _turb_transport(self,PhyTime,comp1,comp2):
        uu_comp = comp1+comp2
        uu_comp2 = uu_comp+'v'

        if ord(uu_comp2[0]) > ord(uu_comp2[1]):
            uu_comp2 = uu_comp2[:2][::-1] + uu_comp2[2]
        if ord(uu_comp2[0]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[::-1]
        if ord(uu_comp2[1]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[0] + uu_comp2[1:][::-1]

        u1u2v = self.avg_data.UUU_tensorDF.loc[PhyTime,uu_comp2]\
                        .values.reshape(self.avg_data.shape)

        u1u2v_dy = CT.Grad_calc_tg(self.avg_data.CoordDF,u1u2v)

        turb_transport = -u1u2v_dy
        return turb_transport.flatten()

    def _pressure_strain(self,PhyTime,comp1,comp2):
        u1u2 = comp1 + chr(ord(comp2)-ord('u')+ord('x'))
        u2u1 = comp2 + chr(ord(comp1)-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.avg_data.PR_Velo_grad_tensorDF.loc[PhyTime,u1u2]\
                        .values.reshape(self.avg_data.shape)
        pdu2dx1 = self.avg_data.PR_Velo_grad_tensorDF.loc[PhyTime,u2u1]\
                        .values.reshape(self.avg_data.shape)
        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain.flatten()

    def _pressure_diffusion(self,PhyTime,comp1,comp2):

        diff1 = chr(ord(comp2)-ord('u')+ord('x'))
        diff2 = chr(ord(comp1)-ord('u')+ord('x'))
        comp_list = ['u','v','w']
        if comp1 not in comp_list:
            raise ValueError("comp1 must be %s, %s, or %s not %s"%(*comp_list,comp1))
        if comp2 not in comp_list:
            raise ValueError("comp2 must be %s, %s, or %s not %s"%(*comp_list,comp2))

        pu1 = self.avg_data.PU_vectorDF.loc[PhyTime,comp1]\
                        .values.reshape(self.avg_data.shape)
        pu2 = self.avg_data.PU_vectorDF.loc[PhyTime,comp2]\
                        .values.reshape(self.avg_data.shape)

        rho_star = 1.0
        if diff1 == 'y':
            pu1_grad = CT.Grad_calc_tg(self.avg_data.CoordDF,pu1)
        else:
            pu1_grad = np.zeros(self.avg_data.shape)

        if diff2 == 'y':
            pu2_grad = CT.Grad_calc_tg(self.avg_data.CoordDF,pu2)
        else:
            pu2_grad = np.zeros(self.avg_data.shape)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff.flatten()

    def _viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        u1u2 = self.avg_data.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape(self.avg_data.shape)

        REN = self.avg_data._metaDF.loc['REN'].values[0]
        viscous_diff = (1/REN)*CT.Scalar_laplacian_tg(self.avg_data.CoordDF,u1u2)
        return viscous_diff.flatten()

    def _production_extract(self,PhyTime,comp1,comp2):
        U1U_comp = comp1 + 'u'
        U2U_comp = comp2 + 'u'
        U1V_comp = comp1 + 'v'
        U2V_comp = comp2 + 'v'
        
        uu_comp_list = [U1U_comp, U2U_comp,U1V_comp, U2V_comp]
        for i in range(len(uu_comp_list)):
            if ord(uu_comp_list[i][0]) > ord(uu_comp_list[i][1]):
                uu_comp_list[i] = uu_comp_list[i][::-1]
                
        U1U_comp, U2U_comp,U1V_comp, U2V_comp = itertools.chain(uu_comp_list)
        u1u = self.avg_data.UU_tensorDF.loc[PhyTime,U1U_comp]\
                        .values.reshape(self.avg_data.shape)
        u2u = self.avg_data.UU_tensorDF.loc[PhyTime,U2U_comp]\
                        .values.reshape(self.avg_data.shape)
        u1v = self.avg_data.UU_tensorDF.loc[PhyTime,U1V_comp]\
                        .values.reshape(self.avg_data.shape)
        u2v = self.avg_data.UU_tensorDF.loc[PhyTime,U2V_comp]\
                        .values.reshape(self.avg_data.shape)

        U1x_comp = comp1 + 'x'
        U2x_comp = comp2 + 'x'
        U1y_comp = comp1 + 'y'
        U2y_comp = comp2 + 'y'
        
        du1dx = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U1x_comp]\
                        .values.reshape(self.avg_data.shape)
        du2dx = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U2x_comp]\
                        .values.reshape(self.avg_data.shape)
        du1dy = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U1y_comp]\
                        .values.reshape(self.avg_data.shape)
        du2dy = self.avg_data.Velo_grad_tensorDF.loc[PhyTime,U2y_comp]\
                        .values.reshape(self.avg_data.shape)

        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production.flatten()
    
    def _dissipation_extract(self,PhyTime,comp1,comp2):
        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        du1dxdu2dx = self.avg_data.DUDX2_tensorDF.loc[PhyTime,dU1dxdU2dx_comp]\
                        .values.reshape(self.avg_data.shape)
        du1dydu2dy = self.avg_data.DUDX2_tensorDF.loc[PhyTime,dU1dydU2dy_comp]\
                        .values.reshape(self.avg_data.shape)
        print(dU1dxdU2dx_comp,du1dxdu2dx)
        print(dU1dydU2dy_comp,du1dydu2dy)
        REN = self.avg_data._metaDF.loc['REN'].values[0]
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation.flatten()

    def budget_plot(self, times_list,wall_units=True, fig='', ax ='',**kwargs):
        PhyTime = float('nan')
        fig, ax = super()._budget_plot(PhyTime, times_list,wall_units=wall_units, fig=fig, ax =ax,**kwargs)

        for a,t in zip(ax,times_list):
            a.set_title(r"$t^*=%.2f$"%t,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,comp=None, wall_units=True, fig='', ax='', **kwargs):
        PhyTime = float('nan')
        fig, ax = super()._plot_integral_budget(comp=comp,PhyTime=PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)
        ax.set_xlabel(r"$t^*$")
        return fig, ax

    def plot_budget_x(self,comp=None,y_vals_list='max',Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        PhyTime = float('nan')        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)

        
        if comp ==None:
            comp_list = [x[1] for x in self.budgetDF.index]
            comp_len = len(comp_list)
            for comp in comp_list:
                fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        else:
            comp_len = 1
            fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        
        ax.set_xlabel(r"$t^*$")
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_autocov_io(cbase.CHAPSim_autocov_base):
    def _autocov_extract(self,comp1,comp2,path_to_folder='',time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = time_extract(path_to_folder,abs_path)
        times = list(filter(lambda x: x > time0, times))
        if TEST:
            times.sort(); times= times[-3:]
            
        meta_data = CHAPSim_meta(path_to_folder)
        comp=(comp1,comp2)
        NCL = meta_data.NCL
        try:
            avg_data = CHAPSim_AVG_io(max(times),meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            avg_data = CHAPSim_AVG_io(max(times),meta_data,path_to_folder,time0)
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
            fluct_data = CHAPSim_fluct_io(timing,avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                autocorr = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_autocorr = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                assert local_autocorr.shape == autocorr.shape, "shape of previous array (%d,%d) " % autocorr.shape\
                    + " and current array (%d,%d) must be the same" % local_autocorr.shape
                autocorr = autocorr*coe3 + local_autocorr*coe2
            i += 1
            index=['x','z']
            autocorrDF = pd.DataFrame(autocorr.T,index=index)
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
   
    def _hdf_extract(self,file_name, group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov_io'
        hdf_file = h5py.File(file_name,'r')
        shape_x = tuple(hdf_file[base_name].attrs["shape_x"][:])
        shape_z = tuple(hdf_file[base_name].attrs["shape_z"][:])
        comp = tuple(np.char.decode(hdf_file[base_name].attrs["comp"][:]))
        hdf_file.close()        
        autocorrDF = pd.read_hdf(file_name,key=base_name+'/autocorrDF')
        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = CHAPSim_AVG_io.from_hdf(file_name,base_name+'/avg_data')
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
    
    def save_hdf(self, file_name, write_mode, group_name=''):
        if not group_name:
            group_name = 'CHAPSim_autocov_io'
        super().save_hdf(file_name, write_mode, group_name=group_name)
    
    @staticmethod
    def _autocov_calc(fluct_data,comp1,comp2,PhyTime,max_x_sep,max_z_sep):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        NCL=fluct_data.NCL        

        fluct_vals1=fluct_data.fluctDF.loc[PhyTime,comp1].values.reshape(fluct_data.shape)
        fluct_vals2=fluct_data.fluctDF.loc[PhyTime,comp2].values.reshape(fluct_data.shape)
        time1=time.time()
        
        R_x = CHAPSim_autocov_io._autocov_calc_x(fluct_vals1,fluct_vals2,*NCL,max_x_sep)
        R_z = CHAPSim_autocov_io._autocov_calc_z(fluct_vals1,fluct_vals2,*NCL,max_z_sep)
        print(time.time()-time1)
        R_z = R_z/(NCL[2]-max_z_sep)
        R_x = R_x/(NCL[2])

        R_z = R_z.reshape((max_z_sep*NCL[1]*NCL[0]))
        Rz_DF = pd.DataFrame(R_z)
        R_x = R_x.reshape((max_x_sep*NCL[1]*(NCL[0]-max_x_sep)))
        Rx_DF = pd.DataFrame(R_x)
        R_DF=pd.concat([Rx_DF,Rz_DF],axis=1)
        
        return R_DF.values
    @staticmethod
    @numba.njit(parallel=True)
    def _autocov_calc_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        R_z = np.zeros((max_z_step,NCL2,NCL1))
        if max_z_step >0:
            for iz0 in numba.prange(max_z_step):
                for iz in numba.prange(NCL3-max_z_step):
                    R_z[iz0,:,:] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
        return R_z
    @staticmethod
    @numba.njit(parallel=True)
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
            a.axes.set_xlabel(r"x^*")
        return fig, ax

    def spectrum_contour(self, comp,*args,**kwargs):
        fig, ax =  super().spectrum_contour(comp,*args,**kwargs)
        for a in ax:
            a.axes.set_xlabel(r"x^*")
        return fig, ax

class CHAPSim_autocov_tg(cbase.CHAPSim_autocov_base):
    def _autocov_extract(self,comp1,comp2,path_to_folder='',time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if TEST:
            times.sort(); times= times[-3:]
            
        meta_data = CHAPSim_meta(path_to_folder)
        comp=(comp1,comp2)
        NCL = meta_data.NCL
        avg_data = CHAPSim_AVG_tg_base(times,meta_data=meta_data,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path)

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
            fluct_data = CHAPSim_fluct_tg(timing,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            if 'R_z' not in locals():
                R_z, R_x = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_R_z, local_R_x = self._autocov_calc(fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                R_z = np.vstack([R_z,local_R_z])
                print(R_z.shape)
                R_x = np.vstack([R_x,local_R_x])

        R_z = pd.Series(R_z.T.flatten())
        R_x = pd.Series(R_x.T.flatten())
        
        autocorr = pd.concat([R_x,R_z],axis=1)
        index=['x','z']
        autocorrDF = pd.DataFrame(autocorr.values.T,index=index)
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z
    
    def _hdf_extract(self,file_name, group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov_tg'
        hdf_file = h5py.File(file_name,'r')
        shape_x = tuple(hdf_file[base_name].attrs["shape_x"][:])
        shape_z = tuple(hdf_file[base_name].attrs["shape_z"][:])
        comp = tuple(np.char.decode(hdf_file[base_name].attrs["comp"][:]))
        hdf_file.close()        
        autocorrDF = pd.read_hdf(file_name,key=base_name+'/autocorrDF')
        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = CHAPSim_AVG_tg.from_hdf(file_name,base_name+'/avg_data')
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

        fluct_vals1=fluct_data.fluctDF.loc[PhyTime,comp1].values.reshape(fluct_data.shape)
        fluct_vals2=fluct_data.fluctDF.loc[PhyTime,comp2].values.reshape(fluct_data.shape)
        print(PhyTime)
        R_x = CHAPSim_autocov_tg._autocov_calc_x(fluct_vals1,fluct_vals2,*NCL,max_x_sep)
        R_z = CHAPSim_autocov_tg._autocov_calc_z(fluct_vals1,fluct_vals2,*NCL,max_z_sep)

        R_z = np.mean(R_z,axis=2)/(NCL[2]-max_z_sep)
        R_x = np.mean(R_x,axis=2)/(NCL[2])
        
        return R_z.flatten(), R_x.flatten()

    @staticmethod
    @numba.njit(parallel=True)
    def _autocov_calc_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        R_z = np.zeros((max_z_step,NCL2,NCL1))
        if max_z_step >0:
            for iz0 in numba.prange(max_z_step):
                for iz in numba.prange(NCL3-max_z_step):
                    R_z[iz0] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
        return R_z
    @staticmethod
    @numba.njit(parallel=True)
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

class CHAPSim_Quad_Anl_io(cbase.CHAPSim_Quad_Anl_base):
    def _quad_extract(self,h_list,path_to_folder='',time0=None,abs_path=True):
        times = time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if TEST:
            times.sort(); times= times[-3:]
        meta_data = CHAPSim_meta(path_to_folder,abs_path)
        NCL = meta_data.NCL
        try:
            avg_data = CHAPSim_AVG_io(max(times),meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            avg_data = CHAPSim_AVG_io(max(times),meta_data,path_to_folder,time0)
        i=1
        for timing in times:
            fluct_data = CHAPSim_fluct_io(timing,avg_data,time0=time0,path_to_folder=path_to_folder,
                                        abs_path=abs_path)
            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,timing,NCL)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
            else:
                local_quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
                assert local_quad_anal_array.shape == quad_anal_array.shape, "shape of previous array (%d,%d) " % quad_anal_array.shape\
                    + " and current array (%d,%d) must be the same" % local_quad_anal_array.shape
                quad_anal_array = quad_anal_array*coe3 + local_quad_anal_array*coe2
            i += 1
        index=[[],[]]
        for h in h_list:
            index[0].extend([h]*4)
        index[1]=[1,2,3,4]*len(h_list)
        QuadAnalDF=pd.DataFrame(quad_anal_array,index=index)
        shape = avg_data.shape
        return meta_data, NCL, avg_data, QuadAnalDF, shape

    def _hdf_extract(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Quad_Anal'
        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL= meta_data.NCL
        avg_data = CHAPSim_AVG_io.from_hdf(file_name,base_name+'/avg_data')
        QuadAnalDF = pd.read_hdf(file_name,key=base_name+'/QuadAnalDF')
        shape = (NCL[1],NCL[0])
        return meta_data, NCL, avg_data, QuadAnalDF, shape

    @staticmethod
    def _quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,PhyTime):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)

        avg_time = list(set([x[0] for x in avg_data.UU_tensorDF.index]))[0]
        uv_q=np.zeros((4,*NCL[::-1][1:]))

    
        uu=avg_data.UU_tensorDF.loc[avg_time,'uu'].values.reshape(NCL[::-1][1:])
        vv=avg_data.UU_tensorDF.loc[avg_time,'vv'].values.reshape(NCL[::-1][1:])
        u_rms = np.sqrt(uu)
        v_rms = np.sqrt(vv)

        quad_anal_array=np.empty((len(h_list)*4,NCL[0]*NCL[1]))
        for h,j in zip(h_list,range(len(h_list))):
            for i in range(1,5):
                quad_array=quadrant_array == i
                # print(quad_array)
                fluct_array = np.abs(quad_array*fluct_uv) > h*u_rms*v_rms
                uv_q[i-1]=np.mean(fluct_uv*fluct_array,axis=0)
            quad_anal_array[j*4:j*4+4]=uv_q.reshape((4,NCL[0]*NCL[1]))
        return quad_anal_array

class CHAPSim_Quad_Anl_tg(cbase.CHAPSim_Quad_Anl_base):
    def _quad_extract(self,h_list,path_to_folder='',time0=None,abs_path=True):
        times = time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if TEST:
            times.sort(); times= times[-3:]
        meta_data = CHAPSim_meta(path_to_folder,abs_path)
        NCL = meta_data.NCL

        avg_data = CHAPSim_AVG_tg_base(times,meta_data=meta_data,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path)
        
        for timing in times:
            fluct_data = CHAPSim_fluct_tg(timing,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,timing,NCL)
            
            if 'quad_anal_array' not in locals():
                quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
            else:
                local_quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
                quad_anal_array = np.vstack([quad_anal_array,local_quad_anal_array])

        index=[[],[]]
        for h in h_list:
            index[0].extend([h]*4)
        index[1]=[1,2,3,4]*len(h_list)
        QuadAnalDF=pd.DataFrame(quad_anal_array,index=index)
        shape = avg_data.shape
        return meta_data, NCL, avg_data, QuadAnalDF, shape
class CHAPSim_joint_PDF_io(cbase.CHAPSim_joint_PDF_base):
    _module = sys.modules[__module__]
    def _extract_fluct(self,x,y,path_to_folder=None,time0=None,y_mode='half-channel',use_ini=True,xy_inner=True,tgpost=False,abs_path=True):
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if TEST:
            times.sort(); times= times[-3:]
        meta_data = self._module.CHAPSim_meta(path_to_folder,abs_path)
        NCL = meta_data.NCL

        try:
            avg_data = self._module.CHAPSim_AVG_io(max(times),meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            avg_data = self._module.CHAPSim_AVG_io(max(times),meta_data,path_to_folder,time0)

        
        if xy_inner:
            if len(x) != len(y):
                msg = "length of x coordinate array must be same"+\
                        " as the y coord array. Lengths provided %d (x),"%len(x)+\
                            " %d (y)"%len(y)
                raise ValueError(msg)
            
            x_coord_list = x; y_coord_list = y
        else:
            x_coord_list = []; y_coord_list=[]
            for x_val in x:
                for y_val in y:
                    x_coord_list.append(x_val)
                    y_coord_list.append(y_val)

        x_index = CT.coord_index_calc(avg_data.CoordDF,'x',x_coord_list)

        x_loc_norm = x_coord_list if not use_ini else [0]*len(y_coord_list)
        y_index = CT.y_coord_index_norm(avg_data,avg_data.CoordDF,y_coord_list,x_loc_norm,y_mode)
        
        y_index = np.diag(np.array(y_index))
        print(y_index)
        u_prime_array = [ [] for _ in range(len(y_index)) ]
        v_prime_array = [ [] for _ in range(len(y_index)) ]
        for time in times:
            fluct_data = CHAPSim_fluct_io(time,avg_data,path_to_folder,abs_path)
            u_prime_data = fluct_data.fluctDF.loc["%g"%time,'u'].values.reshape(fluct_data.shape)
            v_prime_data = fluct_data.fluctDF.loc["%g"%time,'v'].values.reshape(fluct_data.shape)
            for i in range(len(y_index)):
                u_prime_array[i].extend(u_prime_data[:,y_index[i],x_index[i]])
                v_prime_array[i].extend(v_prime_data[:,y_index[i],x_index[i]])

        val_index = list(zip(x_coord_list,y_coord_list))
        comp_index = ['u']*len(val_index) + ['v']*len(val_index)
        columns = [comp_index,val_index*2]

        u_prime_array = np.array(u_prime_array); v_prime_array = np.array(v_prime_array)
        uv_prime_array = np.vstack([u_prime_array,v_prime_array])

        uv_primeDF = pd.DataFrame(uv_prime_array.T,columns = columns)
        return uv_primeDF, avg_data,meta_data,NCL,y_mode,x_loc_norm

    def save_hdf(self,file_name,mode='a',base_name='CHAPSim_joint_PDF_io'):
        super().save_hdf(file_name,mode,base_name)

    def _hdf_extract(self, file_name, base_name):
        if base_name is None:
            base_name = 'CHAPSim_joint_PDF_io' 

        hdf_file = h5py.File(file_name,'r')
        y_mode = hdf_file[base_name].attrs['y_mode'].decode('utf-8')
        x_loc_norm = hdf_file[base_name+'/x_loc_norm'][:]
        hdf_file.close()
        meta_data = self._module.CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = self._module.CHAPSim_AVG_io.from_hdf(file_name,base_name+'/avg_data')
        uv_primeDF = pd.read_hdf(file_name,key=base_name+'/uv_primeDF')
        return uv_primeDF,avg_data, meta_data, NCL, y_mode,x_loc_norm


def file_extract(path_to_folder,abs_path=True):
    if abs_path:
        mypath = os.path.join(path_to_folder,'1_instant_D')
    else:
        mypath = os.path.abspath(os.path.join(path_to_folder,'1_instant_D'))
    file_names = [f for f in os.listdir(mypath) if f[:8]=='DNS_peri']
    return file_names       

def time_extract(path_to_folder,abs_path=True):
    file_names = file_extract(path_to_folder,abs_path)
    time_list =[]
    for file in file_names:
        time_list.append(float(file[20:35]))
    times = list(dict.fromkeys(time_list))
    return list(set(times))

def _figuresize(x_split_list):
    number_v = len(x_split_list)-1
    number_h = 1
    i=2
    while number_v >= i**2:
        number_h += 1
        number_v /= number_h
        number_v = int(np.floor(number_v))
        i+=1
    return number_v, number_h

def _line_extract(DF,xcoordDF,NCL,x_list):
    """ Extracts lines from dataframe/series input """
    #create array from input dataframe
    #print(NCL[1],NCL[0])
    DF_array = DF.values.reshape((NCL[1],NCL[0]))
    list_array = np.zeros((len(x_list),NCL[1]))
    x_coord = []
    i=0
    for x in x_list:
        #print(DF_array.shape)
        list_array[i]=DF_array[:,x]
        i += 1
        x_coord.append(xcoordDF['x'].loc[x])
    
    line_DF = pd.DataFrame(list_array,index=x_coord)
        
        
    return line_DF, x_coord

def wall_unit_calc(AVG_DF,PhyTime):
    if type(PhyTime) != str:
        PhyTime = "{:.9g}".format(PhyTime)
    y_coords = AVG_DF.CoordDF['y'].dropna().values
    NCL = AVG_DF.NCL
    
    u_array = AVG_DF.flow_AVGDF.loc[PhyTime,'u'].values.reshape((NCL[1],NCL[0]))
    #x_coords = coordsDF['x'].dropna().values
    
    mu_star = 1.0
    rho_star = 1.0
    nu_star = mu_star/rho_star
    umean_grad = np.zeros_like(u_array[1])
    REN = AVG_DF._metaDF.loc['REN'].values[0]
    
    wall_velo = AVG_DF._meta_data.moving_wall_calc()
    for i in range(NCL[0]):
        umean_grad[i] = mu_star*(u_array[0,i]-wall_velo[i])/(y_coords[0]--1.0)
        #print(u_array[0,i])
    #The scaled viscosity \mu* is 1 for isothermal flow
    tau_w = umean_grad
   
    u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
    delta_v_star = (nu_star/u_tau_star)/REN
    return u_tau_star, delta_v_star
    

    
def _vector_plot(CoordDF,twoD_DF,NCL,fig,ax,axis1_spacing,\
                         axis2_spacing,axis1='',axis2='',axis3_loc=''):
    if not axis1:
        axis1 = 'x'
        axis2 = 'y'   
        axis1, axis2, velo1_comp, velo2_comp = _axis_check(axis1,axis2)
        DF_values = twoD_DF.loc[[velo1_comp[0],velo2_comp[0]]].copy().values.reshape((2,NCL[1],NCL[0]))
        
    else:
        axis1, axis2, velo1_comp, velo2_comp = _axis_check(axis1,axis2)
        DF_values = twoD_DF.loc[[velo1_comp[0],velo2_comp[0]]].copy().values.reshape((2,NCL[2],NCL[1],NCL[0]))

    axis2_size = int(NCL[velo2_comp[1]]/axis2_spacing)
    axis1_size = int(NCL[velo1_comp[1]]/axis1_spacing)
    x_coords = CoordDF[axis1].dropna().values
    y_coords = CoordDF[axis2].dropna().values
    
    vector_array = np.zeros((2,axis2_size,axis1_size))
    local_x_coords = np.zeros(axis1_size)
    local_y_coords = np.zeros(axis2_size)
    
    for i in range(axis1_size):
        local_x_coords[i] = x_coords[i*axis1_spacing]
        for j in range(axis2_size):
            if not axis3_loc:
                vector_array[:,j,i] = DF_values[:,j*axis2_spacing,i*axis1_spacing]
            else:
                if axis1 == 'x' and axis2 == 'y':
                    vector_array[:,j,i] = DF_values[:,axis3_loc,j*axis2_spacing,i*axis1_spacing]
                elif axis1 == 'x' and axis2 == 'z':
                    vector_array[:,j,i] = DF_values[:,j*axis2_spacing,axis3_loc,i*axis1_spacing]
                elif axis1 == 'z' and axis2 == 'y':
                    vector_array[:,j,i] = DF_values[:,j*axis2_spacing,i*axis1_spacing,axis3_loc]
            if i == axis1_size-1:
                local_y_coords[j] = y_coords[j*axis2_spacing]
    q = ax.quiver(local_x_coords,local_y_coords,vector_array[0],vector_array[1],angles='uv', scale_units=None, scale=None)
    key_size = np.max(vector_array)
    ax.quiverkey(q,0.8,-0.125,key_size,r'$U/U_{b0}=%0.3g$'%key_size,labelpos='E',coordinates='axes')
    ax.set_xlabel(r"$%s$  " % axis1)
    ax.set_ylabel(r"$%s$  " % axis2)
    return fig,ax
def _axis_check(axis1,axis2):
    try:
        if axis1 == axis2:
            raise ValueError
    except ValueError:
        print("Axes cannot be the same")
        raise
    axes = ['x','y','z']
    if axis1 not in axes or axis2 not in axes:
        raise ValueError
    if axis1 != 'x' and axis1 != 'z':
        axis_temp = axis2
        axis2 = axis1
        axis1 = axis_temp
    elif axis1 == 'z' and axis2 == 'x':
        axis_temp = axis2
        axis2 = axis1
        axis1 = axis_temp
    
    if axis1 == 'x':
        velo1_comp = ['u',0]
    elif axis1 == 'z':
        velo1_comp = ['w',2]
    else:
        print('fall through2')
        
    if axis2 == 'y':
        velo2_comp = ['v',1]
    elif axis2 =='z':
        velo2_comp = ['w',2]
    else:
        print('fall through3')

    return axis1, axis2, velo1_comp,velo2_comp
def Grad_calc(coordDF,flowArray,comp,two_D=True):
    if two_D:
        assert(flowArray.ndim == 2)
    else:
        assert(flowArray.ndim == 3)
    coord_array = coordDF[comp].dropna().values
    grad = np.zeros_like(flowArray)
    sol_f, h_list_f = CT.Stencil_calc([0,1,2], 1)
    a_f, b_f, c_f = CT.Stencil_coeffs_eval(sol_f,h_list_f,[coord_array[1]-coord_array[0],coord_array[2]-coord_array[1]])
    
    sol_b, h_list_b = CT.Stencil_calc([-2,-1,0], 1)
    a_b, b_b, c_b = CT.Stencil_coeffs_eval(sol_b,h_list_b,[coord_array[-2]-coord_array[-3],coord_array[-1]-coord_array[-2]])
    
    sol_c, h_list_c = CT.Stencil_calc([-1,0,1],1)
    
    
    if two_D:
        if comp =='x':
            dim_size = flowArray.shape[1]
            
            grad[:,0] = a_f*flowArray[:,0] + b_f*flowArray[:,1] + c_f*flowArray[:,2]#(- flowArray[:,2] + 4*flowArray[:,1] - 3*flowArray[:,0])/(coord_array[2]-coord_array[0])
            for i in range(1,dim_size-1):
                #a_c, b_c, c_c = CT.Stencil_coeffs_eval(sol_c,h_list_c,[coord_array[i]-coord_array[i-1],coord_array[i+1]-coord_array[i]])
                #grad[:,i] = a_c*flowArray[:,i-1] + b_c*flowArray[:,i] + c_c*flowArray[:,i+1]#
                grad[:,i] =(flowArray[:,i+1] - flowArray[:,i-1])/(coord_array[i+1]-coord_array[i-1])
            
            grad[:,dim_size-1] = a_b*flowArray[:,-3] + b_b*flowArray[:,-2] + c_b*flowArray[:,-1]#(3*flowArray[:,dim_size-1] - 4*flowArray[:,dim_size-2] + flowArray[:,dim_size-3])\
                                #/(coord_array[dim_size-1]-coord_array[dim_size-3])
    
        elif comp =='y':
            dim_size = flowArray.shape[0]
            grad[0,:] = a_f*flowArray[0,:] + b_f*flowArray[1,:] + c_f*flowArray[2,:]#(-flowArray[2,:] + 4*flowArray[1,:] - 3*flowArray[0,:])/(coord_array[2]-coord_array[0])
            for i in range(1,dim_size-1):
                # a_c, b_c, c_c = CT.Stencil_coeffs_eval(sol_c,h_list_c,[coord_array[i]-coord_array[i-1],coord_array[i+1]-coord_array[i]])
                # grad[i] = a_c*flowArray[i-1] + b_c*flowArray[i] + c_c*flowArray[i+1]
                h1 = coord_array[i+1]-coord_array[i]
                h0 =  coord_array[i]-coord_array[i-1]
                grad[i] = -h1/(h0*(h0+h1))*flowArray[i-1] + (h1-h0)/(h0*h1)*flowArray[i] + h0/(h1*(h0+h1))*flowArray[i+1]
                #grad[i,:] = (flowArray[i+1,:] - flowArray[i-1,:])/(coord_array[i+1]-coord_array[i-1])
            grad[-1,:] = a_b*flowArray[-3,:] + b_b*flowArray[-2,:] + c_b*flowArray[-1,:]#(3*flowArray[dim_size-1,:] - 4*flowArray[dim_size-2,:] + flowArray[-3,:])\
                                #/(coord_array[dim_size-1]-coord_array[dim_size-3])
        else:    
            raise Exception
    else:
        if comp =='x':
            dim_size = flowArray.shape[2]
            grad[:,:,0] = a_f*flowArray[:,:,0] + b_f*flowArray[:,:,1] + c_f*flowArray[:,:,2]# (flowArray[:,:,1] - flowArray[:,:,0])/(coord_array[1]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[:,:,i] = (flowArray[:,:,i+1] - flowArray[:,:,i-1])/(coord_array[i+1]-coord_array[i-1])
            grad[:,:,-1] = a_b*flowArray[:,:,-3] + b_b*flowArray[:,:,-2] + c_b*flowArray[:,:,-1] #(flowArray[:,:,dim_size-1] - flowArray[:,:,dim_size-2])/(coord_array[dim_size-1]-coord_array[dim_size-2])
    
        elif comp =='y':
            dim_size = flowArray.shape[1]
            grad[:,0,:] = a_f*flowArray[:,0,:] + b_f*flowArray[:,1,:] + c_f*flowArray[:,2,:] #(flowArray[:,1,:] - flowArray[:,0,:])/(coord_array[1]-coord_array[0])
            for i in range(1,dim_size-1):
                a_c, b_c, c_c = CT.Stencil_coeffs_eval(sol_c,h_list_c,[coord_array[i]-coord_array[i-1],coord_array[i+1]-coord_array[i]])
                grad[:,i] = a_c*flowArray[:,i-1] + b_c*flowArray[:,i] + c_c*flowArray[:,i+1]
                #grad[:,i,:] = (flowArray[:,i+1,:] - flowArray[:,i-1,:])/(coord_array[i+1]-coord_array[i-1])
            grad[:,-1,:] = a_b*flowArray[:,-3,:] + b_b*flowArray[:,-2,:] + c_b*flowArray[:,-1,:]#(flowArray[:,dim_size-1,:] - flowArray[:,dim_size-2,:])/(coord_array[dim_size-1]-coord_array[dim_size-2])
        elif comp=='z':
            dim_size = flowArray.shape[0]
            grad[0,:,:] = a_f*flowArray[0,:,:] + b_f*flowArray[1,:,:] + c_f*flowArray[2,:,:]#(flowArray[1,:,:] - flowArray[0,:,:])/(coord_array[1]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[i,:,:] = (flowArray[i+1,:,:] - flowArray[i-1,:,:])/(coord_array[i+1]-coord_array[i-1])
            grad[dim_size-1,:,:] = a_b*flowArray[-3,:,:] + b_b*flowArray[-2,:,:] + c_b*flowArray[-1,:,:]#(flowArray[dim_size-1,:,:] - flowArray[dim_size-2,:,:])/(coord_array[dim_size-1]-coord_array[dim_size-2])

        else:    
            raise Exception
    return grad
def _scalar_grad(coordDF,flow_array,two_D=True):
    if two_D:
        assert(flow_array.ndim == 2)
        grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
    else:
        assert(flow_array.ndim == 3)
        grad_vector = np.zeros((3,flow_array.shape[0],flow_array.shape[1],\
                               flow_array.shape[2]))
    
    grad_vector[0] = Grad_calc(coordDF,flow_array,'x')
    grad_vector[1] = Grad_calc(coordDF,flow_array,'y')
    
    
    if not two_D:
        grad_vector[2] = Grad_calc(coordDF,flow_array,'z')
        
    return grad_vector
def _Vector_div(coordDF,vector_array,two_D=True):
    if two_D:
        assert(vector_array.ndim == 3)
    else:
        assert(vector_array.ndim == 4)
    
    grad_vector = np.zeros_like(vector_array)
    grad_vector[0] = Grad_calc(coordDF,vector_array[0],'x')
    grad_vector[1] = Grad_calc(coordDF,vector_array[1],'y')
    
    
    if not two_D:
        grad_vector[2] = Grad_calc(coordDF,vector_array[2],'z')
    
    if two_D:    
        div_scalar = np.zeros((vector_array.shape[1],vector_array.shape[2]))
    else:
        div_scalar = np.zeros((vector_array.shape[1],vector_array.shape[2],\
                               vector_array.shape[3]))
    
    div_scalar = np.sum(grad_vector,axis=0)
    
    return div_scalar
def _scalar_laplacian(coordDF,flow_array,two_D=True):
    grad_vector = _scalar_grad(coordDF,flow_array,two_D)
    lap_scalar = _Vector_div(coordDF,grad_vector,two_D)
    return lap_scalar
