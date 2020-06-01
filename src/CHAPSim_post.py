'''
#=============================================================================#
#=============== CHAPSim - postprocessing ====================================#
# Postprocessing python scripts developed to be run on desktops and           #  
# workstations. This has been tested on iceberg. Primarily developed for      #
# postprocessing the instantaneous and averaged fields of individual          #  
# timesteps . Methods that require extensive use of the filesystem            #
# particularly creating averages using the instantaneous data from CHAPSim    #
# are advised to use the parallel module CHAPSim_parallel. Based principally  #
# on the Pandas DataFrame class                                               #
#                                                                             #
#===================== Structures and classes ================================#                                             #  
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


import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate, fftpack
import os
import itertools
import time
import warnings
import CHAPSim_Tools as CT
#import plotly.graph_objects as go

#import loop_accel as la
import numba
#on iceberg HPC vtkmodules not found allows it to work as this resource isn't need on iceberg
try: 
    import pyvista as pv
except ImportError:
    warnings.warn("\033[1;33module `pyvista' has missing modules will not work correctly")
mpl.rcParams['mathtext.fontset'] = 'stix'


class CHAPSim_Inst():
    def __init__(self,time,meta_data='',path_to_folder='',abs_path = True,tgpost=False):
        #Extract x,y,z coordinates that match for entire dataset
        #Extract velocity components and pressure from files
        if not meta_data:
            meta_data = CHAPSim_meta(path_to_folder,abs_path,tgpost)
        self.CoordDF = meta_data.CoordDF
        self.NCL = meta_data.NCL
        self._meta_data=meta_data
        #Give capacity for both float and lists
        if isinstance(time,float): 
            self.InstDF = self.__flow_extract(time,path_to_folder,abs_path,tgpost)
        elif hasattr(time,'__iter__'):
            for PhyTime in time:
                if not hasattr(self, 'InstDF'):
                    self.InstDF = self.__flow_extract(PhyTime,path_to_folder,abs_path,tgpost)
                else: #Variable already exists
                    local_DF = self.__flow_extract(PhyTime,path_to_folder,abs_path,tgpost)
                    concat_DF = [self.InstDF,local_DF]
                    self.InstDF = pd.concat(concat_DF)
        else:
            raise TypeError("\033[1;32 `time' must be either float or list")

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
    def inst_contour(self,axis1,axis2,axis3_value,flow_field,PhyTime,fig='',ax=''):
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
            fig,ax = plt.subplots(figsize=[10,3])
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
        cbar.set_label(cbar_label,fontsize=12)
        ax.set_xlabel(r"$%s/\delta$" % axis1,fontsize=18)
        ax.set_ylabel(r"$%s/\delta$" % axis2,fontsize=16)
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
    
    def lambda2_calc(self,PhyTime):
        #Calculating strain rate tensor
        velo_list = ['u','v','w']
        coord_list = ['x','y','z']
        strain_rate = np.zeros((self.NCL[2],self.NCL[1],self.NCL[0],3,3))
        rot_rate =  np.zeros((self.NCL[2],self.NCL[1],self.NCL[0],3,3))
        i=0
        for velo1,coord1 in zip(velo_list,coord_list):
            j=0
            for velo2,coord2 in zip(velo_list,coord_list):
                velo_field1 = self.InstDF.loc[PhyTime,velo1].values\
                    .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))
                velo_field2 = self.InstDF.loc[PhyTime,velo2].values\
                    .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))
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
        del e_vecs
        #lambda2 = np.zeros_like(velo_field1)
        # eigs = np.zeros(3)
        
        lambda2 = np.sort(S2_Omega2_eigvals,axis=3)[:,:,:,1]
        
        # for i in range(inst_data.NCL[2]):
        #     for j in range(inst_data.NCL[1]):
        #         for k in range(inst_data.NCL[0]):
        #             eigs = np.sort(S2_Omega2_eigvals[i,j,k,:])
        #             lambda2[i,j,k] = eigs[1]
        return lambda2
    def plot_lambda2(self,lambda_min=-float("inf"),lambda_max=float("inf"),PhyTime='',ylim='',Y_plus=True,avg_data='',plotter='',cmap='Reds_r'):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.InstDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.InstDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.InstDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        lambda2 = self.lambda2_calc(PhyTime)
        
        
        x_coords = self._meta_data.Coord_ND_DF['x'].dropna().values
        y_coords = self._meta_data.Coord_ND_DF['y'].dropna().values
        z_coords = self._meta_data.Coord_ND_DF['z'].dropna().values
        if ylim:
            if Y_plus:
                y_index= CT.Y_plus_index_calc(avg_data,self.CoordDF,ylim)
            else:
                y_index=CT.coord_index_calc(self.CoordDF,'y',ylim)
            y_coords=y_coords[:y_index+1]
            lambda2=lambda2[:,:y_index,:]
            
        Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)
        
        grid = pv.StructuredGrid(X,Z,Y)
        grid.cell_arrays['lambda_2']=lambda2.flatten()
        pgrid=grid.cell_data_to_point_data()
        contour=pgrid.contour(isosurfaces=1,scalars='lambda_2',preference='point',rng=(lambda_min,lambda_max))
        #grid.set_active_scalars('lambda_2')
        if not plotter:
            plotter = pv.BackgroundPlotter(notebook=False)
            plotter.set_background('white')
            plotter.show_bounds(color='black')
            
        
        #plotter.add_mesh(grid)
        #filter_grid = grid.threshold((lambda_min,lambda_max),'lambda_2')
        
        
        #plotter.add_axes(color='black')
        plotter.add_mesh(contour,interpolate_before_map=True,cmap=cmap)
        plotter.remove_scalar_bar()
        
        
        return plotter
    # def plot_streaks(self,AVG_DF,comp,vals_list,PhyTime='',ylim='',Y_plus=True,cmap='Greens_r',plotter=''):
    #     if PhyTime:
    #         if type(PhyTime) == float:
    #             PhyTime = "{:.9g}".format(PhyTime)
                
    #     if len(set([x[0] for x in self.InstDF.index])) == 1:
    #         avg_time = list(set([x[0] for x in self.InstDF.index]))[0]
    #         if PhyTime and PhyTime != avg_time:
    #             warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
    #         PhyTime = avg_time
    #     else:
    #         assert PhyTime in set([x[0] for x in self.InstDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
    #     x_coords = self._meta_data.Coord_ND_DF['x'].dropna().values
    #     y_coords = self._meta_data.Coord_ND_DF['y'].dropna().values
    #     z_coords = self._meta_data.Coord_ND_DF['z'].dropna().values
        
    #     inst_velo = self.InstDF.loc[PhyTime,comp].values\
    #                 .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))
    #     avg_velo = AVG_DF.flow_AVGDF.loc[PhyTime,comp].values\
    #                 .reshape((self.NCL[1],self.NCL[0]))
    #     if ylim:
    #         if Y_plus:
    #             y_index= CT.Y_plus_index_calc(AVG_DF,self.CoordDF,ylim)
    #         else:
    #             y_index=CT.coord_index_calc(self.CoordDF,'y',ylim)
    #         y_coords=y_coords[:y_index+1]
    #     else:
    #         y_index=self.NCL[1]
            
    #     fluct_velo=np.zeros((self.NCL[2],y_index,self.NCL[0]))
    #     for i in range(self.NCL[2]):
    #         fluct_velo[i] = inst_velo[i,:y_index]-avg_velo[:y_index]
                    
    #     Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)
    #     grid = pv.StructuredGrid(X,Z,Y)
    #     grid.cell_arrays['fluct']=fluct_velo.flatten()
    #     pgrid=grid.cell_data_to_point_data()
    #     if isinstance(vals_list, int) or isinstance(vals_list, float):
    #         vals_list=[vals_list]
    #     elif not isinstance(vals_list, list):
    #         raise TypeError("\033[1;32 Argument `vals_list' must be type int, float, or list ")
    #     cmp_list=['Greens_r','Blues','Reds_r']
    #     if cmap != 'Greens_r':
    #         cmp_list.insert(0,cmap)
    #     for val,i in zip(vals_list,range(len(vals_list))):
    #         contour=pgrid.contour(isosurfaces=1,scalars='fluct',preference='point',rng=(val,val))
    #         #grid.set_active_scalars('lambda_2')
    #         if not plotter:
    #             plotter = pv.BackgroundPlotter(notebook=False)
    #             plotter.set_background('white')
    #             plotter.show_bounds(color='black')
            
    #         plotter.add_mesh(contour,interpolate_before_map=True,
    #                          cmap=cmp_list[i%len(cmp_list)])
    #         plotter.remove_scalar_bar()
        
    #     return plotter
    def vorticity_calc(self):
        pass
    def plot_entrophy(self):
        pass
    def __str__(self):
        return self.InstDF.__str__()
class CHAPSim_AVG():
    def __init__(self,time,meta_data='',path_to_folder='',time0='',abs_path=True,tgpost=False):
        
        if meta_data:
            self._meta_data = meta_data
        else:
            self._meta_data = CHAPSim_meta(path_to_folder,abs_path,tgpost)
        self.CoordDF = self._meta_data.CoordDF
        self._metaDF = self._meta_data.metaDF
        self.NCL = self._meta_data.NCL
       
        if isinstance(time,float):
            DF_list = self.__AVG_extract(time,time0,path_to_folder,abs_path,tgpost)
            self.flow_AVGDF = DF_list[0]
            self.PU_vectorDF = DF_list[1]
            self.UU_tensorDF = DF_list[2]
            self.UUU_tensorDF = DF_list[3]
            self.Velo_grad_tensorDF = DF_list[4]
            self.PR_Velo_grad_tensorDF = DF_list[5]
            self.DUDX2_tensorDF = DF_list[6]
        elif isinstance(time,list):
            for PhyTime in time:
                if not hasattr(self, 'flow_AVGDF'):
                    DF_list = self.__AVG_extract(PhyTime,time0,path_to_folder,abs_path,tgpost)
                else:
                    DF_temp=[]
                    local_DF_list = self.__AVG_extract(PhyTime,time0,path_to_folder,abs_path,tgpost)
                    for DF, local_DF in zip(DF_list, local_DF_list):
                        concat_DF = [DF, local_DF]
                        
                        DF_temp.append(pd.concat(concat_DF))
                    DF_list=DF_temp
                    
                self.flow_AVGDF = DF_list[0]
                self.PU_vectorDF = DF_list[1]
                self.UU_tensorDF = DF_list[2]
                self.UUU_tensorDF = DF_list[3]
                self.Velo_grad_tensorDF = DF_list[4]
                self.PR_Velo_grad_tensorDF = DF_list[5]
                self.DUDX2_tensorDF = DF_list[6]
        else:
            raise TypeError("\033[1;32 `time' can only be a float or a list")
        
        
    def __AVG_extract(self,Time_input,time0,path_to_folder,abs_path,tgpost):
        if time0:
            instant = "%0.9E" % time0
            if tgpost:
                file_string = "DNS_perixz_AVERAGD_T" + instant + "_FLOW.D"
            else:
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
            #REN = r_info[1]
            #DT = r_info[2]
            AVG_info0 = np.zeros(dummy_size)
            AVG_info0 = np.fromfile(file,dtype='float64',count=dummy_size)
        
        
        instant = "%0.9E" % Time_input
        if tgpost:
            file_string = "DNS_perixz_AVERAGD_T" + instant + "_FLOW.D"
        else:
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
                               Phy_string],['uu','uv','uw','vu','vv','vw',\
                                         'wu','wv','ww']]
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
    def AVG_flow_contour(self,flow_field,PhyTime,relative=False,fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        if flow_field == 'u' or flow_field =='v' or flow_field =='w' or flow_field =='P':
            local_velo = self.flow_AVGDF.loc[PhyTime,flow_field].values
        elif flow_field == 'mag':
            index = pd.MultiIndex.from_arrays([[PhyTime,PhyTime,\
                                                PhyTime],['u','v','w']])
            local_velo = np.sqrt(np.square(self.flow_AVGDF.loc[index]).sum(axis=0)).values
            
        else:
            raise ValueError("\033[1;32 Not a valid argument")
        local_velo = local_velo.reshape(self.NCL[1],self.NCL[0]) 
        wall_velo = self._meta_data.moving_wall_calc()
        if relative and flow_field == 'u':
            for i in range(self.NCL[0]):
                local_velo[:,i] -= wall_velo[i]
        
        if not fig:
            fig, ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.subplots()
        
        x_coords = self.CoordDF['x'].dropna().values
        y_coords = self.CoordDF['y'].dropna().values
        X, Y = np.meshgrid(x_coords,y_coords)
            
        ax1 = ax.pcolormesh(X,Y,local_velo,cmap='jet')
        ax = ax1.axes
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
        cbar=fig.colorbar(ax1,ax=ax)
        if flow_field=='mag':
            cbar_label=r"$\vert\bar{U}\vert$"
        else:
            cbar_label=r"$\langle\bar{%s}\rangle$"%flow_field.upper()
        cbar.set_label(cbar_label,fontsize=12)
        ax.set_xlabel("x direction")
        ax.set_ylabel("y direction")
        #ax.axes().set_aspect('equal')
        #plt.colorbar(orientation='horizontal',shrink=0.5,pad=0.2)    
        
        return fig, ax1
    def fluct_contour_plot(self,comp,PhyTime,fig='',ax=''):
        assert(comp=='u' or comp=='v' or comp=='w')
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        comp_uu = comp+comp
        
        UU = self.UU_tensorDF.loc[PhyTime,comp_uu].values.reshape((self.NCL[1], self.NCL[0]))
        U_mean = self.flow_AVGDF.loc[PhyTime,comp].values.reshape((self.NCL[1], self.NCL[0]))
        fluct = np.sqrt(UU - U_mean*U_mean)
        if not fig:
            fig, ax = plt.subplots()
        elif not ax:
            ax = fig.add_subplot(1,1,1)
            
        x_coords = self.CoordDF['x'].dropna().values
        y_coords = self.CoordDF['y'].dropna().values
        X, Y = np.meshgrid(x_coords,y_coords)
            
        ax1 = ax.pcolormesh(X,Y,fluct,cmap='jet')
        ax = ax1.axes
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
        cbar=fig.colorbar(ax1,ax=ax)
        cbar.set_label(r"$%s^\prime_{rms}$"%comp,fontsize=12)
        ax.set_xlabel(r"$x/\delta$", fontsize=18)
        ax.set_ylabel(r"$y/\delta$",fontsize=16)
        return fig, ax
    def AVG_line_extract(self,flow_field,PhyTime,x_list,fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        #line_DF, x_coord = _line_extract(self.flow_AVGDF.loc[PhyTime,flow_field],self.CoordDF,self.NCL,x_list)
        velo_array=np.zeros((self.NCL[1]+2,self.NCL[0]))
        velo_array[1:-1,:] = self.flow_AVGDF.loc[PhyTime,flow_field].values.reshape((self.NCL[1],self.NCL[0]))    
        velo_array[0,:]=self._meta_data.moving_wall_calc()
        velo_array[-1,:]=self._meta_data.moving_wall_calc()
        #velo_array=velo_array[:,x_list]
        if not fig:
            fig, ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplots(1,1,1)

        y_coords=np.zeros(self.NCL[1]+2)
        y_coords[1:-1]=self.CoordDF['y'].dropna().values
        y_coords[0]=-1.0 ; y_coords[-1]=1.0
        x_coords = self.CoordDF['x'].dropna().values
        for x in x_list:
            ax.plot(y_coords,velo_array[:,x],label=r"$x= %.2f$" % x_coords[x])
        ax.set_ylabel(r"$\langle U/U_{b0}\rangle$",fontsize=20)
        ax.grid()
        ax.set_xlabel(r"$y/\delta$",fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
              loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
              bbox_to_anchor=(0.5,-0.2),
              fontsize=16)
        return fig, ax
    #def non_dom_calc(self,PhyTime,)
    def plot_near_wall(self,PhyTime,x_loc,U_plus=True,plotsep=True,Y_plus_lim='',fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        #wall_params = self._metaDF.loc['moving_wallflg':'VeloWall']
        wall_velo = self._meta_data.moving_wall_calc()
        
        U = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape((self.NCL[1],self.NCL[0]))
        for i in range(self.NCL[1]):
            U[i,:]=U[i,:] - wall_velo
        
        u_tau_star, delta_v_Star = wall_unit_calc(self,PhyTime)

        Y, X = np.meshgrid(self.CoordDF['y'].dropna().values,self.CoordDF['x'].dropna().values)
            
            #for i in range(3):
                #non_dom_coordsDF[i] = non_dom_coordsDF[i]*u_tau_star/np.sqrt(REN)
        if U_plus:
            Y_plus = np.zeros_like(Y)
            U_plus = np.zeros_like(U)
            for i in range(self.CoordDF['y'].dropna().size):
                for j in range(self.CoordDF['x'].dropna().size):
                    Y_plus[j,i] = (1-np.abs(Y[j,i]))/delta_v_Star[j]#*np.sqrt(REN)
                    U_plus[i,j] = U[i,j]/u_tau_star[j]
            #print(Y_plus[300],U_plus[:,300])
            if not fig:
                fig = plt.figure(figsize=[10,5])
            if not ax:
                ax = fig.add_subplot(1,1,1)
            if isinstance(x_loc,int):
                ax.plot(Y_plus[x_loc,:int(self.NCL[1]/2)],U_plus[:int(self.NCL[1]/2),x_loc])
                ax.plot(Y_plus[x_loc,:int(self.NCL[1]/2)],Y_plus[x_loc,:int(self.NCL[1]/2)],'b--')
            elif isinstance(x_loc,list):
                x_coord = []
                linestyle_list=['-','--','-.']
                for x,j in zip(x_loc,range(len(x_loc))):    
                    ax.plot(Y_plus[x,:int(self.NCL[1]/2)],U_plus[:int(self.NCL[1]/2),x],linestyle_list[j%len(linestyle_list)])
                    x_coord.append(r"$x/\delta =%.3g$" % self.CoordDF['x'].dropna().values[x])
                ax.plot(Y_plus[x_loc[0],:int(self.NCL[1]/2)],Y_plus[x_loc[0],:int(self.NCL[1]/2)],'r--')
                x_coord.append(r"$U^+=Y^+$")
                ax.legend(x_coord,loc = 'upper center',ncol=4*(len(x_coord)>3)+len(x_coord)*(len(x_coord)<4),bbox_to_anchor=(0.5,-0.2))
            else:
                raise TypeError("\033[1;32 Not a valid type")
            ax.set_xscale('log')
            ax.set_ylabel(r"$U^+$")
            ax.set_xlabel(r"$Y^+$")
            ax.set_ylim(top=1.2*np.max(U_plus),bottom=0.0)
            ax.grid()
            ax.set_title("Wall-normal velocity profile in wall units")

        else:
            U=U[:,x_loc]
            max_u=np.max(U)
            Y=Y[x_loc].T
            if Y_plus_lim:
                U=U[:int(np.ceil(Y.shape[0]/2))]
                Y=Y[:int(np.ceil(Y.shape[0]/2))]
                if Y_plus_lim:
                    y_max = Y_plus_lim*delta_v_Star[0]-1.0
                # for i in range(len(x_loc)):
                #     Y[:,i] =(1-np.abs( Y[:,i]))/delta_v_Star[x_loc[i]] #Y[:,i]/delta_v_Star[x_loc[i]]
            if plotsep:
                if not fig:
                    fig,ax = plt.subplots(1,len(x_loc),figsize=[10,5],gridspec_kw = {'wspace':0, 'hspace':0})

                #print("hello",flush=True)
                for i in range(len(x_loc)):
                    ax[i].plot(U[:,i],Y[:,i],linewidth=1)
                    ax[i].set_xlim(right=max_u)
                    ax[i].annotate("$x=%.3f$"%self.CoordDF['x'].dropna().values[x_loc[i]],
                                xy=(0.01,1.03),xytext=(0.01,1.03),textcoords='axes fraction',
                                fontsize=12)
                    ax[i].xaxis.set_ticks_position('both')
                    ax[i].xaxis.set_major_locator(mpl.ticker.FixedLocator([0,max_u]))
                    ax[i].xaxis.set_minor_locator(mpl.ticker.MaxNLocator(3))
                    if i <len(x_loc)-1:
                        ax[i].spines['right'].set_visible(False)
                    if i > 0:
                        ax[i].spines['left'].set_visible(False)
                        ax[i].set_xticklabels([])
                        ax[i].set_yticks([],[])
                    if i==0:
                        ax[i].yaxis.set_ticks_position('left')
                        if Y_plus_lim:
                            ax[i].set_ylabel(r"$Y^+$",fontsize=20)
                        else:
                            ax[i].set_ylabel(r"$y/\delta$",fontsize=20)
                    if Y_plus_lim:
                        ax[i].set_ylim(bottom=0,top=Y_plus_lim)
                    if i == int(np.ceil(len(x_loc)/2)):
                        ax[i].set_xlabel(r'$\langle U/U_{b0}\rangle$', fontsize=20)    
            else:
                if not fig:
                    fig,ax = plt.subplots(figsize=[10,5])
                elif not ax:
                    ax=fig.add_subplot(1,1,1)
                linestyles=['-','--',':','-.']
                for i in range(len(x_loc)):
                    ax.plot(Y[:,i],U[:,i],linewidth=1,linestyle=linestyles[i%len(linestyles)],
                            label=r"$x=%.3f$"%self.CoordDF['x'].dropna().values[x_loc[i]])
                ax.set_ylabel(r'$\langle U/U_{b0}\rangle$', fontsize=20)    
               
                ax.set_xlabel(r"$y/\delta$",fontsize=20)
                        
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                      loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                      bbox_to_anchor=(0.5,-0.2),
                      fontsize=16)
                ax.grid()
                if Y_plus_lim:
                        ax.set_xlim(left=-1,right=y_max)
                fig.tight_layout()
        
        
        
        
        #ax.set_yscale('log')

        return fig, ax
    def AVG_vector_plot(self,PhyTime,axis1_spacing, axis2_spacing,fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(1,1,1)
        fig, ax = _vector_plot(self.CoordDF,self.flow_AVGDF.loc[PhyTime],self.NCL,fig,ax,\
                               axis1_spacing,axis2_spacing)
        ax.set_title("Averaged velocity vector plot")
        return fig, ax
    def int_thickness_calc(self,PhyTime,relative=False):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        U0_index = int(np.floor(self.NCL[1]*0.5))
        U_mean = self.flow_AVGDF.loc[PhyTime,'u'].values.reshape((self.NCL[1],self.NCL[0]))
        if relative:
            wall_velo = self._meta_data.moving_wall_calc()
            for i in range(wall_velo.size):
                U_mean[:,i] -= wall_velo[i]
        y_coords = self.CoordDF['y'].dropna().values
        U0 = U_mean[U0_index]
        theta_integrand = np.zeros((U0_index,self.NCL[0]))
        delta_integrand = np.zeros((U0_index,self.NCL[0]))
        mom_thickness = np.zeros(self.NCL[0])
        disp_thickness = np.zeros(self.NCL[0])
        for i in range(U0_index):
            theta_integrand[i] = (np.divide(U_mean[i],U0))*(1 - np.divide(U_mean[i],U0))
            delta_integrand[i] = (1 - np.divide(U_mean[i],U0))
        for j in range(self.NCL[0]):
            mom_thickness[j] = integrate.simps(theta_integrand[:,j],y_coords[:U0_index])
            disp_thickness[j] = integrate.simps(delta_integrand[:,j],y_coords[:U0_index])
        shape_factor = np.divide(disp_thickness,mom_thickness)
        
        return disp_thickness, mom_thickness, shape_factor
    def plot_shape_factor(self,PhyTime,relative=False,fig='',ax=''):
        delta, theta, shape_factor = self.int_thickness_calc(PhyTime,relative)
        x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(1,1,1)
        ax.plot(x_coords,shape_factor)
        ax.set_xlabel(r"$x$ direction")
        ax.set_ylabel(r"Shape Factor, $H$")
        ax.grid()
        return fig, ax
    def accel_param_calc(self,PhyTime):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
       
        U_mean = self.flow_AVGDF.loc[PhyTime,'u'].values.reshape((self.NCL[1],self.NCL[0]))
        U0_index = int(np.floor(self.NCL[1]*0.5))
        U0 = U_mean[U0_index]
        wall_velo = self._meta_data.moving_wall_calc()
        x_coords = self.CoordDF['x'].dropna().values
        
        U_infty_grad = np.zeros(self.NCL[0])
        U_infty = U0 - wall_velo
        REN = self._metaDF.loc['REN'].values[0]
        for i in range(self.NCL[0]):
            if i ==0:
                U_infty_grad[i] = (U_infty[i+1] - U_infty[i])/\
                                (x_coords[i+1] - x_coords[i])
            elif i == self.NCL[0]-1:
                 U_infty_grad[i] = (U_infty[i] - U_infty[i-1])/\
                                (x_coords[i] - x_coords[i-1])
            else:
                U_infty_grad[i] = (U_infty[i+1] - U_infty[i-1])/\
                                (x_coords[i+1] - x_coords[i-1])
        accel_param = (1/(REN*U_infty**2))*U_infty_grad
        
#        if not fig:
#            fig=plt.figure()
#        if not ax:
#            ax =fig.add_subplot(1,1,1)
#        
#        ax.plot(x_coords,accel_param)
#        ax.set_xlabel(r"$x$ direction")
#        ax.set_ylabel(r"Acceleration parameter, $K$")
#        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
#        ax.grid()
        return accel_param
    
    
    def plot_Reynolds(self,comp1,comp2,x_loc,PhyTime,norm=None,Y_plus=True,fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        comp_uu =comp1 + comp2
        if comp1 == 'w' and (comp2=='v' or comp2 =='u'):
            comp_uu = comp_uu[::-1]
        elif comp1 == 'v' and comp2 =='u':
            comp_uu = comp_uu[::-1]     
                       
        UU = self.UU_tensorDF.loc[PhyTime,comp_uu].values.reshape((self.NCL[1], self.NCL[0]))[:int(self.NCL[1]/2)]
        U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape((self.NCL[1], self.NCL[0]))[:int(self.NCL[1]/2)]
        U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape((self.NCL[1], self.NCL[0]))[:int(self.NCL[1]/2)]
        #wall_params = self._metaDF.loc['moving_wallflg':'VeloWall']
        uu = UU-U1_mean*U2_mean
        if comp_uu == 'uv':
            uu *= -1.0
        
        
        y_coord = self.CoordDF['y'].dropna().values[:int(self.NCL[1]/2)]
        #Reynolds_wall_units = np.zeros_like(uu)
        if norm=='wall':
            u_tau_star, delta_v_star = wall_unit_calc(self,PhyTime)
            for i in range(self.NCL[0]):
                uu[:,i] = uu[:,i]/(u_tau_star[i]*u_tau_star[i])
        elif norm=='local-bulk':
            velo_bulk=self.bulk_velo_calc(PhyTime,relative=True)
            for i in range(self.NCL[0]):
                uu[:,i] = uu[:,i]/(velo_bulk[i]*velo_bulk[i])
        if not fig:
            fig,ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        if isinstance(x_loc,int):
            if Y_plus:
                y_coord_local = (1-np.abs(y_coord))/delta_v_star[x_loc]
            else:
                y_coord_local = y_coord
            ax.plot(y_coord_local,uu[:,x_loc])
        elif isinstance(x_loc,list):
            x_coord=[]
            linestyle_list=['-','--','-.']
            for x,j in zip(x_loc,range(len(x_loc))):
                if Y_plus:
                    y_coord_local = (1-np.abs(y_coord))/delta_v_star[x]
                else:
                    y_coord_local = y_coord
                label=r"$x/\delta =%.3g$" % self.CoordDF['x'].dropna().values[x]
                ax.plot(y_coord_local,uu[:,x],label=label,linestyle=linestyle_list[j%len(linestyle_list)])
                x_coord.append(r"$x/\delta =%.3g$" % self.CoordDF['x'].dropna().values[x])
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                  loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                  bbox_to_anchor=(0.5,-0.2),
                  fontsize=16)
        else:
            raise TypeError("\033[1;32 x_loc must be of type list or int")
        if norm=='wall':
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/u_\tau^2$"% comp_uu,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/u_\tau^2$"% comp_uu,fontsize=20)
        elif norm=='local-bulk':
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/U_b^2$"% comp_uu,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/U_b^2$"% comp_uu,fontsize=20)
        else:
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/U_{b0}^2$"% comp_uu,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"% comp_uu,fontsize=20)
        
        if Y_plus:
            ax.set_xlabel(r"$Y^+$",fontsize=20)
        else:
            ax.set_xlabel(r"$y/\delta$",fontsize=20)
        ax.grid()
        
            
        
        return fig, ax
    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        
        comp_uu =comp1 + comp2
        if comp1 == 'w' and (comp2=='v' or comp2 =='u'):
            comp_uu = comp_uu[::-1]
        elif comp1 == 'v' and comp2 =='u':
            comp_uu = comp_uu[::-1]  
        
        if y_vals_list != 'max':
            if Y_plus:
                y_index = CT.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
            else:
                y_index = CT.coord_index_calc(self.CoordDF,'y',y_vals_list)
            UU = self.UU_tensorDF.loc[PhyTime,comp_uu].values.reshape((self.NCL[1],self.NCL[0]))[y_index]
            U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape((self.NCL[1], self.NCL[0]))[y_index]
            U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape((self.NCL[1], self.NCL[0]))[y_index]
            rms_vals = UU-U1_mean*U2_mean
            
        else:
            y_index = [None]
            UU = self.UU_tensorDF.loc[PhyTime,comp_uu].values.reshape((self.NCL[1],self.NCL[0]))
            U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape((self.NCL[1], self.NCL[0]))
            U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape((self.NCL[1], self.NCL[0]))
            rms_vals = UU-U1_mean*U2_mean
            rms_vals = np.amax(rms_vals,axis=0)
        if not fig:
            fig,ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax=fig.add_subplot(1,1,1)

        x_coords = self.CoordDF['x'].dropna().values            

        if y_vals_list != 'max':
            linestyle_list=['-','--','-.']
            
            for i in range(len(y_index)):
                ax.plot(x_coords,rms_vals[i],label=r"$Y^+=%.3g$"% y_vals_list[i],linestyle=linestyle_list[i%len(linestyle_list)])
            # y_coord = [r"$Y^+=%.3g$"% x for x in y_vals_list]
            handles, labels = ax.get_legend_handles_labels()
            
            ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                      loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                      bbox_to_anchor=(0.5,-0.2),
                      fontsize=16)
            ax.set_xlabel(r"$x/\delta$",fontsize=20)
            ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%comp_uu,fontsize=20)#,fontsize=22)
            
        else:
            ax.plot(x_coords,rms_vals,label=r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%comp_uu)
            ax.set_xlabel(r"$x/\delta$",fontsize=20)
            ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"%comp_uu,fontsize=20)#,fontsize=22)
        
        # ax.set_xlabel(r"$x/\delta$",text_kwargs)
        # ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%comp_uu,text_kwargs)#,fontsize=22)
        ax.grid()
        fig.tight_layout()
        return fig, ax
    
    def plot_bulk_velocity(self,PhyTime,fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
       
        U_mean = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape((self.NCL[1],self.NCL[0]))
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(self.NCL[1]):
            U_mean[i] = U_mean[i] - wall_velo 
        x_coords = self.CoordDF['x'].dropna().values
        y_coords = self.CoordDF['y'].dropna().values
        bulk_velo = np.zeros(self.NCL[0])
        for i in range(self.NCL[0]):
            bulk_velo[i] = 0.5*integrate.simps(U_mean[:,i],y_coords)
        if not fig:
            fig, ax=plt.subplots()
        elif not ax:
            ax =fig.add_subplot(1,1,1)
        ax.plot(x_coords,bulk_velo)
        ax.set_ylabel(r"$U_b^*$",fontsize=20)
        ax.set_xlabel(r"$x/\delta$",fontsize=20)
        ax.grid()
        return fig, ax
    def plot_accel_param(self,PhyTime,fig='',ax=''):
        accel_param = self.accel_param_calc(PhyTime)
        x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            fig=plt.figure()
        if not ax:
            ax =fig.add_subplot(1,1,1)
        
        ax.plot(x_coords,accel_param)
        ax.set_xlabel(r"$x$ direction")
        ax.set_ylabel(r"Acceleration parameter, $K$")
        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        ax.grid()
        return fig,ax
    def tau_calc(self,PhyTime):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
            
        u_velo = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape((self.NCL[1],self.NCL[0]))
        ycoords = self.CoordDF['y'].dropna().values
        
        wall_velo = self._meta_data.moving_wall_calc()
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        #time0=time.time()
        #sol, h_list = CT.Stencil_calc([0,1,2,3], 1)
        #print(time.time()-time0)
        #a,b,c,d = CT.Stencil_coeffs_eval(sol,h_list,[ycoords[0]--1.0,ycoords[1]-ycoords[0],ycoords[2]-ycoords[1]])
        for i in range(self.NCL[0]):
            #tau_star[i] = mu_star*(a*wall_velo[i] + b*u_velo[0,i] + c*u_velo[1,i] + d*u_velo[2,i])
            tau_star[i] = mu_star*(u_velo[0,i]-wall_velo[i])/(ycoords[0]--1.0)#*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(0.5*ycoords[1]-1.5*(-1.0)+y_coords[0])
    
        return tau_star
    def bulk_velo_calc(self,PhyTime,relative=True):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
            
        u_velo = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape((self.NCL[1],self.NCL[0]))
        ycoords = self.CoordDF['y'].dropna().values
        wall_velo = self._meta_data.moving_wall_calc()
        
        bulk_velo=np.zeros(self.NCL[0])
        if relative:
            for i in range(self.NCL[1]):
                u_velo[i,:]=u_velo[i,:] - wall_velo
        for i in range(self.NCL[0]):
            bulk_velo[i] = 0.5*integrate.simps(u_velo[:,i],ycoords)
            
        return bulk_velo
    # def skin_friction_calc(self,PhyTime):
    #     if type(PhyTime) == float:
    #         PhyTime = "{:.9g}".format(PhyTime)
            
    #     u_velo = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape((self.NCL[1],self.NCL[0]))
    #     ycoords = self.CoordDF['y'].dropna().values
        
    #     wall_velo = self._meta_data.moving_wall_calc()
        
    #     tau_star = np.zeros_like(u_velo[1])
    #     mu_star = 1.0
    #     rho_star = 1.0
    #     REN = self._metaDF.loc['REN'].values[0]
    #     for i in range(self.NCL[0]):
    #         tau_star[i] = mu_star*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(ycoords[1]--1.0)
        
    #     bulk_velo=np.zeros(self.NCL[0])
    #     for i in range(self.NCL[1]):
    #         u_velo[i,:]=u_velo[i,:] - wall_velo
    #     for i in range(self.NCL[0]):
    #         bulk_velo[i] = 0.5*integrate.simps(u_velo[:,i],ycoords)
            
    #     skin_friction = (2.0/(rho_star*bulk_velo**2))*(1/REN)*tau_star

    #     return skin_friction
    def plot_skin_friction(self,PhyTime,fig='',ax=''):
        rho_star = 1.0
        REN = self._metaDF.loc['REN'].values[0]
        tau_star = self.tau_calc(PhyTime)
        bulk_velo = self.bulk_velo_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star
        xcoords = self.CoordDF['x'].dropna().values
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(1,1,1)
        
        ax.plot(xcoords,skin_friction)
        ax.set_xlabel(r"$x/\delta$",fontsize=20)
        ax.set_ylabel(r"$C_f$",fontsize=20)
        
        ax.grid()
        return fig, ax
    def plot_eddy_visc(self,x_loc,PhyTime='',Y_plus=True,Y_plus_max=100,fig='',ax=''):
        
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        if isinstance(x_loc,int):
            x_loc_local = [x_loc]
        elif isinstance(x_loc,list):
            x_loc_local=x_loc
        else:
            raise TypeError("\033[1;32 variable x_loc must be of type int of list of int")
        UV = self.UU_tensorDF.loc[PhyTime,'uv'].values.reshape((self.NCL[1], self.NCL[0]))
        U = self.flow_AVGDF.loc[PhyTime,'u'].values.reshape((self.NCL[1], self.NCL[0]))
        V = self.flow_AVGDF.loc[PhyTime,'v'].values.reshape((self.NCL[1], self.NCL[0]))
        uv = UV-U*V
        dUdy = self.Velo_grad_tensorDF.loc[PhyTime,'uv'].values.reshape((self.NCL[1], self.NCL[0]))
        dVdx = self.Velo_grad_tensorDF.loc[PhyTime,'vu'].values.reshape((self.NCL[1], self.NCL[0]))
        REN = self._metaDF.loc['REN'].values[0]
        nu_t = -uv*REN/(dUdy + dVdx)
        nu_t = nu_t[:,x_loc_local]
        y_coord = self._meta_data.CoordDF['y'].dropna().values
        
        
        if not fig:
            fig, ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        linestyle_list=['-','--','-.']
        x_coord = self._meta_data.CoordDF['x'].dropna().values
        
        for i in range(len(x_loc_local)):
            if Y_plus:
                avg_time = self.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                u_tau_star, delta_v_star = wall_unit_calc(self,avg_time)
                y_coord_local = y_coord[:int(y_coord.size/2)]
                
                y_coord_local = (1-np.abs(y_coord_local))/delta_v_star[x_loc_local[i]]
                
                #y_coord_local = y_coord_local[y_coord_local<Y_plus_max]
                
                nu_t_local = nu_t[:int(y_coord.size/2),i]
                
            else:
                y_coord_local = y_coord
                nu_t_local = nu_t[:,i]
            
            label = r"$x/\delta = %.2g$" %x_coord[x_loc_local[i]]
            
            ax.plot(y_coord_local,nu_t_local,label=label,
                    linestyle=linestyle_list[i%len(linestyle_list)])
            if Y_plus:
                ax.set_xlabel(r"$Y^+$",fontsize=18)
                ax.set_xlim([0,Y_plus_max])
                ax.set_ylim([-0.5,max(nu_t_local)*1.2])
            else:
                ax.set_xlabel(r"$y/\delta$",fontsize=18)
                ax.set_xlim([-1,-0.1])
                ax.set_ylim([-0.5,max(nu_t_local)*1.2])
            ax.set_ylabel(r"$\mu_t/\mu$",fontsize=16)

            handles, labels = ax.get_legend_handles_labels()
            
            ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                      loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                      bbox_to_anchor=(0.5,-0.2),
                      fontsize=16)
        return fig, ax
    def __str__(self):
        return self.flow_AVGDF.__str__()
    
                
        
class CHAPSim_peturb():
    def __init__(self,time,AVG_DF='', meta_data='',path_to_folder='',time0='',abs_path=True,tgpost=False):
        if AVG_DF:
            self.__AVGDF = AVG_DF
        else:
            assert time, "In the absence of input of class CHAPSim_AVG a time must be provided"
            if not path_to_folder:
                warnings.warn("\033[1;33No path_to_folder selected in the absence of an CHAPSim_AVG input class")
            if not time0:
                warnings.warn("\033[1;33No time0 input selected in the absence of an CHAPSim_AVG input class")
            self.__AVGDF = CHAPSim_AVG(time,meta_data,path_to_folder,time0,abs_path,tgpost)
        if not meta_data:
            meta_data = CHAPSim_meta(path_to_folder,abs_path,tgpost)
        self._meta_data = meta_data
    def tau_du_calc(self):
        tau_w = self.__AVGDF.tau_calc(self.__AVGDF.flow_AVGDF.index[0][0])
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(wall_velo.size):
            if wall_velo[i] != 0:
                index = i
                break
        
        tau_w_du = tau_w[(index+1):] - tau_w[index]
        
        
        return tau_w_du, index
    def mean_velo_peturb_calc(self,comp):
        U_velo_mean = self.__AVGDF.flow_AVGDF.loc[self.__AVGDF.flow_AVGDF.index[0][0],comp]\
                        .values.copy().reshape((self.__AVGDF.NCL[1],self.__AVGDF.NCL[0]))
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(self.__AVGDF.NCL[1]):
            U_velo_mean[i] -= wall_velo
        
        centre_index =int(0.5*self.__AVGDF.NCL[1])
        U_c0 = U_velo_mean[centre_index,0]
        mean_velo_peturb = np.zeros_like(U_velo_mean)
        for i in range(1,self.__AVGDF.NCL[0]):
            mean_velo_peturb[:,i] = (U_velo_mean[:,i]-U_velo_mean[:,0])/(U_velo_mean[centre_index,i]-U_c0)
        return mean_velo_peturb
    def rms_velo_peturb_calc(self,comp):
        velo_uu = self.__AVGDF.UU_tensorDF.loc[self.__AVGDF.UU_tensorDF.index[0][0],comp+comp]\
                        .values.reshape((self.__AVGDF.NCL[1],self.__AVGDF.NCL[0]))
        wall_velo = self._meta_data.moving_wall_calc()
        velo_mean = self.__AVGDF.flow_AVGDF.loc[self.__AVGDF.flow_AVGDF.index[0][0],comp]\
                        .values.reshape((self.__AVGDF.NCL[1],self.__AVGDF.NCL[0]))
        velo_rms = np.sqrt(velo_uu-velo_mean*velo_mean)
        
        bulk_velo = self.__AVGDF.bulk_velo_calc(self.__AVGDF.UU_tensorDF.index[0][0])
        rms_velo_peturb=np.zeros_like(velo_rms)
        for i in range(1,self.__AVGDF.NCL[0]):
            rms_velo_peturb[:,i] = (velo_rms[:,i] - velo_rms[:,0])/(bulk_velo[i]-bulk_velo[0])
        return rms_velo_peturb
    def plot_peturb_velo(self,x_indices,mode='mean',comp='u',Y_plus=False,Y_plus_max=100,fig='',ax =''):
        try:
            if mode =='mean':
                velo_peturb = self.mean_velo_peturb_calc(comp)
            elif mode == 'rms':
                velo_peturb = self.rms_velo_peturb_calc(comp)
            else:
                raise ValueError("\033[1;32 mode must be equal to 'mean' or 'rms'")
        except UnboundLocalError:
            warnings.warn("\033[1;33This function can only be used if moving wall is used, ignoring")
            return None, None
        if not fig:
            fig, ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        y_coord = self._meta_data.CoordDF['y'].dropna().values
        x_coord = self._meta_data.CoordDF['x'].dropna().values
        avg_time = self.__AVGDF.flow_AVGDF.index[0][0]
        u_tau_star, delta_v_star = wall_unit_calc(self.__AVGDF,avg_time)
        
        if Y_plus:
            #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
            
            y_coord = y_coord[:int(y_coord.size/2)]
            y_coord = (1-np.abs(y_coord))/delta_v_star[0]
            
            #y_coord = y_coord[y_coord<Y_plus_max]
            velo_peturb = velo_peturb[:int(y_coord.size)]
        else:
            y_max= Y_plus_max*delta_v_star[0]-1.0
        linestyle_list=['-','--','-.']
        for x,j in zip(x_indices,range(len(x_indices))):
            label=r"$x/\delta = %.2f$" % x_coord[x_indices[j]]
            ax.plot(velo_peturb[:,x],y_coord,label=label,
                    linestyle=linestyle_list[j%len(linestyle_list)])
        if mode =='mean':
            ax.set_xlabel(r"$\bar{U}^{\wedge}$",fontsize=18)
        elif mode =='rms':
            ax.set_xlabel(r"${%s\prime}_{rms}^{\wedge}$" % comp,fontsize=18)
        if Y_plus:
            ax.set_ylabel(r"$Y^+$",fontsize=16)
            ax.set_ylim([0,Y_plus_max])
        else:
            ax.set_ylabel(r"$y/\delta$",fontsize=16)
            ax.set_ylim([-1,y_max])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                    loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                    bbox_to_anchor=(0.5,-0.2),
                    fontsize=16)
        fig.tight_layout()
        return fig, ax
    def plot_peturb_cf(self,wall_units=False,fig='',ax=''):
        try:
            tau_du = self.tau_du_calc()
        except UnboundLocalError:
            warnings.warn("\033[1;33This function can only be used if moving wall is used, ignoring")
            return None, None
        avg_time = self.__AVGDF.flow_AVGDF.index[0][0]
        bulkvelo = self.__AVGDF.bulk_velo_calc(avg_time)
        REN = self._meta_data.metaDF.loc['REN'].values[0]
        
        rho_star = 1.0
        if wall_units:
            u_tau_star, delta_v_star = wall_unit_calc(self.__AVGDF,avg_time)
            Cf_du = tau_du/(0.5*REN*rho_star*u_tau_star[0]\
                                        *(bulkvelo[(1):]-bulkvelo[0]))
        else:
            Cf_du = tau_du/(0.5*REN*rho_star*(bulkvelo[1:]-bulkvelo[0])**2)
        x_coord = self._meta_data.CoordDF['x'].dropna().values[1:] 
        
        if not fig:
            fig, ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplot(1,1,1)
            
        ax.plot(x_coord, Cf_du)
        ax.set_xlabel(r"$x/\delta$",fontsize=18)
        if wall_units:
            ax.set_ylabel(r"$C\prime_{f,du}$",fontsize=16)
        else:
            ax.set_ylabel(r"$C_{f,du}$",fontsize=16)
        ax.set_ylim([0,2*Cf_du[-1]])
        ax.grid()
        return fig, ax
    
class CHAPSim_fluct():
    def __init__(self,*args,**kwargs):
        i=0 
        for arg in args:
           if isinstance(arg,CHAPSim_Inst):
               inst_data = arg ; i+=1
               
        # assert i <2 and j<2 , "Only one arg of CHAPSim_Inst (%d) or CHAPSim_AVG (%d)" %(i,j)+\
        #                         " can be provided "
        file_names = time_extract(kwargs.get('path_to_folder',''),kwargs.get('abs_path',True))
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        times = list(dict.fromkeys(time_list))
        maxtime= np.amax(times)
        avg_list = [arg for arg in args if isinstance(arg,CHAPSim_AVG)]
        if len(avg_list)>0:
            j=1
            for avg in avg_list:
                avg_times=list(set([float(x[0]) for x in avg.flow_AVGDF.index]))
                max_avg_times=max(avg_times)
                if max_avg_times == maxtime:
                    avg_data=avg
                    break
                else:
                    if j == len(avg_list):
                        avg_args = [x for x in args if not isinstance(x,CHAPSim_AVG) and not isinstance(x,CHAPSim_Inst)]
                        CHAPSim_fluct.__check_args(avg_args,True)
                        try:
                            avg_data=CHAPSim_AVG(maxtime,*avg_args[1:],**kwargs)
                        except Exception:
                            times.remove(maxtime)
                            maxtime = max(times)
                            avg_data=CHAPSim_AVG(maxtime,*avg_args[1:],**kwargs)
                j+=1
        else:
            avg_args = [x for x in args if not isinstance(x,CHAPSim_Inst)]
            CHAPSim_fluct.__check_args(avg_args,True)
            avg_data = CHAPSim_AVG(maxtime,*avg_args[1:],**kwargs)
        avg_values=avg_data.flow_AVGDF.values

        args = [x for x in args if not isinstance(x,CHAPSim_AVG)]
        if i==0:
            kwargs.pop('time0')
            inst_data=CHAPSim_Inst(*args,**kwargs)
            pd.testing.assert_frame_equal(left=avg_data._metaDF,right=inst_data._meta_data.metaDF)
            inst_values=inst_data.InstDF.values
            inst_times=list(set([x[0] for x in inst_data.InstDF.index]))
        else:
            inst_data_list=[arg for arg in args if isinstance(arg,CHAPSim_Inst)]
            for inst_data in inst_data_list:
                pd.testing.assert_frame_equal(left=avg_data._metaDF,
                                            right=inst_data._meta_data.metaDF)

            inst_values_list=[inst.InstDF.values for inst in inst_data_list]
            # print([y[0] for y in itertools.chain.from_iterable([inst.InstDF.index for inst in inst_data_list])])
            inst_times=list(set([y[0] for y in itertools.chain.from_iterable(\
                            [inst.InstDF.index for inst in inst_data_list])]))
            inst_values=np.vstack(inst_values_list)
            
            
            args = [x for x in args if not isinstance(x,CHAPSim_Inst)]
            CHAPSim_fluct.__check_args(args,False)
            try:
                if args:
                    args[0] = [x for x in args[0] if str(x) not in inst_times and x in times]
                    if 'time0' in kwargs: del kwargs['time0']
                    if args[0]:
                        inst_data = CHAPSim_Inst(*args,**kwargs)
                        inst_values=np.vstack((inst_data.InstDF.values,*inst_values))
                        inst_times.extend(list(set([x[0] for x in inst_data.InstDF.index])))
            except IndexError as e:
                raise ValueError("\033[1;32 Check the arguments list for this function it should match CHAPSim_AVG with the options"\
                                +" of addition CHAPSim_AVG or CHAPSim_Inst")
        # inst_times = list(set([x[0] for x in inst_data.InstDF.index]))
        # assert inst_times ==avg_times, "The times in CHAPSim_Inst and CHAPSim_AVG must be the same"
        self._meta_data=avg_data._meta_data 
        self.CoordDF=self._meta_data.CoordDF
        self.NCL=self._meta_data.NCL
        
        # inst_velo = inst_data.InstDF.values\
        #              .reshape((4*len(inst_times),self.NCL[2],self.NCL[1],self.NCL[0]))
        # avg_velo = avg_data.flow_AVGDF.values\
        #              .reshape((4,self.NCL[1],self.NCL[0]))

        inst_values=inst_values.reshape((len(inst_times),4,self.NCL[2],self.NCL[1],self.NCL[0]))
        avg_values=avg_values.reshape(4,self.NCL[1],self.NCL[0])
        fluct_velo=np.zeros_like(inst_values)
        for j in range(len(inst_times)):
            for i in range(self.NCL[2]):
                fluct_velo[j,:,i] = inst_values[j,:,i]-avg_values
             
        fluct_velo=fluct_velo.reshape((4*len(inst_times),self.NCL[2]*self.NCL[1]*self.NCL[0]))
        index=[[],[]]

        for times in inst_times:
            index[0].extend([times]*4)
            index[1].extend(['u','v','w','p'])
        self.fluctDF = pd.DataFrame(fluct_velo,index=pd.MultiIndex.from_arrays(index))
    def plot_contour(self,comp,y_vals,PhyTime='',x_split_list='',fig='',ax=''):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time))
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        if isinstance(y_vals,int):
            y_vals=[y_vals]
        elif not isinstance(y_vals,list):
            raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
        x_coords = self.CoordDF['x'].dropna().values
        z_coords = self.CoordDF['z'].dropna().values
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF.loc[PhyTime,comp].values\
                .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))[:,y_vals,:]
        
        #fluct=fluct.reshape((self.NCL[2],self.NCL[0]))
        if isinstance(y_vals,int):
            y_vals=[y_vals]
        elif not isinstance(y_vals,list):
            raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
        
        if not x_split_list:
                x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        if not fig:
            fig, ax = plt.subplots(len(x_split_list)-1,len(y_vals),
                                    figsize=[10*len(y_vals),3*(len(x_split_list)-1)],
                                    squeeze=False)
        elif not ax:
            ax=fig.subplots(len(x_split_list)-1,len(y_vals),squeeze=False)
        ax=ax.flatten()
        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)
        X, Z = np.meshgrid(x_coords, z_coords)
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        if len(x_split_list)==2:
            for i in range(len(y_vals)):
                # print(X.shape,Z.shape,fluct.shape)
                ax1 = ax[i].pcolormesh(X[:,x_coords_split[0]:x_coords_split[1]],
                                        Z[:,x_coords_split[0]:x_coords_split[1]],
                                        fluct[:,i,x_coords_split[0]:x_coords_split[1]],
                                        cmap='jet')
                ax2 = ax1.axes
                ax1.set_clim(min_val,max_val)
                cbar=fig.colorbar(ax1,ax=ax[i])
                cbar.set_label(r"$%s^\prime$"%comp,fontsize=12)
                ax2.set_xlabel(r"$%s/\delta$" % 'x',fontsize=20)
                ax2.set_ylabel(r"$%s/\delta$" % 'z',fontsize=20)
                ax[i]=ax1
        else:
            ax=ax.flatten()
            max_val = np.amax(fluct); min_val=np.amin(fluct)
            for j in range(len(x_split_list)-1):
                for i in range(len(y_vals)):
                    # print(X.shape,Z.shape,fluct.shape)
                    ax1 = ax[j*len(y_vals)+i].pcolormesh(X[:,x_coords_split[j]:x_coords_split[j+1]],
                                             Z[:,x_coords_split[j]:x_coords_split[j+1]],
                                             fluct[:,i,x_coords_split[j]:x_coords_split[j+1]],
                                             cmap='jet')
                    ax2 = ax1.axes
                    ax1.set_clim(min_val,max_val)
                    cbar=fig.colorbar(ax1,ax=ax[j*len(y_vals)+i])
                    cbar.set_label(r"$%s^\prime$"%comp,fontsize=12)
                    ax2.set_xlabel(r"$%s/\delta$" % 'x',fontsize=20)
                    ax2.set_ylabel(r"$%s/\delta$" % 'z',fontsize=20)
                    ax[j*len(y_vals)+i]=ax1
                    ax[j*len(y_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()

        return fig, ax
    def plot_streaks(self,comp,vals_list,PhyTime='',ylim='',Y_plus=True,cmap='Greens_r',plotter=''):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time))
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        x_coords = self._meta_data.Coord_ND_DF['x'].dropna().values
        y_coords = self._meta_data.Coord_ND_DF['y'].dropna().values
        z_coords = self._meta_data.Coord_ND_DF['z'].dropna().values
        fluct = self.fluctDF.loc[PhyTime,comp].values\
                .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))
                
        Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)
        grid = pv.StructuredGrid(X,Z,Y)
        grid.cell_arrays['fluct']=fluct.flatten()
        pgrid=grid.cell_data_to_point_data()
        if isinstance(vals_list, int) or isinstance(vals_list, float):
            vals_list=[vals_list]
        elif not isinstance(vals_list, list):
            raise TypeError("\033[1;32 Argument `vals_list' must be type int, float, or list ")
        cmp_list=['Greens_r','Blues','Reds_r']
        if cmap != 'Greens_r':
            cmp_list.insert(0,cmap)
        for val,i in zip(vals_list,range(len(vals_list))):
            contour=pgrid.contour(isosurfaces=1,scalars='fluct',preference='point',rng=(val,val))
            #grid.set_active_scalars('lambda_2')
            if not plotter:
                plotter = pv.BackgroundPlotter(notebook=False)
                plotter.set_background('white')
                plotter.show_bounds(color='black')
            
            plotter.add_mesh(contour,interpolate_before_map=True,
                             cmap=cmp_list[i%len(cmp_list)])
            plotter.remove_scalar_bar()
        
        return plotter
    def plot_fluct3D_xz(self,y_vals,comp,PhyTime='',x_split_list='',fig='',ax=''):
        # def axisEqual3D(ax):
        #     extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        #     sz = extents[:,1] - extents[:,0]
        #     centers = np.mean(extents, axis=1)
        #     maxsize = max(abs(sz))
        #     r = maxsize/2
        #     for ctr, dim in zip(centers, 'xyz'):
        #         getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time))
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        if isinstance(y_vals,int):
            y_vals=[y_vals]
        elif not isinstance(y_vals,list):
            raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
            
        x_coords = self.CoordDF['x'].dropna().values
        z_coords = self.CoordDF['z'].dropna().values
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF.loc[PhyTime,comp].values\
                .reshape((self.NCL[2],self.NCL[1],self.NCL[0]))[:,y_vals,:]
        
        if not x_split_list:
            x_split_list=[np.min(x_coords),np.max(x_coords)]
            
        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)
          
        
        if not fig:
            subplot_kw={'projection':'3d'}
            fig, ax = plt.subplots((len(x_split_list)-1),len(y_vals),figsize=[10*len(y_vals),5*(len(x_split_list)-1)],subplot_kw=subplot_kw,squeeze=False)
        elif not ax:
            ax_list=[]
            for i in range(len(x_split_list)-1):
                for j in range(len(y_vals)):
                    ax_list.append(fig.add_subplot(len(x_split_list)-1,len(y_vals),i*len(y_vals)+j+1,projection='3d'))
            ax=np.array(ax_list)
            #raise Exception("axis must be provided with figure")
        #     ax=np.empty(len(y_vals)*(len(x_split_list)-1),dtype='Axes3DSubplot')
        # else:
        #     if isinstance(ax,np.ndarray):
        #         assert ax.dtype==mpl.Axes.axes, "Axes array provided must be of type matplotlib.axes\n"
        #         assert ax.size==len(y_vals)*(len(x_split_list)-1), "Axes array provided must be the correct size\n"
        #     elif isinstance(ax,mpl.Axes.axes):
        #         assert len(y_vals)*(len(x_split_list)-1)==1,"If Axes provided the total number of plots to be processed must be 1\n"
        ax=ax.flatten()
        for j in range(len(x_split_list)-1):
            for i in range(len(y_vals)):
                max_val = np.amax(fluct[:,i,:]); min_val=np.amin(fluct[:,i,:])
                
                #ax[j*len(y_vals)+i] = fig.add_subplot(j+1,i+1,j*len(y_vals)+i+1,projection='3d')
                #axisEqual3D(ax)
                surf=ax[j*len(y_vals)+i].plot_surface(Z[:,x_coords_split[j]:x_coords_split[j+1]],
                                     X[:,x_coords_split[j]:x_coords_split[j+1]],
                                     fluct[:,i,x_coords_split[j]:x_coords_split[j+1]],
                                     rstride=1, cstride=1, cmap='jet',
                                            linewidth=0, antialiased=False)
                ax[j*len(y_vals)+i].set_ylabel(r'$x/\delta$',fontsize=20)
                ax[j*len(y_vals)+i].set_xlabel(r'$z/\delta$',fontsize=20)
                ax[j*len(y_vals)+i].set_zlabel(r'$%s^\prime$'%comp,fontsize=20)
                ax[j*len(y_vals)+i].set_zlim(min_val,max_val)
                surf.set_clim(min_val,max_val)
                cbar=fig.colorbar(surf,ax=ax[j*len(y_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp,fontsize=12)
                #ax_list.append(ax)
                ax[j*len(y_vals)+i]=surf
        fig.tight_layout()
        return fig, ax
    @staticmethod
    def __check_args(args,avg):
        args_type_list_avg=[float,CHAPSim_meta,str,
                        float,bool,bool]
        args_type_list_inst=[float,CHAPSim_meta,str,
                        bool,bool]
        if avg: 
            type_list = args_type_list_avg
        else: 
            type_list = args_type_list_inst
        for arg,type in zip(args,type_list):
            print(arg,type,isinstance(arg,type))
            if not isinstance(arg,type):
                if arg==args[0] and not isinstance(arg,list):
                    
                    raise TypeError("\033[1;32 Argument for %s invalid check input positional arguments" \
                                    %(CHAPSim_AVG.__name__ if avg else CHAPSim_Inst.__name__))
                                
                else:
                    raise TypeError("\033[1;32 Argument for %s invalid check input positional arguments" \
                                    %(CHAPSim_AVG.__name__ if avg else CHAPSim_Inst.__name__))
    @staticmethod
    def create_video(y_vals,comp,contour=True,meta_data='',path_to_folder='',time0='',
                            abs_path=True,tgpost=False,x_split_list='',lim_min=-1.,lim_max=1,
                            fig='',ax=''):
        file_names= time_extract(path_to_folder,abs_path)
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        times = list(dict.fromkeys(time_list))
        if time0:
            times = list(filter(lambda x: x > time0, times))
        #print(times.sort())
        times.sort()
        max_time = np.amax(times)
        avg_data=CHAPSim_AVG(max_time,meta_data,path_to_folder,time0,abs_path,tgpost)
        if isinstance(y_vals,int):
            y_vals=[y_vals]
        elif not isinstance(y_vals,list):
            raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
        
        x_coords = avg_data.CoordDF['x'].dropna().values
        if not x_split_list:
            x_split_list=[np.min(x_coords),np.max(x_coords)]
        if not fig:
            fig = plt.figure(figsize=[7*len(y_vals),3*(len(x_split_list)-1)])
        fig_list = [fig]*len(times)

        
        frames=list(zip(times,fig_list))
        def animate(frames):
            time=frames[0]; fig=frames[1]
            ax_list=fig.axes
            for ax in ax_list:
                ax.remove()
            fluct_data = CHAPSim_fluct(avg_data,time,meta_data=meta_data,path_to_folder=path_to_folder,
                                    time0=time0,abs_path=abs_path)

            if contour:
                fig, ax = fluct_data.plot_contour(comp,y_vals,time,x_split_list=x_split_list,fig=fig)
            else:
                fig,ax = fluct_data.plot_fluct3D_xz(y_vals,comp,time,x_split_list,fig)
            ax[0].axes.set_title(r"$t^*=%.3f$"%time)

            for im in ax:
                im.set_clim(lim_min,lim_max)

            fig.tight_layout()
            return ax
        anim = mpl.animation.FuncAnimation(fig,animate,frames=frames)
        
        # print("\033[1;32m The colorbar limits are set to (%.3f,%.3f) the maximum limits are (%.3f,%.3f)\033[0;37m\n"%(lim_min,lim_max,lims[0],lims[1]))
        # del lims
        return anim

        
    def __str__(self):
        return self.fluctDF.__str__()
                
        
        
        
class CHAPSim_meta():
    def __init__(self,path_to_folder='',abs_path=True,tgpost=False):
        self.CoordDF, self.NCL = self._coord_extract(path_to_folder,abs_path,tgpost)
        self.Coord_ND_DF = self.Coord_ND_extract(path_to_folder,abs_path,tgpost)
        self.metaDF = self._meta_extract(path_to_folder,abs_path)
        self.path_to_folder = path_to_folder
        self._abs_path = abs_path
    def _meta_extract(self,path_to_folder,abs_path):
        
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
                
    def _coord_extract(self,path_to_folder,abs_path,tgpost):
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
        
        if tgpost:
            XND = x_coord[:-1]
        else:
            for i in range(x_size):
                if x_coord[i] == 0.0:
                    index=i
                    break
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
    def Coord_ND_extract(self,path_to_folder,abs_path,tgpost):
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
        
        if tgpost:
            XND = x_coord[:-1]
        else:
            for i in range(x_size):
                if x_coord[i] == 0.0:
                    index=i
                    break
            XND = x_coord[index+1:]#np.delete(x_coord,np.arange(index+1))
        
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        YND=np.loadtxt(file,comments='#',usecols=1)
        YND=YND[:self.NCL[1]+1]
        
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
    def moving_wall_calc(self):
        wall_velo = np.zeros(self.NCL[0])
        if int(self.metaDF.loc['moving_wallflg',0]) == 1:
            if not self._abs_path:
                file_path = os.path.abspath(os.path.join(self.path_to_folder,'CHK_MOVING_WALL.dat'))
            else:
                file_path = os.path.join(self.path_to_folder,'CHK_MOVING_WALL.dat')
            
            mw_file = open(file_path)
            wall_velo_ND = np.loadtxt(mw_file,comments='#',usecols=1)
            for i in range(1,self.NCL[0]+1):
                wall_velo[i-1] = 0.5*(wall_velo_ND[i+1] + wall_velo_ND[i])

        return wall_velo
    def __str__(self):
        return self.metaDF.__str__()
class CHAPSim_budget():
    
    def __init__(self,time,AVG_DF):
        self.AVG_DF = AVG_DF
        self.budgetDF = None
        self.comp = None
    def _budget_extract(self,PhyTime,comp1,comp2):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
            
        production = self.__production_extract(PhyTime,comp1,comp2)
        advection = self.__advection_extract(PhyTime,comp1,comp2)
        turb_transport = self.__turb_transport(PhyTime,comp1,comp2)
        pressure_diffusion = self.__pressure_diffusion(PhyTime,comp1,comp2)
        pressure_strain = self.__pressure_strain(PhyTime,comp1,comp2)
        viscous_diff = self.__viscous_diff(PhyTime,comp1,comp2)
        dissipation = self.__dissipation_extract(PhyTime,comp1,comp2)
        array_concat = [production,advection,turb_transport,pressure_diffusion,\
                        pressure_strain,viscous_diff,dissipation]
        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['production','advection','turbulent transport','pressure diffusion',\
                     'pressure strain','viscous diffusion','dissipation']  
        phystring_index = []
        for i in range(7):
            phystring_index.append(PhyTime)
        
        budgetDF = pd.DataFrame(budget_array,index = pd.MultiIndex.\
                                from_arrays([phystring_index,budget_index]))
        
        return budgetDF
    def __advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 
        
        flow_mean1 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean2 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        U_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'u']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        V_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'v']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        UU_tensor = self.AVG_DF.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
               
        uu = UU_tensor - flow_mean1*flow_mean2
    
        uu_dx = Grad_calc(self.AVG_DF.CoordDF,uu,'x')
        uu_dy = Grad_calc(self.AVG_DF.CoordDF,uu,'y')
        
        advection = -(U_mean*uu_dx + V_mean*uu_dy)
        return advection.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def __turb_transport(self,PhyTime,comp1,comp2):
        U1U2U_comp = comp1 + comp2 + 'u'
        
        if U1U2U_comp == 'wwu' or U1U2U_comp == 'vvu':
            U1U2U_comp = U1U2U_comp[::-1]
        elif U1U2U_comp =='uvu':
            U1U2U_comp = 'uuv'
            
        U1U2V_comp = comp1 + comp2 + 'v'
        if U1U2V_comp == 'wwv':
            U1U2V_comp = U1U2V_comp[::-1]
            
        u1u_comp = comp1 + 'u'
        if u1u_comp == 'vu' or u1u_comp == 'wu':
            u1u_comp = u1u_comp[::-1]
        elif u1u_comp == 'wv':
            u1u_comp = u1u_comp[::-1]
            
        u1v_comp = comp1 + 'v'
        if u1v_comp == 'wv':
            u1v_comp = u1v_comp[::-1]
            
        u2u_comp = comp2 + 'u'
        if u2u_comp == 'vu' or u2u_comp == 'wu':
            u2u_comp = u2u_comp[::-1]
        elif u2u_comp == 'wv':
            u2u_comp = u2u_comp[::-1]
            
        u2v_comp = comp2 + 'v'
        if u2v_comp == 'wv':
            u2v_comp = u2v_comp[::-1]
        u1u2_comp = comp1 + comp2
        
        U1U = self.AVG_DF.UU_tensorDF.loc[PhyTime,u1u_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U1V = self.AVG_DF.UU_tensorDF.loc[PhyTime,u1v_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U2U = self.AVG_DF.UU_tensorDF.loc[PhyTime,u2u_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))                
        U2V = self.AVG_DF.UU_tensorDF.loc[PhyTime,u2v_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0])) 
        U1U2 = self.AVG_DF.UU_tensorDF.loc[PhyTime,u1u2_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))   
                
        U1U2U = self.AVG_DF.UUU_tensorDF.loc[PhyTime,U1U2U_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U1U2V = self.AVG_DF.UUU_tensorDF.loc[PhyTime,U1U2V_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        flow_mean1 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean2 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'u']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        V_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'v']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        u1u = U1U - flow_mean1*U_mean
        u1v = U1V - flow_mean1*V_mean
        u2u = U2U - flow_mean2*U_mean
        u2v = U2V - flow_mean2*V_mean
        u1u2 = U1U2 - flow_mean1*flow_mean2
        
        u1u2u = U1U2U - flow_mean1*flow_mean2*U_mean - flow_mean1*u2u - flow_mean2*u1u - U_mean*u1u2
        u1u2v = U1U2V - flow_mean1*flow_mean2*V_mean - flow_mean1*u2v - flow_mean2*u1v - V_mean*u1u2
        
        u1u2u_dx = Grad_calc(self.AVG_DF.CoordDF,u1u2u,'x')
        u1u2v_dy = Grad_calc(self.AVG_DF.CoordDF,u1u2v,'y')
        
        turb_transport = -(u1u2u_dx + u1u2v_dy)
        return turb_transport.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def __pressure_strain(self,PhyTime,comp1,comp2):
        u1u2 = comp1 + comp2
        u2u1 = comp2 + comp1
        
        rho_star = 1.0
        Pdu1dx2_mean = self.AVG_DF.PR_Velo_grad_tensorDF.loc[PhyTime,u1u2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        Pdu2dx1_mean = self.AVG_DF.PR_Velo_grad_tensorDF.loc[PhyTime,u2u1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        du1dx2_mean = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,u1u2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        du2dx1_mean = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,u2u1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        P_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'P']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
                        
        pdu1dx2 = Pdu1dx2_mean - P_mean*du1dx2_mean
        pdu2dx1 = Pdu2dx1_mean - P_mean*du2dx1_mean
        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def __pressure_diffusion(self,PhyTime,comp1,comp2):
        if comp1 == 'u' and comp2 =='u':
            diff1 = diff2 = 'x'
        elif comp1 == 'v' and comp2 =='v':
            diff1 = diff2 = 'y'
        elif comp1 =='u' and comp2 =='v':
            diff1 = 'y'
            diff2 = 'x'
        elif comp1 == 'w' and comp2 == 'w':
            pressure_diff = np.zeros((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
            return pressure_diff.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
        else:
            raise ValueError
        PU1 = self.AVG_DF.PU_vectorDF.loc[PhyTime,comp1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        PU2 = self.AVG_DF.PU_vectorDF.loc[PhyTime,comp2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        P_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'P']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean1 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean2 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        pu1 = PU1 - P_mean*flow_mean1
        pu2 = PU2 - P_mean*flow_mean2
        
        rho_star = 1.0
        pu1_grad = Grad_calc(self.AVG_DF.CoordDF,pu1,diff1)
        pu2_grad = Grad_calc(self.AVG_DF.CoordDF,pu2,diff2)
        
        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def __viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        UU = self.AVG_DF.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean1 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean2 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
                        
        u1u2 = UU - flow_mean1*flow_mean2
        REN = self.AVG_DF._metaDF.loc['REN'].values[0]
        viscous_diff = (1/REN)*_scalar_laplacian(self.AVG_DF.CoordDF,u1u2)
        return viscous_diff.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def __production_extract(self,PhyTime,comp1,comp2):
        U1U_comp = comp1 + 'u'
        U2U_comp = comp2 + 'u'
        if U1U_comp == 'vu' or U1U_comp == 'wu':
            U1U_comp = U1U_comp[::-1]
        if U2U_comp == 'vu' or U2U_comp == 'wu':
            U2U_comp = U2U_comp[::-1]
        
        U1V_comp = comp1 + 'v'
        U2V_comp = comp2 + 'v'
        
        if U1V_comp == 'wv':
            U1V_comp = U1V_comp[::-1]
        if U2V_comp == 'wv':
            U2V_comp = U2V_comp[::-1]
        
        U1U = self.AVG_DF.UU_tensorDF.loc[PhyTime,U1U_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U2U = self.AVG_DF.UU_tensorDF.loc[PhyTime,U2U_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U1V = self.AVG_DF.UU_tensorDF.loc[PhyTime,U1V_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U2V = self.AVG_DF.UU_tensorDF.loc[PhyTime,U2V_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        flow_mean1 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp1]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        flow_mean2 = self.AVG_DF.flow_AVGDF.loc[PhyTime,comp2]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        U_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'u']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        V_mean = self.AVG_DF.flow_AVGDF.loc[PhyTime,'v']\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        u1u = U1U  - flow_mean1*U_mean
        u2u = U2U  - flow_mean2*U_mean
        u1v = U1V  - flow_mean1*V_mean
        u2v = U2V  - flow_mean2*V_mean
        
        U1x_comp = comp1 + 'u'
        U2x_comp = comp2 + 'u'
        U1y_comp = comp1 + 'v'
        U2y_comp = comp2 + 'v'
        
        du1dx = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,U1x_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        du2dx = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,U2x_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        du1dy = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,U1y_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        du2dy = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,U2y_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def __dissipation_extract(self,PhyTime,comp1,comp2):
        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        dU1dx_comp = comp1 + 'u'
        dU2dx_comp = comp2 + 'u'
        dU1dy_comp = comp1 + 'v'
        dU2dy_comp = comp2 + 'v'
        
        dU1dxdU2dx = self.AVG_DF.DUDX2_tensorDF.loc[PhyTime,dU1dxdU2dx_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        dU1dydU2dy = self.AVG_DF.DUDX2_tensorDF.loc[PhyTime,dU1dydU2dy_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        dU1dx_mean = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,dU1dx_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        dU2dx_mean = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,dU2dx_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        dU1dy_mean = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,dU1dy_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        dU2dy_mean = self.AVG_DF.Velo_grad_tensorDF.loc[PhyTime,dU2dy_comp]\
                        .values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
        
        REN = self.AVG_DF._metaDF.loc['REN'].values[0]
        du1dxdu2dx = dU1dxdU2dx - dU1dx_mean*dU2dx_mean
        du1dydu2dy = dU1dydU2dy - dU1dy_mean*dU2dy_mean
        
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation.reshape(self.AVG_DF.NCL[0]*self.AVG_DF.NCL[1])
    def budget_plot(self,PhyTime, x_list,wall_units=True,plotx0=True, fig='', ax =''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        #wall_params = self.AVG_DF._metaDF.loc['moving_wallflg':'VeloWall']
        u_tau_star, delta_v_star = wall_unit_calc(self.AVG_DF,PhyTime)
        budget_scale = u_tau_star**3/delta_v_star
        x_coords = self.AVG_DF.CoordDF['x'].dropna().values
        
        Y, X = np.meshgrid(self.AVG_DF.CoordDF['y'].dropna().values,self.AVG_DF.CoordDF['x'].dropna().values)
        
        Y_coords = np.zeros_like(Y)
        if wall_units:
            for i in range(self.AVG_DF.CoordDF['x'].dropna().size):
                Y_coords[i,:] = (1-np.abs(Y[i,:]))/delta_v_star[i]
        else:
            Y_coords = (1-np.abs(Y))
        
        if isinstance(x_list,list):
            ax_size = len(x_list)
        else:
            ax_size = 1
        if not fig:
            if ax_size>4:
                ax_size=(int(np.ceil(ax_size/2)),2)
                #print(type(ax_size[0]),type(ax_size[1]))
                fig, ax = plt.subplots(ax_size[0],ax_size[1],figsize=[6.5*ax_size[1],4*ax_size[0]+1])
                
            else:
                fig, ax = plt.subplots(ax_size,figsize=[6,3*ax_size])
        elif not ax:
            ax=[]
            for i in range(1,ax_size +1):
                ax.append(fig.add_subplot(ax_size,1,i))
            
        comp_list = tuple([x[1] for x in self.budgetDF.index])#('production','advection','turbulent_transport','pressure_diffusion',\
                     #'pressure_strain','viscous_diffusion','dissipation')   
        #print(comp_list)
        k=0
        def ax_convert(j):
            if type(ax_size)==int:
                return j
            else:
                return (int(j/2),j%2)
        
                
        Y_extent= int(np.floor(self.AVG_DF.NCL[1]/2))
        linestyle_list=['-','--','-.']
        markerstyle=['v','x','+','o','s','^','D']
        if (isinstance(ax,list) or isinstance(ax,np.ndarray)) and ax_size!=1:
            for x in x_list:
                for comp, j in zip(comp_list,range(len(comp_list))):
                    budget_values = self.budgetDF.loc[PhyTime,comp].copy().values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
                    if wall_units:
                        budget = budget_values[:Y_extent,x]/budget_scale[x]
                    else:
                        budget = budget_values[:Y_extent,x]
                    if x ==x_list[0] and plotx0:
                        budget0=self.budgetDF.loc[PhyTime].copy().values.reshape((7,self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
                    if self.comp == 'uv':
                        budget= budget * -1.0
                    i=ax_convert(k)
                    ax[i].plot(Y_coords[x,:Y_extent],budget,linestyle_list[j%len(linestyle_list)],label=comp)
                    if plotx0 and x != x_list[0]:
                        ax[i].plot(Y_coords[x_list[0],:Y_extent],budget0[j,:Y_extent,x_list[0]],marker=markerstyle[j],markevery=10,
                                   color='k',linestyle='',label=comp+" x=%.3f"%x_coords[x_list[0]])
                    
                    if wall_units:
                        if self.comp == 'uv':
                            ax[i].set_ylabel(r'$-\overline{%s} \cdot \nu/u_\tau^4$'%self.comp,fontsize=16)
                        elif self.comp == 'k':
                            ax[i].set_ylabel(r'$%s \cdot \nu/u_\tau^4$'%self.comp,fontsize=16)
                        else:
                            ax[i].set_ylabel(r'$\overline{%s} \cdot \nu/u_\tau^4$'%self.comp,fontsize=16)
                        ax[i].set_xscale('log')
                        ax[i].set_xlim(left=1.0)
                       # if i == len(x_list) -1:
                        ax[i].set_xlabel(r"$Y^+$",fontsize=18)
                    else:
                        if self.comp == 'uv':
                            ax[i].set_ylabel(r'$-\overline{%s}/U_{b0}^2$'%self.comp,fontsize=16)
                        elif self.comp == 'k':
                            ax[i].set_ylabel(r'$%s/U_{b0}^2$'%self.comp,fontsize=16)
                        else:
                            ax[i].set_ylabel(r'$\overline{%s}/U_{b0}^2$'%self.comp,fontsize=16)
                        #if i == len(x_list) -1:
                        ax[i].set_xlabel(r"$y/\delta$",fontsize=18)
                    ax[i].set_title(r"$x/\delta=%.3f$" %x_coords[x],loc='right',pad=-6.0)
                    
                    ax[i].grid()
                    
                    if k ==len(x_list)-1:
                        handles, labels = ax[i].get_legend_handles_labels()
                        labels=tuple(x.title() for x in labels)
                        leg=fig.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                              loc = 'lower center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                              bbox_to_anchor=(0.5,0.0),
                              fontsize=13)
                    
                k += 1
            if type(ax_size)==tuple:
                while k <ax_size[0]*ax_size[1]:
                    i=ax_convert(k)
                    ax[i].remove()
                    k+=1
            leg.set_in_layout(False)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.15)
        else:
            for comp,j in zip(comp_list,range(len(comp_list))):
                budget_values = self.budgetDF.loc[PhyTime,comp].copy().values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
                if wall_units:
                    budget = budget_values[:Y_extent,x_list]/budget_scale[x_list]
                else:
                    budget = budget_values[:Y_extent,x_list]
                if self.comp == 'uv':
                    budget= budget * -1.0
                ax.plot(Y_coords[x_list,:Y_extent],budget,linestyle_list[j%len(linestyle_list)],label=comp)
                if wall_units:
                    if self.comp == 'uv':
                        ax.set_ylabel(r'$-\overline{%s} \cdot \nu/u_\tau^4$'%self.comp,fontsize=16)
                    elif self.comp == 'k':
                        ax.set_ylabel(r'$%s \cdot \nu/u_\tau^4$'%self.comp,fontsize=16)
                    else:
                        ax.set_ylabel(r'$\overline{%s} \cdot \nu/u_\tau^4$'%self.comp,fontsize=16)
                    ax.set_xlabel(r"$Y^+$")
                    ax.set_xscale('log')
                    ax.set_xlim(left=1.0)
                else:
                    if self.comp == 'uv':
                        ax.set_ylabel(r'$-\overline{%s}/U_{b0}^2$'%self.comp,fontsize=16)
                    elif self.comp == 'k':
                        ax.set_ylabel(r'$%s/U_{b0}^2$'%self.comp,fontsize=16)
                    else:
                        ax.set_ylabel(r'$\overline{%s}/U_{b0}^2$'%self.comp,fontsize=16)
                        ax.set_xlabel(r"$y/\delta$",fontsize=18)
                
            
                
                ax.grid()
            leg=fig.legend(labels=comp_list,fontsize='x-large',loc = 'lower center',ncol=4,bbox_to_anchor=(0.45,0.0))
            leg.set_in_layout(False)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.1)
            if isinstance(ax,list):
                if len(x_list)==1:
                    return fig, ax[0]
                else:
                    return fig, ax
            else:
                return fig, ax
            
        return fig, ax
    def plot_integral_budget(self,PhyTime,wall_units=True,fig='',ax=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        comp_list = tuple([x[1] for x in self.budgetDF.index])
        y_coords = self.AVG_DF.CoordDF['y'].dropna().values
        x_coords = self.AVG_DF.CoordDF['x'].dropna().values
        u_tau_star, delta_v_star = wall_unit_calc(self.AVG_DF,PhyTime)
        if not fig:
            fig,ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplots(1,1,1)
        for comp in comp_list:
            integral_budget = np.zeros(self.AVG_DF.NCL[0])
            budget_term = self.budgetDF.loc[PhyTime,comp].copy().values.reshape((self.AVG_DF.NCL[1],self.AVG_DF.NCL[0]))
            for i in range(self.AVG_DF.NCL[0]):
                integral_budget[i] = 0.5*integrate.simps(budget_term[:,i],y_coords)
                if wall_units:
                    delta_star=1.0
                    integral_budget[i] /=(delta_star*u_tau_star[i]**3/delta_v_star[i])
            ax.plot(x_coords,integral_budget)
        
        if wall_units:
            ax.set_ylabel(r"$\int_{-\delta}^{\delta} %s$ budget $dy\times u_{\tau}^4\delta/\nu $" %self.comp,fontsize=16)
        else:
            ax.set_ylabel(r"$\int_{-\delta}^{\delta} %s$ budget $dy\times 1/U_{b0}^3 $"%self.comp,fontsize=16)
        ax.set_xlabel(r"$x/\delta$",fontsize=18)
        ax.legend(comp_list,loc = 'upper center',ncol=4,bbox_to_anchor=(0.5,-0.25))
        ax.grid()
        return fig, ax
    def __str__(self):
        return self.budgetDF.__str__()
class CHAPSim_u2_budget(CHAPSim_budget):
    def __init__(self,time,AVG_DF):
        super().__init__(time,AVG_DF)
        self.comp = 'uu'
        try:      
            if isinstance(time,float) or isinstance(time,int):
                self.budgetDF = self.__u2_extract(time)
            
            elif isinstance(time,list):
                for PhyTime in time:
                    if not hasattr(self, 'budgetDF'):
                        self.budgetDF = self.__u2_extract(time)
                        
                    else:
                        local_u2_budgetDF = self.__u2_extract(time)
                        DF_concat = [self.budgetDF,local_u2_budgetDF]
                        self.budgetDF = pd.concat(DF_concat)
            else:
                raise TypeError
        except TypeError:
            print("`time' can only be a float or a list")
            raise
    def __u2_extract(self,PhyTime):

        Normal_u2DF = super()._budget_extract(PhyTime,self.comp[0],self.comp[1])

        return Normal_u2DF

class CHAPSim_v2_budget(CHAPSim_budget):
    def __init__(self,time,AVG_DF):
        super().__init__(time,AVG_DF)
        self.comp = 'vv'
        try:      
            if isinstance(time,float):
                self.budgetDF = self.__v2_extract(time)
                
            elif isinstance(time,list):
                for PhyTime in time:
                    if not hasattr(self, 'budgetDF'):
                        self.budgetDF = self.__v2_extract(time)
                        
                    else:
                        local_v2_budgetDF = self.__v2_extract(time)
                        DF_concat = [self.budgetDF,local_v2_budgetDF]
                        self.budgetDF = pd.concat(DF_concat)
                        
            else:
                raise TypeError
        except TypeError:
            print("`time' can only be a float or a list")
            raise
    def __v2_extract(self,PhyTime):
        Normal_v2DF = super()._budget_extract(PhyTime,self.comp[0],self.comp[1])
        return Normal_v2DF

class CHAPSim_w2_budget(CHAPSim_budget):
    def __init__(self,time,AVG_DF):
        super().__init__(time,AVG_DF)
        self.comp = 'ww'
        try:      
            if isinstance(time,float):
                self.budgetDF = self.__w2_extract(time)
            elif isinstance(time,list):
                for PhyTime in time:
                    if not hasattr(self, 'budgetDF'):
                        self.budgetDF = self.__w2_extract(time)
                    else:
                        local_w2_budgetDF = self.__w2_extract(time)
                        DF_concat = [self.budgetDF,local_w2_budgetDF]
                        self.budgetDF = pd.concat(DF_concat)
            else:
                raise TypeError
        except TypeError:
            print("`time' can only be a float or a list")
            raise
    def __w2_extract(self,PhyTime):
        Normal_w2DF = super()._budget_extract(PhyTime,self.comp[0],self.comp[1])
        return Normal_w2DF
    
class CHAPSim_uv_budget(CHAPSim_budget):
    def __init__(self,time,AVG_DF):
        super().__init__(time,AVG_DF)
        self.comp = 'uv'
        try:      
            if isinstance(time,float):
                self.budgetDF = self.__uv_extract(time)
            elif isinstance(time,list):
                for PhyTime in time:
                    if not hasattr(self, 'budgetDF'):
                        self.budgetDF = self.__uv_extract(time)
                    else:                        
                        local_uv_budgetDF = self.__uv_extract(time)
                        DF_concat = [self.budgetDF,local_uv_budgetDF]
                        self.budgetDF = pd.concat(DF_concat)
            else:
                raise TypeError
        except TypeError:
            print("`time' can only be a float or a list")
            raise
    def __uv_extract(self,PhyTime):
        Normal_uvDF = super()._budget_extract(PhyTime,self.comp[0],self.comp[1])
        return Normal_uvDF
class CHAPSim_k_budget(CHAPSim_budget):
    def __init__(self,time,AVG_DF):
        super().__init__(time,AVG_DF)
        self.comp = 'k'
        try:      
            if isinstance(time,float):
                #u2_budget = CHAPSim_u2_budget(time,AVG_DF)
                #v2_budget = CHAPSim_v2_budget(time,AVG_DF)
                #w2_budget = CHAPSim_w2_budget(time,AVG_DF)
                #self.budgetDF = 0.5*(u2_budget.budgetDF +v2_budget.budgetDF\
                #                     +w2_budget.budgetDF)
                self.budgetDF = self.__k_extract(time)
            elif isinstance(time,list):
                for PhyTime in time:
                    if not hasattr(self, 'budgetDF'):
                        self.budgetDF = self.__k_extract(PhyTime)
                    else: 
                        local_uv_budgetDF = self.__k_extract(PhyTime)
                        DF_concat = [self.budgetDF,local_uv_budgetDF]
                        self.budgetDF = pd.concat(DF_concat)
            else:
                raise TypeError
        except TypeError:
            print("`time' can only be a float or a list")
            raise
    def __k_extract(self,PhyTime):
        
        u2_budget = CHAPSim_u2_budget(PhyTime,self.AVG_DF)
        v2_budget = CHAPSim_v2_budget(PhyTime,self.AVG_DF)
        w2_budget = CHAPSim_w2_budget(PhyTime,self.AVG_DF)
        
        u2_values = u2_budget.budgetDF.values
        v2_values = v2_budget.budgetDF.values
        w2_values = w2_budget.budgetDF.values
        
        k_values = 0.5*(u2_values + v2_values + w2_values)

        budget_index = tuple([x[1] for x in u2_budget.budgetDF.index])
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        phystring_index = []
        for i in range(7):
            phystring_index.append(PhyTime)
        k_budgetDF = pd.DataFrame(k_values,index = pd.MultiIndex.\
                                from_arrays([phystring_index,budget_index]))
        return k_budgetDF
        
class CHAPSim_autocov():
    def __init__(self,comp1,comp2,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True):
        file_names = time_extract(path_to_folder,abs_path)
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        times = list(dict.fromkeys(time_list))
        if time0:
            times = list(filter(lambda x: x > time0, times))
        #times=times[:10]
        self._meta_data = CHAPSim_meta(path_to_folder)
        self.comp=(comp1,comp2)
        if x_split_list:
            self.x_split_list = x_split_list
        try:
            self._AVG_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times_temp= times
            times_temp.remove(max(times))
            self._AVG_data = CHAPSim_AVG(max(times_temp),self._meta_data,path_to_folder,time0)
        i=1
        
        for timing in times:       
            self._inst_data = CHAPSim_Inst(timing,self._meta_data,path_to_folder)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                autocorr, index = self.__autocorr(timing,max(times),comp1,comp2,homogen,x_split_list)
            else:
                local_autocorr, index = self.__autocorr(timing,max(times),comp1,comp2,homogen,x_split_list)
                assert local_autocorr.shape == autocorr.shape, "shape of previous array (%d,%d) " % autocorr.shape\
                    + " and current array (%d,%d) must be the same" % local_autocorr.shape
                autocorr = autocorr*coe3 + local_autocorr*coe2
            i += 1
        autocorr = autocorr.T
        print(index)
        #if hasattr(self,x_split_list):
        #    self.autocorrDF = pd.DataFrame(autocorr,index=pd.MultiIndex.from_arrays(index))
        #else:
        self.autocorrDF = pd.DataFrame(autocorr,index=index)
    
    def __autocorr(self,time,AVG_time,comp1,comp2,homogen,x_split_list=''):
        
        if x_split_list:
            split_index = []
            direction_index = []
            for x in x_split_list:
                if x > self._meta_data.NCL[0]:
                    raise ValueError("\033[1;32 value in x_split_list cannot be larger"\
                                     +"than x_size: %d, %d" %(x,self._meta_data.NCL[0]))
            for i in range(len(x_split_list)-1):
                x_point1 = x_split_list[i]
                x_point2 = x_split_list[i+1]
                autocorr_tempDF = self.__autocorr_calc(time,AVG_time,comp1,comp2,x_point1, x_point2,homogen)
                if i==0:
                    autocorrDF = autocorr_tempDF   
                else:
                    concatDF =[autocorrDF,autocorr_tempDF]
                    autocorrDF = pd.concat(concatDF, axis=1)
                split_list = ['Split ' + str(i+1),'Split ' + str(i+1)]
                direction_list = ['x','z']
                split_index.extend(split_list)
                direction_index.extend(direction_list)
            index = [split_index,direction_index]
                
        else:
            x_point1 = 0
            x_point2 = self._meta_data.NCL[0]
            autocorrDF = self.__autocorr_calc(time,AVG_time,comp1,comp2,x_point1, x_point2,homogen)
            index = ['x','z']
        return autocorrDF.values, index
    def __autocorr_calc(self,PhyTime,AVG_time,comp1,comp2,x_point1, x_point2,homogen):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        if type(AVG_time) == float:
            AVG_time = "{:.9g}".format(AVG_time)
        
        NCL_local = self._meta_data.NCL
        velo1 = self._inst_data.InstDF.loc[PhyTime,comp1].values.\
                    reshape((NCL_local[2],NCL_local[1],NCL_local[0]))
        AVG1 = self._AVG_data.flow_AVGDF.loc[AVG_time,comp1].values\
                    .reshape((NCL_local[1],NCL_local[0]))
        fluct1 = np.zeros_like(velo1)
        for i in range(NCL_local[2]):
            fluct1[i] = velo1[i] - AVG1
        
        velo2 = self._inst_data.InstDF.loc[PhyTime,comp2].values.\
                    reshape((NCL_local[2],NCL_local[1],NCL_local[0]))
        AVG2 = self._AVG_data.flow_AVGDF.loc[AVG_time,comp2].values\
                    .reshape((NCL_local[1],NCL_local[0]))
        fluct2 = np.zeros_like(velo2)
        
        for i in range(NCL_local[2]):
            fluct2[i] = velo2[i] - AVG2
        fluct1_0 = fluct1[:,:,x_point1:x_point2]
        fluct2_0 = fluct2[:,:,x_point1:x_point2]
        #x direction calculator
        x_size = int(np.trunc(0.5*fluct1_0.shape[2]))
        z_size = int(np.trunc(0.5*fluct2_0.shape[0]))
        
        R_x = np.zeros((NCL_local[1],x_size))
        R_z = np.zeros((NCL_local[1],z_size))
        #x_end = int(np.trunc((x_point1 +x_point2)*0.5))
        t0 = time.time()
        if homogen:
            R_x = CHAPSim_autocov._loop_accelerator_x(fluct1_0,fluct2_0,R_x,x_size)
            R_z = CHAPSim_autocov._loop_accelerator_z(fluct1_0,fluct2_0,R_z,z_size)
            R_x = R_x/(fluct2_0.shape[0]*x_size) #Divide by the number of values
            R_z = R_z/(fluct1_0.shape[2]*z_size)
        else:
            R_x = CHAPSim_autocov._loop_accelerator_x_non_homogen(fluct1_0,
                                                        fluct2_0,R_x,x_size)
                                                                    
            R_z = CHAPSim_autocov._loop_accelerator_z_non_homogen(fluct1_0,
                                                        fluct2_0,R_z,z_size)
            R_x = R_x/(fluct1_0.shape[0])
            R_z = R_z/(z_size)
        

        t1 = time.time()
    
        
        R_x = R_x.reshape((NCL_local[1]*x_size))
        R_z = R_z.reshape(z_size*NCL_local[1])
        
        R_xDF = pd.DataFrame(R_x)
        R_zDF = pd.DataFrame(R_z)
        autocorr = pd.concat([R_xDF,R_zDF],axis=1)
        print(t1-t0)
        return autocorr
        
    def autocorr_contour(self,comp,Y_plus=False,Y_plus_max ='', which_split='',norm=True,fig='',ax=''):
        assert(comp=='x' or comp =='z')
        if Y_plus_max and Y_plus == False:
            warnings.warn("\033[1;33Ignoring `Y_plus_max' value: Y_plus == False")
        NCL_local = self._meta_data.NCL
        if (hasattr(self,'x_split_list') and which_split) or not hasattr(self,'x_split_list'):
            if which_split and hasattr(self,'x_split_list'):
                assert(type(which_split)==int)
                split_string = "Split "+ str(which_split)
                x_point2 = self.x_split_list[which_split]
                x_point1 = point1 = self.x_split_list[which_split-1]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                Ruu = self.autocorrDF.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))

            else:
                x_point1 = point1 = 0
                x_point2 = NCL_local[0]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                Ruu = self.autocorrDF.loc[comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
            # Normalising autocovariance to form autocorrelation
            if norm:
                Ruu_0 = Ruu[:,0].copy()
                for i in range(size):
                    Ruu[:,i] = Ruu[:,i]/Ruu_0
            if not fig:
                fig,ax = plt.subplots(figsize=[10,3])
            elif not ax:
                ax = fig.add_subplot(1,1,1)
                
            
            coord = self._meta_data.CoordDF[comp].copy().dropna()\
                    .values[point1:point1+size]
            y_coord = self._meta_data.CoordDF['y'].copy().dropna()\
                    .values
            
            if Y_plus:
                avg_time = self._AVG_data.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                u_tau_star, delta_v_star = wall_unit_calc(self._AVG_data,avg_time)
                y_coord = y_coord[:int(y_coord.size/2)]
                y_coord = (1-np.abs(y_coord))/delta_v_star[point1]
                if Y_plus_max:
                    y_coord = y_coord[y_coord<Y_plus_max]
                    Ruu = Ruu[:len(y_coord)]
            X,Y = np.meshgrid(coord,y_coord)
            ax1 = ax.pcolormesh(X,Y,Ruu,cmap='jet')
            ax = ax1.axes
            if Y_plus:
                ax.set_ylabel(r"$Y^{+0}$", fontsize=16)
            else:
                ax.set_ylabel(r"$y/\delta$", fontsize=16)
            ax.set_xlabel(r"$\Delta %s/\delta$" %comp, fontsize=18)
            fig.colorbar(ax1,ax=ax)
        elif hasattr(self,'x_split_list') and not which_split:
            if fig or ax:
                warnings.warn("\033[1;33fig and ax are overridden in this case")
            ax_size = len(self.x_split_list)-1
            fig,ax = plt.subplots(ax_size,figsize=[8,ax_size*2.6])
            point1 = 0
            for i in range(1,len(self.x_split_list)):
                split_string = "Split "+ str(i)
                if comp =='x':
                    x_point2 = self.x_split_list[i]   
                    x_point1 = point1 = self.x_split_list[i-1]
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))

                Ruu = self.autocorrDF.loc[split_string,comp].dropna().values\
                        .reshape((NCL_local[1],size)) 
                #Noramlise if this is indicated
                if norm:
                    Ruu_0 = Ruu[:,0].copy()
                    for j in range(size):
                        Ruu[:,j] = Ruu[:,j]/Ruu_0
                    
                coord = self._meta_data.CoordDF[comp].dropna()\
                        .values[point1:point1+size]
                y_coord = self._meta_data.CoordDF['y'].dropna()\
                        .values
                if Y_plus:
                    avg_time = self._AVG_data.flow_AVGDF.index[0][0]
                    #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                    u_tau_star, delta_v_star = wall_unit_calc(self._AVG_data,avg_time)
                    y_coord = y_coord[:int(y_coord.size/2)]
                    y_coord = (1-np.abs(y_coord))/delta_v_star[point1]
                    if Y_plus_max:
                        y_coord = y_coord[y_coord<Y_plus_max]
                        Ruu = Ruu[:len(y_coord)]
                        
                X,Y = np.meshgrid(coord,y_coord)
                ax1 = ax[i-1].pcolormesh(X,Y,Ruu,cmap='jet')
                ax[i-1] = ax1.axes
                if i==len(self.x_split_list)-1:
                    if Y_plus:
                        ax[i-1].set_xlabel(r"$\Delta %s/\delta$" %comp, fontsize=18)
                if Y_plus:
                    ax[i-1].set_ylabel(r"$Y^{+0}$", fontsize=16)
                else:
                    ax[i-1].set_ylabel(r"$y/\delta$", fontsize=16)
                fig.colorbar(ax1,ax=ax[i-1])
        fig.subplots_adjust(hspace = 0.4)
        
        return fig,ax
    def autocorr_line_plot(self,comp,y_val,y_type=0,which_split='',norm=True,fig='',ax=''):
        
        pass
    @staticmethod
    @numba.njit(parallel=True)
    def _loop_accelerator_x(fluct1,fluct2,R_x,x_size):
        for ix0 in numba.prange(x_size):
            for z in numba.prange(fluct1.shape[0]):
                for ix in numba.prange(x_size):
                    R_x[:,ix0] += fluct1[z,:,ix]*fluct2[z,:,ix0+ix]
        return R_x
    @staticmethod
    @numba.njit(parallel=True)
    def _loop_accelerator_z(fluct1,fluct2,R_z,z_size):
        for iz0 in numba.prange(z_size):
            for ix in numba.prange(fluct1.shape[2]):
                for iz in numba.prange(z_size):
                    R_z[:,iz0] += fluct1[iz,:,ix]*fluct2[iz+iz0,:,ix]
        return R_z
    @staticmethod
    @numba.njit(parallel=True)
    def _loop_accelerator_x_non_homogen(fluct1,fluct2,R_x,x_size):
        for ix0 in numba.prange(x_size):
            for z in numba.prange(fluct1.shape[0]):
                    R_x[:,ix0] += fluct1[z,:,0]*fluct2[z,:,ix0]
        return R_x
    @staticmethod
    @numba.njit(parallel=True)
    def _loop_accelerator_z_non_homogen(fluct1,fluct2,R_z,z_size):
        for iz0 in numba.prange(z_size):
            for ix in numba.prange(fluct1.shape[2]):
                    R_z[:,iz0] += fluct1[0,:,ix]*fluct2[iz0,:,ix]
        return R_z
    def Spectra_calc(self,comp, y_index, which_split='',fig='',ax=''):
        NCL_local = self._meta_data.NCL
        if not fig:
            fig,ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        x_coord = self._meta_data.CoordDF['x'].dropna().values
        if (hasattr(self,'x_split_list') and which_split) or not hasattr(self,'x_split_list'):
            if which_split and hasattr(self,'x_split_list'):
                assert(type(which_split)==int)
                split_string = "Split "+ str(which_split)
                x_point2 = self.x_split_list[which_split]
                x_point1 = point1 = self.x_split_list[which_split-1]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                Ruu = self.autocorrDF.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))

            else:
                x_point1 = point1 = 0
                x_point2 = NCL_local[0]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                Ruu = self.autocorrDF.loc[comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
            wavenumber_spectra = fftpack.dct(Ruu[y_index])
            coord = self._meta_data.CoordDF[comp].dropna()\
                        .values[point1:point1+size]
            delta_comp = coord[1]-coord[0]
            Fs = (2.0*np.pi)/delta_comp
            comp_size= wavenumber_spectra.size
            wavenumber_comp = np.arange(comp_size)*Fs/comp_size
            
            ax.plot(wavenumber_comp,np.abs(wavenumber_spectra))
            wavenumber_comp = wavenumber_comp[wavenumber_comp>0]
            ax.plot(wavenumber_comp,wavenumber_comp**(-5/3),label='Kolmogorov spectrum')
                    
        elif hasattr(self,'x_split_list') and not which_split:
            point1 = 0
            x_coord = self._meta_data.CoordDF['x'].dropna().values[self.x_split_list[:-1]]       
            for i in range(1,len(self.x_split_list)):
                split_string = "Split "+ str(i)
                if comp =='x':
                    x_point2 = self.x_split_list[i]   
                    x_point1 = point1 = self.x_split_list[i-1]
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))

                Ruu = self.autocorrDF.loc[split_string,comp].dropna().values\
                        .reshape((NCL_local[1],size))[y_index]
                wavenumber_spectra = fftpack.dct(Ruu)
                coord = self._meta_data.CoordDF[comp].dropna()\
                        .values[point1:point1+size]
                delta_comp = coord[1]-coord[0]
                Fs = (2.0*np.pi)/delta_comp
                comp_size= wavenumber_spectra.size
                wavenumber_comp = np.arange(comp_size)*Fs/comp_size
                
                ax.plot(wavenumber_comp,np.abs(wavenumber_spectra),label="x=%.3f"%x_coord[i-1])
            wavenumber_comp = wavenumber_comp[wavenumber_comp>0]
            ax.plot(wavenumber_comp,wavenumber_comp**(-5/3),label='Kolmogorov spectrum')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(CT.flip_leg_col(handles,4),CT.flip_leg_col(labels,4),
                  loc = 'upper center',ncol=4*(len(labels)>3)+len(labels)*(len(labels)<4),
                  bbox_to_anchor=(0.5,-0.2),
                  fontsize=16)
        ax.set_xlabel(r"$\kappa_%s$"%comp,fontsize=20)
        string= (ord(self.comp[0])-ord('u')+1,ord(self.comp[1])-ord('u')+1,comp)
        ax.set_ylabel(r"$E_{%d%d}(\kappa_%s)$"%string,fontsize=20)
        ax.grid()
        ax.set_xscale('log')
        fig.tight_layout()
        #ax.set_yscale('log')
        return fig, ax
    def __str__(self):
        return self.autocorrDF.__str__()
class CHAPSim_Ruu(CHAPSim_autocov):
    def __init__(self,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True):
        super().__init__('u','u',x_split_list,path_to_folder,time0,abs_path,homogen)
class CHAPSim_Rvv(CHAPSim_autocov):
    def __init__(self,x_split_list='',path_to_folder='',time0='',abs_path='',homogen=True):
        super().__init__('v','v',x_split_list,path_to_folder,time0,abs_path,homogen)
class CHAPSim_Rww(CHAPSim_autocov):
    def __init__(self,x_split_list='',path_to_folder='',time0='',abs_path='',homogen=True):
        super().__init__('w','w',x_split_list,path_to_folder,time0,abs_path,homogen)
class CHAPSim_Ruv(CHAPSim_autocov):
    def __init__(self,x_split_list='',path_to_folder='',time0='',abs_path='',homogen=True):
        super().__init__('u','v',x_split_list,path_to_folder,time0,abs_path,homogen)
class CHAPSim_k_spectra(CHAPSim_autocov):
    def __init__(self,x_split_list='',path_to_folder='',time0='',abs_path='',homogen=True):
        
        super().__init__('u','u',x_split_list,path_to_folder,time0,abs_path,homogen)
        self.autocorr_uu = self.autocorrDF
        super().__init__('v','v',x_split_list,path_to_folder,time0,abs_path,homogen)
        self.autocorr_vv = self.autocorrDF
        super().__init__('w','w',x_split_list,path_to_folder,time0,abs_path,homogen)
        self.autocorr_ww = self.autocorrDF
        
    def spectra_plot(self,comp, y_index, which_split='',fig='',ax=''):
        NCL_local = self._meta_data.NCL
        print(NCL_local[2])
        if not fig:
            fig,ax = plt.subplots()
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        if (hasattr(self,'x_split_list') and which_split) or not hasattr(self,'x_split_list'):
            if which_split:
                assert(type(which_split)==int)
                split_string = "Split "+ str(which_split)
                x_point2 = self.x_split_list[which_split]
                x_point1 = point1 = self.x_split_list[which_split-1]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    print(NCL_local[2])
                    size = int(np.trunc(NCL_local[2]))
                print(size)
                R_uu = self.autocorr_uu.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                R_vv = self.autocorr_vv.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                R_ww = self.autocorr_ww.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
            else:
                x_point1 = point1 = 0
                x_point2 = NCL_local[0]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                R_uu = self.autocorr_uu.loc[comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                R_vv = self.autocorr_vv.loc[comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                R_ww = self.autocorr_ww.loc[comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
            Phi_uu = fftpack.dct(R_uu[y_index])
            Phi_vv = fftpack.dct(R_vv[y_index])
            Phi_ww = fftpack.dct(R_ww[y_index])
            wavenumber_spectra = 0.5*(Phi_uu + Phi_vv + Phi_ww)
            coord = self._meta_data.CoordDF[comp].dropna()\
                        .values[point1:point1+size]
            delta_comp = coord[1]-coord[0]
            Fs = (2.0*np.pi)/delta_comp
            comp_size= wavenumber_spectra.size
            wavenumber_comp = np.arange(comp_size)*Fs/comp_size
            
            ax.plot(wavenumber_comp,np.abs(wavenumber_spectra))
            ax.plot(wavenumber_comp,wavenumber_comp**(-5/3))
                    
        elif hasattr(self,'x_split_list') and not which_split:
            point1 = 0
            for i in range(1,len(self.x_split_list)):
                split_string = "Split "+ str(i)
                if comp =='x':
                    x_point2 = self.x_split_list[i]   
                    x_point1 = point1 = self.x_split_list[i-1]
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))

                R_uu = self.autocorr_uu.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                R_vv = self.autocorr_vv.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                R_ww = self.autocorr_ww.loc[split_string,comp].copy().dropna().values\
                    .reshape((NCL_local[1],size))
                Phi_uu = fftpack.dct(R_uu[y_index])
                Phi_vv = fftpack.dct(R_vv[y_index])
                Phi_ww = fftpack.dct(R_ww[y_index])
                wavenumber_spectra = 0.5*(Phi_uu + Phi_vv + Phi_ww)
                coord = self._meta_data.CoordDF[comp].dropna()\
                        .values[point1:point1+size]
                delta_comp = coord[1]-coord[0]
                Fs = (2.0*np.pi)/delta_comp
                comp_size= wavenumber_spectra.size
                wavenumber_comp = np.arange(comp_size)*Fs/comp_size
                
                ax.plot(wavenumber_comp,np.abs(wavenumber_spectra))
            ax.plot(wavenumber_comp,wavenumber_comp**(-5/3))
        ax.set_xlabel(r"$\kappa_z$",fontsize=18)
        ax.set_ylabel(r"$E(\kappa_z)$")
        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig, ax
#==================================================================================================
#Autocorr v2
class CHAPSim_autocov2():
    def __init__(self,comp1,comp2,max_x_sep=None,max_z_sep=None,path_to_folder='',time0='',abs_path=True):
        file_names = time_extract(path_to_folder,abs_path)
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        times = list(dict.fromkeys(time_list))
        if time0:
            times = list(filter(lambda x: x > time0, times))
        #times.sort(); times= times[-3:]
        self._meta_data = CHAPSim_meta(path_to_folder)
        self.comp=(comp1,comp2)
        self.NCL = self._meta_data.NCL
        try:
            self._avg_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            self._avg_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0)
        i=1

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]/2)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("\033[1;32 Variable max_z_sep must be less than half NCL3 in readdata file\n")
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]/2)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("\033[1;32 Variable max_x_sep must be less than half NCL3 in readdata file\n")

        for timing in times:
            self._fluct_data = CHAPSim_fluct(self._avg_data,timing,self._meta_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                autocorr = self.__autocov_calc(self._fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
            else:
                local_autocorr = self.__autocov_calc(self._fluct_data,comp1,comp2,timing,max_x_sep,max_z_sep)
                assert local_autocorr.shape == autocorr.shape, "shape of previous array (%d,%d) " % autocorr.shape\
                    + " and current array (%d,%d) must be the same" % local_autocorr.shape
                autocorr = autocorr*coe3 + local_autocorr*coe2
            i += 1
            index=['x','z']
            self.autocorrDF = pd.DataFrame(autocorr.T,index=index)
            self.max_x_sep=max_x_sep
            self.max_z_sep=max_z_sep
    @staticmethod
    def __autocov_calc(fluct_data,comp1,comp2,PhyTime,max_x_sep,max_z_sep):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)
        NCL=fluct_data.NCL        

        fluct_vals1=fluct_data.fluctDF.loc[PhyTime,comp1].values.reshape(NCL[::-1])
        fluct_vals2=fluct_data.fluctDF.loc[PhyTime,comp2].values.reshape(NCL[::-1])
        time1=time.time()
        
        R_x = CHAPSim_autocov2.__autocov_calc_x(fluct_vals1,fluct_vals2,*NCL,max_x_sep)
        R_z = CHAPSim_autocov2.__autocov_calc_z(fluct_vals1,fluct_vals2,*NCL,max_z_sep)
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
    def __autocov_calc_z(fluct1,fluct2,NCL1,NCL2,NCL3,max_z_step):
        R_z = np.zeros((max_z_step,NCL2,NCL1))
        if max_z_step >0:
            for iz0 in numba.prange(max_z_step):
                for iz in numba.prange(NCL3-max_z_step):
                    R_z[iz0,:,:] += fluct1[iz,:,:]*fluct2[iz+iz0,:,:]
        return R_z
    @staticmethod
    @numba.njit(parallel=True)
    def __autocov_calc_x(fluct1,fluct2,NCL1,NCL2,NCL3,max_x_step):
        
        R_x = np.zeros((max_x_step,NCL2,NCL1-max_x_step))
        if max_x_step >0:
            for ix0 in numba.prange(max_x_step):
                for ix in numba.prange(NCL1-max_x_step):
                    for iz in numba.prange(NCL3):
                        R_x[ix0,:,ix] += fluct1[iz,:,ix]*fluct2[iz,:,ix0+ix]
        return R_x
    def autocorr_contour_y(self,axis_vals,comp,Y_plus=False,Y_plus_0=False,Y_plus_max ='',norm=True,fig='',ax=''):
        if comp == 'x':
            shape=(self.max_x_sep,self.NCL[1],self.NCL[0]-self.max_x_sep)
        elif comp=='z':
            shape=(self.max_z_sep,self.NCL[1],self.NCL[0])
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        index_vals = CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
        if not hasattr(index_vals,'__iter__'):
            axis_vals=[index_vals]   
        #raise TypeError("Variable `axis_vals' must be an int or iterable\n")
        Ruu = self.autocorrDF.loc[comp].dropna().values.reshape(shape)[:,:,index_vals]
        # print(CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,0.8,[10,20,30]))
        # print(CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,0.8,[10,20,30],mode='disp_thickness'))
        # print(CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,20,[10,20,30],mode='wall'))
#        CT.y_coord_delta_star_index(self._avg_data,self._meta_data.CoordDF,0.8,None)
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0
        if not fig:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif not ax:
            subplot_kw = {'squeeze':'False'}
            ax = fig.subplots(len(axis_vals),subplot_kw)
        x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
                    .values
        for i in range(len(axis_vals)):
            y_coord = self._meta_data.CoordDF['y'].copy().dropna()\
                    .values
            coord = self._meta_data.CoordDF[comp].copy().dropna()\
                    .values[:shape[0]]
            if Y_plus:
                avg_time = self._avg_data.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                u_tau_star, delta_v_star = wall_unit_calc(self._avg_data,avg_time)
                y_coord = y_coord[:int(y_coord.size/2)]
                if i==0:
                    Ruu = Ruu[:,:y_coord.size]
                if Y_plus_0:
                    y_coord = (1-np.abs(y_coord))/delta_v_star[0]
                else:   
                    y_coord = (1-np.abs(y_coord))/delta_v_star[index_vals[i]]

            X,Y = np.meshgrid(coord,y_coord)
            ax1 = ax[i,0].pcolormesh(X,Y,np.squeeze(Ruu[:,:,i]).T,cmap='jet')
            ax1.axes.set_xlabel(r"$\Delta %s/\delta$" %comp, fontsize=20)
            if Y_plus and Y_plus_0:
                ax1.axes.set_ylabel(r"$Y^{+0}$", fontsize=20)
            elif Y_plus and not Y_plus_0:
                ax1.axes.set_ylabel(r"$Y^{+}$", fontsize=20)
            else:
                ax1.axes.set_ylabel(r"$y/\delta$", fontsize=20)
            ax1.axes.set_title(r"$x=%.3f$"%axis_vals[i],loc='left')
            if Y_plus_max:
                ax1.axes.set_ylim(top=Y_plus_max)
            fig.colorbar(ax1,ax=ax1.axes)
            fig.tight_layout()
        return fig, ax

    def autocorr_contour_x(self,comp,axis_vals,axis_mode='half_channel',norm=True,fig='',ax=''):
        if comp == 'x':
            shape=(self.max_x_sep,self.NCL[1],self.NCL[0]-self.max_x_sep)
        elif comp=='z':
            shape=(self.max_z_sep,self.NCL[1],self.NCL[0])
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        if not hasattr(axis_vals,'__iter__'):
            axis_vals=[axis_vals]
            #raise TypeError("Variable `axis_vals' must be an int or iterable\n")
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        Ruu_all = self.autocorrDF.loc[comp].dropna().values.reshape(shape)#[:,axis_vals,:]
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i].reshape(Ruu[:,:,i].shape)
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
        x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
                .values[:shape[2]]
        y_coord = self._meta_data.CoordDF['y'].copy().dropna().values
        coord = self._meta_data.CoordDF[comp].copy().dropna()\
                .values[:shape[0]]
        for i in range(len(axis_vals)):

            X,Y = np.meshgrid(x_coord,coord)
            ax1 = ax[i].pcolormesh(X,Y,Ruu[:,i],cmap='jet')
            ax1.axes.set_xlabel(r"$x/\delta$", fontsize=20)
            
            ax1.axes.set_ylabel(r"$\Delta %s/\delta$" %comp, fontsize=20)
            title = r"$%s=%.3f$"%("y" if axis_mode=='half_channel' \
                        else "\delta_u" if axis_mode=='disp_thickness' \
                        else "\theta" if axis_mode=='mom_thickness' else "Y^+", axis_vals[i] )
            ax1.axes.set_title(title,fontsize=15,loc='left')
            fig.colorbar(ax1,ax=ax1.axes)
        fig.tight_layout()
        return fig, ax
    def plot_autocorr_line(self,comp,axis_vals,y_vals,y_mode='half_channel',norm=True,fig='',ax=''):
        if comp == 'x':
            shape=(self.max_x_sep,self.NCL[1],self.NCL[0]-self.max_x_sep)
        elif comp=='z':
            shape=(self.max_z_sep,self.NCL[1],self.NCL[0])
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        axis_index = CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
        if not hasattr(axis_index,'__iter__'):
            axis_index = [axis_index]
            #raise TypeError("Variable `axis_vals' must be an int or iterable\n")

        

        coord = self._meta_data.CoordDF[comp].dropna().values[:shape[0]]
        if not hasattr(y_vals,'__iter__'):
            y_vals = [y_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,axis_vals,y_mode)
        Ruu = self.autocorrDF.loc[comp].dropna().values.reshape(shape)#[:,axis_vals,:]
        # Ruu=np.zeros((shape[0],len(y_vals),len(axis_index)))
        # for x_index,y_indices in zip(axis_index,y_index_axis_vals):
        #     Ruu[:,:,i] = Ruu_all[:,y_indices,x_index].reshape(Ruu[:,y_indices,x_index].shape)
        # print(Ruu.shape)
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0
        linestyle_list=['-','--','-.']
        if not fig:
            fig,ax = plt.subplots(len(y_vals),figsize=[10,4*len(y_vals)],squeeze=False)
            
        elif not ax:
            subplot_kw={'squeeze',False}
            ax = fig.subplots(len(y_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()
        coord = self._meta_data.CoordDF[comp].dropna().values[:shape[0]]

        for j in range(len(y_vals)):
            for i in range(len(axis_index)):
                ax[j].plot(coord,Ruu[:,y_index_axis_vals[j],axis_index[i]],
                        label=r"$x=%.3f$"%(axis_vals[i]),
                        linestyle=linestyle_list[i%len(linestyle_list)])
                y_unit="y" if y_mode=='half_channel' \
                        else "\delta_u" if y_mode=='disp_thickness' \
                        else "\theta" if y_mode=='mom_thickness' else "Y^+"
                ax[j].set_title(r"$%s=%.3f$"%(y_unit,y_vals[j]),loc='left')
            ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp,fontsize=20)
            ax[j].set_xlabel(r"$%s/\delta$"%comp,fontsize=20)
            if j==0:
                handles, labels = ax[j].get_legend_handles_labels()
                ax[j].legend(handles, labels,fontsize=15)
        
        fig.tight_layout()
        return fig, ax
    def plot_spectra(self,comp,axis_vals,y_vals,y_mode='half_channel',fig='',ax=''):
        if comp == 'x':
            shape=(self.max_x_sep,self.NCL[1],self.NCL[0]-self.max_x_sep)
        elif comp=='z':
            shape=(self.max_z_sep,self.NCL[1],self.NCL[0])
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        axis_index = CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
        if not hasattr(axis_index,'__iter__'):
            axis_index = [axis_index]
            #raise TypeError("Variable `axis_vals' must be an int or iterable\n")

        

        coord = self._meta_data.CoordDF[comp].dropna().values[:shape[0]]
        if not hasattr(y_vals,'__iter__'):
            y_vals = [y_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,axis_vals,y_mode)
        Ruu = self.autocorrDF.loc[comp].dropna().values.reshape(shape)#[:,axis_vals,:]
        # Ruu=np.zeros((shape[0],len(y_vals),len(axis_index)))
        # for x_index,y_indices in zip(axis_index,y_index_axis_vals):
        #     Ruu[:,:,i] = Ruu_all[:,y_indices,x_index].reshape(Ruu[:,y_indices,x_index].shape)
        # print(Ruu.shape)
        linestyle_list=['-','--','-.']
        if not fig:
            fig,ax = plt.subplots(len(y_vals),figsize=[10,4*len(y_vals)],squeeze=False)
        elif not ax:
            subplot_kw={'squeeze',False}
            ax = fig.subplots(len(y_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()

        for j in range(len(y_vals)):
            for i in range(len(axis_index)):
                wavenumber_spectra = fftpack.dct(Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                delta_comp = coord[1]-coord[0]
                Fs = (2.0*np.pi)/delta_comp
                comp_size= wavenumber_spectra.size
                wavenumber_comp = np.arange(comp_size)*Fs/comp_size
                y_unit="y" if y_mode=='half_channel' \
                        else "\delta_u" if y_mode=='disp_thickness' \
                        else "\theta" if y_mode=='mom_thickness' else "Y^+"
                ax[j].plot(wavenumber_comp,np.abs(wavenumber_spectra),
                        label=r"$x=%.3f$"%(axis_vals[i]),
                        linestyle=linestyle_list[i%len(linestyle_list)])
                ax[j].set_title(r"$%s=%.3f$"%(y_unit,y_vals[j]),loc='left')
            string= (ord(self.comp[0])-ord('u')+1,ord(self.comp[1])-ord('u')+1,comp)
            ax[j].set_ylabel(r"$E_{%d%d}(\kappa_%s)$"%string,fontsize=20)
            ax[j].set_xlabel(r"$\kappa_%s$"%comp,fontsize=20)
        
            if j==0:
                handles, labels = ax[j].get_legend_handles_labels()
                ax[j].legend(handles, labels,fontsize=15)
        fig.tight_layout()
        return fig, ax
#==================================================================================================
    def spectrum_contour(self,comp,axis_vals,axis_mode='half_channel',fig='',ax=''):
        if comp == 'x':
            shape=(self.max_x_sep,self.NCL[1],self.NCL[0]-self.max_x_sep)
        elif comp=='z':
            shape=(self.max_z_sep,self.NCL[1],self.NCL[0])
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        if not hasattr(axis_vals,'__iter__'):
            axis_vals=[axis_vals]
            #raise TypeError("Variable `axis_vals' must be an int or iterable\n")
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        Ruu_all = self.autocorrDF.loc[comp].dropna().values.reshape(shape)#[:,axis_vals,:]
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i].reshape(Ruu[:,:,i].shape)

        if not fig:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif not ax:
            subplot_kw = {'squeeze':False}
            ax = fig.subplots(len(axis_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()
        x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
                .values[:shape[2]]
        coord = self._meta_data.CoordDF[comp].copy().dropna()\
                .values[:shape[0]]
        ax_out=[]
        for i in range(len(axis_vals)):
            wavenumber_spectra = np.zeros((shape[0],shape[2]))
            for j in range(shape[2]):
                wavenumber_spectra[:,j]=fftpack.dct(Ruu[:,i,j])
            delta_comp = coord[1]-coord[0]
            Fs = (2.0*np.pi)/delta_comp
            comp_size= shape[0]
            wavenumber_comp = np.arange(comp_size)*Fs/comp_size
            X, Y = np.meshgrid(x_coord,wavenumber_comp)
            ax1 = ax[i].pcolormesh(X,Y,np.abs(wavenumber_spectra),cmap='jet')
            ax1.axes.set_xlabel(r"$x/\delta$", fontsize=20)
            ax1.axes.set_ylabel(r"$\kappa_%s$"%comp, fontsize=20)
            title = r"$%s=%.3f$"%("y" if axis_mode=='half_channel' \
                        else "\delta_u" if axis_mode=='disp_thickness' \
                        else "\theta" if axis_mode=='mom_thickness' else "Y^+", axis_vals[i] )
            ax1.axes.set_ylim([np.amin(wavenumber_comp[1:]),np.amax(wavenumber_comp)])
            ax1.axes.set_title(title,fontsize=15,loc='left')
            fig.colorbar(ax1,ax=ax1.axes)
            ax_out.append(ax1)
        fig.tight_layout()
        return fig, np.array(ax_out)

class CHAPSim_Quad_Anal():
    def __init__(self,h_list,path_to_folder='',time0='',abs_path=True):
        file_names = time_extract(path_to_folder,abs_path)
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        times = list(dict.fromkeys(time_list))
        if time0:
            times = list(filter(lambda x: x > time0, times))
        # times.sort(); times= times[-3:]
        self._meta_data = CHAPSim_meta(path_to_folder)
        self.NCL = self._meta_data.NCL
        try:
            self._avg_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            self._avg_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0)
        i=1
        for timing in times:
            self._fluct_data = CHAPSim_fluct(self._avg_data,timing,self._meta_data,
                                            time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)
            fluct_uv, quadrant_array = self.__quadrant_extract(self._fluct_data.fluctDF,timing,self.NCL)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                quad_anal_array = self.__quad_calc(self._avg_data,fluct_uv,quadrant_array,self.NCL,h_list,timing)
            else:
                local_quad_anal_array = self.__quad_calc(self._avg_data,fluct_uv,quadrant_array,self.NCL,h_list,timing)
                assert local_quad_anal_array.shape == quad_anal_array.shape, "shape of previous array (%d,%d) " % quad_anal_array.shape\
                    + " and current array (%d,%d) must be the same" % local_quad_anal_array.shape
                autocorr = quad_anal_array*coe3 + local_quad_anal_array*coe2
            i += 1
        index=[[],[]]
        for h in h_list:
            index[0].extend([h]*4)
        index[1]=[1,2,3,4]*len(h_list)
        self.QuadAnalDF=pd.DataFrame(quad_anal_array,index=index)

    @staticmethod
    def __quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,PhyTime):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)

        avg_time = list(set([x[0] for x in avg_data.UU_tensorDF.index]))[0]
        uv_q=np.zeros((4,*NCL[::-1][1:]))

        u=avg_data.flow_AVGDF.loc[avg_time,'u'].values.reshape(NCL[::-1][1:])
        v=avg_data.flow_AVGDF.loc[avg_time,'v'].values.reshape(NCL[::-1][1:])
        uu=avg_data.UU_tensorDF.loc[avg_time,'uu'].values.reshape(NCL[::-1][1:])
        vv=avg_data.UU_tensorDF.loc[avg_time,'vv'].values.reshape(NCL[::-1][1:])
        u_rms = np.sqrt(uu-u*u)
        v_rms = np.sqrt(vv-v*v)

        quad_anal_array=np.empty((len(h_list)*4,NCL[0]*NCL[1]))
        for h,j in zip(h_list,range(len(h_list))):
            for i in range(1,5):
                quad_array=quadrant_array == i
                # print(quad_array)
                fluct_array = np.abs(quad_array*fluct_uv) > h*u_rms*v_rms
                uv_q[i-1]=np.mean(fluct_uv*fluct_array,axis=0)
            quad_anal_array[j*4:j*4+4]=uv_q.reshape((4,NCL[0]*NCL[1]))
        return quad_anal_array

    @staticmethod
    def __quadrant_extract(fluctDF,PhyTime,NCL):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)
        if len(set([x[0] for x in fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in fluctDF.index]))[0]

        u_array=fluctDF.loc[PhyTime,'u'].values.reshape(NCL[::-1])
        v_array=fluctDF.loc[PhyTime,'v'].values.reshape(NCL[::-1])

        u_array_isneg=u_array<0
        v_array_isneg=v_array<0

        quadrant_array = np.zeros_like(v_array_isneg,dtype='i4')

        for i in range(1,5): #determining quadrant
            if i ==1:
                quadrant_array_temp = np.logical_and(~u_array_isneg,~v_array_isneg)#not fluct_u_isneg and not fluct_v_isneg
                quadrant_array += quadrant_array_temp*1
            elif i==2:
                quadrant_array_temp = np.logical_and(u_array_isneg,~v_array_isneg)#not fluct_u_isneg and fluct_v_isneg
                quadrant_array += quadrant_array_temp*2
            elif i==3:
                quadrant_array_temp =  np.logical_and(u_array_isneg,v_array_isneg)
                quadrant_array += quadrant_array_temp*3
            elif i==4:
                quadrant_array_temp =  np.logical_and(~u_array_isneg,v_array_isneg)#fluct_u_isneg and not fluct_v_isneg
                quadrant_array += quadrant_array_temp*4

        assert(quadrant_array.all()<=4 and quadrant_array.all()>=1)  
        fluct_uv=u_array*v_array 

        return fluct_uv, quadrant_array 
    def line_plot(self,h_list,coord_list,prop_dir,x_vals=0,y_mode='half_channel',norm=False,fig='',ax=''):
        assert x_vals is None or not hasattr(x_vals,'__iter__')
        if not fig:
            fig, ax = plt.subplots(4,len(coord_list),figsize=[12,5*len(coord_list)],squeeze=False)
        elif not ax:
            subplot_kw={'squeeze':False}
            ax = fig.subplots(4,len(coord_list),figsize=[12,5*len(coord_list)],subplot_kw=subplot_kw)
        if prop_dir =='y':
            index = CT.coord_index_calc(self._meta_data.CoordDF,'x',coord_list)
            arr_len = self.NCL[2]
        elif prop_dir == 'x':
            index = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                        coord_list,x_vals,y_mode)
            arr_len = self.NCL[1]
            if x_vals is not None:
                index=list(itertools.chain(*index))
        else:
            raise ValueError("The propagation direction of the quadrant analysis must be `x' or `y'")
        if norm: 
            avg_time = list(set([x[0] for x in self._avg_data.UU_tensorDF.index]))[0]
            u=self._avg_data.flow_AVGDF.loc[avg_time,'u'].values.reshape(self.NCL[::-1][1:])
            v=self._avg_data.flow_AVGDF.loc[avg_time,'v'].values.reshape(self.NCL[::-1][1:])
            UV=self._avg_data.UU_tensorDF.loc[avg_time,'uv'].values.reshape(self.NCL[::-1][1:])
            uv = UV-u*v

        coords = self._meta_data.CoordDF[prop_dir].dropna().values

        unit="x/\delta"if prop_dir =='y' else "y/\delta" if y_mode=='half_channel' \
                else "\delta_u" if y_mode=='disp_thickness' \
                else "\theta" if y_mode=='mom_thickness' else "Y^+" \
                if x_vals is None or x_vals!=0 else "Y^{+0}"

        for i in range(1,5):
            for h in h_list:
                quad_anal = self.QuadAnalDF.loc[h,i].values.reshape(self.NCL[::-1][1:])
                if norm:
                    quad_anal/=uv
                for j in range(len(coord_list)):
                    if x_vals is None and prop_dir=='x':
                        quad_anal_index= np.zeros(self.NCL[0])
                        for k in range(self.NCL[0]):
                            quad_anal_index[k]=quad_anal[index[k][j],k]
                    else:
                        quad_anal_index=quad_anal[index[j],:] if prop_dir == 'x' else quad_anal[:,index[j]].T
                    ax[i-1,j].plot(coords,quad_anal_index,label=r"$h=%.5g$"%h)
                    ax[i-1,j].set_xlabel(r"$%s/\delta$"%prop_dir,fontsize=20)
                    ax[i-1,j].set_ylabel(r"$Q%d$"%i,fontsize=20)
                    ax[i-1,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left',fontsize=16)
            
        handles, labels = ax[0,0].get_legend_handles_labels()
        ax[0,0].legend(handles, labels,fontsize=12)
        fig.tight_layout()
        return fig, ax
    
class CHAPSim_mom_balance():
    def __init__(self,meta_data='',path_to_folder='',time0='',abs_path=True,tgpost=False):
        file_names = time_extract(path_to_folder, abs_path)
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        times = list(dict.fromkeys(time_list))
        if time0:
            times = list(filter(lambda x: x > time0, times))
        if meta_data:
            self._meta_data = meta_data
        else:
            self._meta_data = CHAPSim_meta(path_to_folder,abs_path,tgpost)
        self.NCL = self._meta_data.NCL
        self._AVG_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0,abs_path)
        self.Mom_balanceDF = self.__mom_balance_calc(max(times))
    def __mom_balance_calc(self,time):
        if type(time) == float:
            time = "{:.9g}".format(time)
        #Advection
        comp_list=['u','v','w']
        for comp in comp_list:
            advectionDF_comp = self.__advection_calc(comp,time)
            viscousDF_comp = self.__viscous_calc(comp, time)
            ReynoldsDF_comp = self.__Reynolds_calc(comp,time)
            if comp ==comp_list[0]:
                advectionDF = advectionDF_comp.copy()
                viscousDF = viscousDF_comp.copy()
                ReynoldsDF = ReynoldsDF_comp.copy()
            else:
                advectionDF = pd.concat([advectionDF,advectionDF_comp],axis=1)
                viscousDF = pd.concat([viscousDF,viscousDF_comp],axis=1)
                ReynoldsDF = pd.concat([ReynoldsDF,ReynoldsDF_comp],axis=1)
        
        advectionDF.columns = [['Advection']*3,['u','v','w']]
        viscousDF.columns = [['Viscous']*3,['u','v','w']]
        ReynoldsDF.columns = [['Reynolds']*3,['u','v','w']]
        
        
        P_gradDF = self.__p_grad_calc(time)
        P_gradDF.columns = [['P_grad']*3,['u','v','w']]
        concatDF_list = [advectionDF,P_gradDF,viscousDF,ReynoldsDF]
      
        Momentum_balance = pd.concat(concatDF_list,axis=1)
        Momentum_balance = Momentum_balance.T
        return Momentum_balance
        
    def __advection_calc(self,comp,time):
        u_velo_comp = self._AVG_data.flow_AVGDF.loc[time,comp].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
            
        u_velo = self._AVG_data.flow_AVGDF.loc[time,'u'].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        v_velo = self._AVG_data.flow_AVGDF.loc[time,'v'].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        
        comp_u = np.zeros_like(u_velo_comp)
        comp_v = np.zeros_like(u_velo_comp)
        
        NCL=self._AVG_data.NCL
        uu = u_velo_comp*u_velo
        uv = u_velo_comp*v_velo
        
        comp_u = Grad_calc(self._meta_data.CoordDF, uu, 'x')
        comp_v = Grad_calc(self._meta_data.CoordDF, uv, 'y')
        
        advection = comp_u.reshape(NCL[0]*NCL[1]) + comp_v.reshape(NCL[0]*NCL[1])
        advectionDF = pd.DataFrame(advection)
        return advectionDF
    def __p_grad_calc(self,time):
        p_field = self._AVG_data.flow_AVGDF.loc[time,'P'].values\
                    .reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        p_grad = np.zeros((3,self._AVG_data.NCL[1],self._AVG_data.NCL[0]))            
        p_grad[:2] = _scalar_grad(self._meta_data.CoordDF, p_field)
        p_grad = p_grad.reshape((3,self._AVG_data.NCL[1]*self._AVG_data.NCL[0]))
        p_gradDF = pd.DataFrame(p_grad.T)
        return p_gradDF
    
    def __viscous_calc(self,comp,time):
        velo_comp = self._AVG_data.flow_AVGDF.loc[time,comp].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        REN = self._AVG_data._metaDF.loc['REN'].values[0]
        viscous_comp = (-1/REN)*_scalar_laplacian(self._meta_data.CoordDF,velo_comp)
        viscous_comp = viscous_comp.reshape((self._AVG_data.NCL[1]*self._AVG_data.NCL[0]))
        viscous_compDF = pd.DataFrame(viscous_comp)
        return viscous_compDF
        
    def __Reynolds_calc(self,comp,time):
        UU_tensor = self._AVG_data.UU_tensorDF.loc[time,'u'+comp].values\
            .reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        if comp=='u':
            UV_tensor = self._AVG_data.UU_tensorDF.loc[time,comp+'v'].values\
            .reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        else:
            UV_tensor = self._AVG_data.UU_tensorDF.loc[time,'v'+comp].values\
            .reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        
            
        U_velo= self._AVG_data.flow_AVGDF.loc[time,'u'].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        V_velo= self._AVG_data.flow_AVGDF.loc[time,'v'].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
        
            
        comp_velo = self._AVG_data.flow_AVGDF.loc[time,comp].values.\
            reshape((self._AVG_data.NCL[1],self._AVG_data.NCL[0]))
            
        uu = UU_tensor - comp_velo*U_velo
        uv = UV_tensor - comp_velo*V_velo
        
        Reynolds = Grad_calc(self._meta_data.CoordDF, uu, 'x') + Grad_calc(self._meta_data.CoordDF, uv, 'y')
        Reynolds = Reynolds.reshape((self._AVG_data.NCL[1]*self._AVG_data.NCL[0]))
        ReynoldsDF = pd.DataFrame(Reynolds)
        return ReynoldsDF
    def diff_contour(self,comp='',norm=False,abs=False,fig='',ax=''):
        term_list = ['Advection','P_grad','Viscous','Reynolds']
        momentum = np.zeros((3,self.NCL[1]*self.NCL[0]))
        if norm:
            momentum_norm = np.zeros((3,self.NCL[1]*self.NCL[0]))
        for term in term_list:
            momentum += self.Mom_balanceDF.loc[term].values
            if norm:
                momentum_norm += np.abs(self.Mom_balanceDF.loc[term].values)
        
        momentumDF = pd.DataFrame(momentum,index=['u','v','w'])
        if norm:
            momentum_normDF = pd.DataFrame(momentum_norm,index=['u','v','w'])
        y_coord = self._meta_data.CoordDF['y'].dropna().values
        x_coord = self._meta_data.CoordDF['x'].dropna().values
        
        X, Y = np.meshgrid(x_coord,y_coord)
        
        
        if comp:
            if not fig:
                fig, ax = plt.subplots(figsize=[10,5])
            elif not ax:
                ax = fig.add_subplot(1,1,1)
            momentum_comp = momentumDF.loc[comp].copy().values.reshape((self.NCL[1],self.NCL[0]))
            if norm:
                momentum_norm = momentum_normDF.loc[comp].values.reshape((self.NCL[1],self.NCL[0]))
                momentum_comp /= momentum_norm
            if abs:
                momentum_comp = np.abs(momentum_comp)
            #print(momentum_comp[1])
            ax1 = ax.pcolormesh(X,Y,momentum_comp,cmap='jet')
            ax = ax1.axes
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            fig.colorbar(ax1,ax=ax)
            ax.set_xlabel(r"$x/\delta$",fontsize=18)
            ax.set_ylabel(r"$y/\delta$",fontsize=18)
        else:
            if not fig:
                fig, ax = plt.subplots(3,figsize=[10,15])
            elif not ax:
                ax = fig.add_subplot(1,1,1)
            i=0    
            for comp in ['u','v','w']:
                momentum_comp = momentumDF.loc[comp].values.reshape((self.NCL[1],self.NCL[0]))
                ax1 = ax[i].pcolormesh(X,Y,momentum_comp,cmap='jet')
                ax = ax1.axes
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
                fig.colorbar(ax1,ax=ax)
                ax.set_xlabel(r"$x/\delta$",fontsize=18)
                ax.set_ylabel(r"$y/\delta$",fontsize=18)
                i+=1
        return fig, ax
    def plot_total_imbalance(self,comp,norm=False,abs=False,fig='',ax=''):
        assert comp in ['u','v','w','norm2']
        term_list = ['Advection','P_grad','Viscous','Reynolds']
        momentum = np.zeros((self.NCL[1],self.NCL[0]))
        if norm:
            momentum_norm = np.zeros((self.NCL[1],self.NCL[0]))
        if comp=='norm2':
            comp_list = ['u','v','w']
            momentum_norm2 = np.zeros((self.NCL[1],self.NCL[0]))
            momentum_norm_norm2 = np.zeros((self.NCL[1],self.NCL[0]))
            for comp_dir in comp_list:
                for term in term_list:
                    momentum += self.Mom_balanceDF.loc[term,comp_dir].values.reshape((self.NCL[1],self.NCL[0]))
                    if norm:
                        momentum_norm += np.abs(self.Mom_balanceDF.loc[term,comp_dir].values.reshape((self.NCL[1],self.NCL[0])))
                momentum_norm2 += momentum**2
                if norm:
                    momentum_norm_norm2 += momentum_norm**2
            momentum = np.sqrt(momentum_norm2)
            momentum_norm = np.sqrt(momentum_norm_norm2)
        else:        
            for term in term_list:
                momentum += self.Mom_balanceDF.loc[term,comp].values.reshape((self.NCL[1],self.NCL[0]))
                if norm:
                    momentum_norm += np.abs(self.Mom_balanceDF.loc[term,comp].values.reshape((self.NCL[1],self.NCL[0])))
        y_coord = self._meta_data.CoordDF['y'].dropna().values
        x_coord = self._meta_data.CoordDF['x'].dropna().values
        total_imbal = np.zeros(self.NCL[0])
        norm_terms = np.zeros(self.NCL[0])
        for i in range(self.NCL[0]):
            total_imbal[i] = integrate.simps(momentum[:,i],y_coord)
            if norm:
                norm_terms[i] =  integrate.simps(momentum_norm[:,i],y_coord)
                total_imbal[i] /= norm_terms[i]
        if not fig:
            fig, ax = plt.subplots(figsize=[10,5])
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        if abs:
            total_imbal = np.abs(total_imbal)    
        ax.plot(x_coord,total_imbal)
        ax.grid()
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$\int^\delta_{-\delta}$ Momentum Imabalance $dy \times 1/U_{bo}^2$")
        return fig, ax
    def __str__(self):
        return self.Mom_balanceDF.__str__()
        
def time_extract(path_to_folder,abs_path=True):
    if abs_path:
        mypath = os.path.join(path_to_folder,'1_instant_D')
    else:
        mypath = os.path.abspath(os.path.join(path_to_folder,'1_instant_D'))
    file_names = [f for f in os.listdir(mypath) if f[:20]=='DNS_perioz_INSTANT_T']
    return file_names       

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
    ax.quiverkey(q,0.8,-0.125,key_size,r'$U/U_{b0}=%0.3f$'%key_size,labelpos='E',coordinates='axes')
    ax.set_xlabel(r"$%s$ direction" % axis1)
    ax.set_ylabel(r"$%s$ direction" % axis2)
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
