#=============================================================================#
#=============== CHAPSim Parallel - postprocessing ===========================#
# Postprocessing python scripts developed to be run on the parallel           #
# nodes on HPC machines. This has been tested on iceberg. This uses           #
# the message passing interface (MPI) for python (mpi4py). This has           #
# been developed to allow fast access to iceberg's filesystem and to          #  
# make use parallelism. If HPC resources have a sufficiently up-to-date       #  
# version of the module `numba' than this may not be required as the          #
# shared memory nodes can be used and numba parallel can be used.             #
# Methods that require extensive use of the filesystem particularly           #
# creating averages using the instantaneous data from CHAPSim                 #
#                                                                             #
#=====================  Capabilities =========================================#
# Parallelises the instantaneous and averaged fields spatially in either      #
# streamwise or wall-normal direction. It makes use of the pandas             #  
# library particularlly the DataFrame object to store data in easily          #  
# accessible form.                                                            #  
#===================== Structures and classes ================================#
# = parallel => provides tools to parallelise a particular direction          #  
#               determining local process array sizes and global start        #
#              and end points.                                                #  
# = CHAPSim_Inst => Creates DataFrame for the instantaneous flow fields       #  
#                   velocity vector and pressure                              #  
#                => method for producing a contour plot, mainly for           #  
#                   testing the parallelisation                               #
# = CHAPSim_AVG => Creates DataFrames for all the average varables            #      
#                  produced by the CHAPSim Solver                             #  
#               => method for producing a contour plot, mainly for            #  
#                  testing the parallelisation                                #  
# = CHAPSim_autocov => Creates DataFrames for the autocovariance in the       #   
#                      x and z direction                                      #  
#                   => Creates contour plot of the autocovariance or the      #  
#                      autocorrelation function (normalised ACF)              #  
#                   => Creates spectrum of the autocovariance                 #  
#       = CHAPSim_R<uu> => Children of CHAPSim_autocov for creating the above #
#                          for specific velocity components                   #  
#                       => for uu, vv, ww, uv velocity components             #  
#       = CHAPSim_k_spectra => creates spectrum of :                          #  
#                              0.5*(uu + vv + ww)                             #  
# = CHAPSim_uv_pdf => Creates joint probability distribution of  U and V to   #  
#                     aid in quadrant analysis                                #  
#                  => ***TO BE COMPLETED                                      #  
# = CHAPSim_Quad_Anal => Performs quadrant analysis using the method of       #  
#                        Willmarth and Lu 1972                                #  
#=============================================================================# 
#================ STATUS - In development and testing ========================#
# 
# -> testing on iceberg and sharc for large meshes to test memory issues
# -> test on single process
# -> CHAPSim_uv_pdf needs completing although this is not a priority
# -> Optimisations
#=============================================================================#

# MODULE IMPORTS
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import CHAPSim_post as cp
import CHAPSim_Tools as CT
import traceback
from scipy import  fftpack #,integrate
import os
#import time
import warnings
import gc

from mpi4py import MPI #Importing MPI
import numba
mpl.rcParams['mathtext.fontset'] = 'stix'

class parallel():
    # Class for paralllelising a particular direction
    def __init__(self,NCL2): # input length to be parallelised
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        if rank ==0:
            array_sizes = np.zeros(size,dtype='i4')
            array_sizes[:] = int(np.trunc(NCL2/size)) #finds minimum possible size
            i=0
            # Adds to each process until the cumulative size equals the input array size
            while np.sum(array_sizes) < NCL2: 
                array_sizes[i] += 1
                i += 1
                if i == size:
                    i=0
        else:
            array_sizes = None
            
        array_sizes = comm.bcast(array_sizes,root=0) #broadcasts array
        if size > 1:
            self.array_size = array_sizes[rank] #finds local array size
            self.array_start = np.sum(array_sizes[:rank]) #finds start of global array
            self.array_end = self.array_start + self.array_size
        else: # If there is 1 process parallelisation returns original size
            self.array_size = array_sizes[rank]
            self.array_start = 0
            self.array_end = self.array_size
            
        self.array_sizes  = array_sizes
        
class CHAPSim_Inst():
    #Class for holding the instantaneous data
    def __init__(self,time,meta_data='',path_to_folder='',abs_path = True,par_dir='y',root=0):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        assert(par_dir == 'y' or par_dir =='x') # only x and y parallelisation
        
        if not meta_data: # if meta_data is not provided calculate it directly
            meta_data = cp.CHAPSim_meta(path_to_folder,abs_path)
        self.CoordDF = meta_data.CoordDF #have local copy of coordinate datafame
        self.NCL = meta_data.NCL # local copy of coordinate dimensions
        if par_dir=='y': # parallelise y direction
            self.par = parallel(self.NCL[1])
        else: # parallelise x direction
            self.par = parallel(self.NCL[0])
        self._par_dir=par_dir 
        assert(root<size and root>=0)
        self.__root=root
        #Give capacity for both float and lists
        if isinstance(time,float): 
            self.InstDF = self.__flow_extract(time,path_to_folder,abs_path)
        elif isinstance(time,list): # Allow for stacked dataframe at different times, practically unused thus requires retesting
            for PhyTime in time:
                if not hasattr(self, 'InstDF'): # calculates first datafrmae
                    self.InstDF = self.__flow_extract(PhyTime,path_to_folder,abs_path)
                else: #Variable already exists therfore stacks dataframe
                    local_DF = self.__flow_extract(PhyTime,path_to_folder,abs_path)
                    concat_DF = [self.InstDF,local_DF]
                    self.InstDF = pd.concat(concat_DF)
        else:
            raise TypeError("`time' must be either float or list")
    def __flow_extract(self,Time_input,path_to_folder,abs_path):
        """
        Extract velocity and pressure from the instantanous files on root process \
        then scatter among all processes using MPI command Scatterv, interpolate \
        and store data in DataFrame distributed on each process

        Parameters
        ----------
        Time_input : float
            time to extracted
        path_to_folder : string
            path to results folder
        abs_path : bool
            True if path given is absolute or false if relative

        Returns
        -------
        Instant_DF : Pandas DataFrame
            2-D DataFrame with primary indices [U,V,W,P] with length \
            the number of elements per process: NCL3*NCLX*array_size

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        # for i in range(size):
            # if i == rank:
        veloVector = ["_U","_V","_W","_P"]
        i=0
        for comp in veloVector:
            if rank == self.__root:
                #creating file names for data extraction
                instant = "%0.9E" % Time_input
                file_folder = "1_instant_D"                     
                file_string = "DNS_perioz_INSTANT_T" + instant
                #veloVector = ["_U","_V","_W","_P"]
                file_ext = ".D"
                
                # file_list=[]
                # for velo in veloVector:
                #creating path name
                if not abs_path:
                    file = (os.path.abspath(os.path.join(path_to_folder, \
                                 file_folder, file_string + comp + file_ext)))
                else:
                    file = (os.path.join(path_to_folder, \
                                 file_folder, file_string + comp + file_ext))
                #A list of all the relevant files for this timestep                           
                #open_list =[]
                #opening all the relevant files and placing them in a list
                
                file_temp = open(file,'rb')
                    
                #allocating arrays
                # int_info=np.zeros((4,4))
                # r_info = np.zeros((4,3))
        
         
                #extract meta data associated with file
                int_info=np.fromfile(file_temp,dtype='int32',count=4) 
                r_info=np.fromfile(file_temp,dtype='float64',count=3)
                    
                #Assign meta data to variables
                PhyTime=r_info[0]
                NCL1=int(int_info[0])
                NCL2=int(int_info[1])
                NCL3=int(int_info[2])
                
                dummy_size=NCL1*NCL2*NCL3 # size of output data
        
                # flow_info=np.zeros((dummy_size))
        
                
                #Extracting flow information
                flow_info=np.fromfile(file,dtype='float64',count=dummy_size)
                   
                #reshape to split data along parallelisation axis
                flow_info=flow_info.reshape((NCL3,NCL2,NCL1))
                
                if self._par_dir == 'y': #For y parallelisation split along axis 1
                    flow_info_par_chunks = np.split(flow_info,np.cumsum(self.par.array_sizes),axis=1)
                    NCLX=NCL1
                else: #For x parallelisation split along axis 2
                    if i ==0: # on first pass change the array size is one larger prior to interpolation
                        self.par.array_sizes[size-1] +=1
                    flow_info_par_chunks = np.split(flow_info,np.cumsum(self.par.array_sizes),axis=2)
                    NCLX=NCL2
                    
                for block, process in zip(flow_info_par_chunks,range(size)):
                    #For each split flatten and stack sequentially for MPI scattering
                    block=block.reshape(NCL3*NCLX*self.par.array_sizes[process])
                    if process == 0:
                        flow_info_par_scatter = block
                    else:
                        flow_info_par_scatter = np.hstack((flow_info_par_scatter,block))
                del flow_info ; del flow_info_par_chunks #Free memory
                file_temp.close()
                if i == 3:
                    del file_temp
            else:
                #initialisation on non-root ranks
                flow_info_par_scatter=None
                NCLX=None
                NCL3=None
                NCL1=None
                NCL2=None
                PhyTime=None
            
            if self._par_dir =='x' and  i==0 and rank ==size-1:
                    self.par.array_size += 1 #Update array to match size during scattering
            self.par.array_sizes = comm.bcast(self.par.array_sizes,root=self.__root) # update array to match root
            #Broadcasting variables from root to all
            NCLX = comm.bcast(NCLX,root=self.__root)
            NCL3 = comm.bcast(NCL3,root=self.__root)
            NCL1 = comm.bcast(NCL1,root=self.__root)
            NCL2 = comm.bcast(NCL2,root=self.__root)
            PhyTime = comm.bcast(PhyTime,root=self.__root)
            
            if i ==0: #Declare once on first pass
                flow_info_par = np.empty((4,NCL3*NCLX*self.par.array_size))
            
            sizes_array= tuple(NCL3*NCLX*self.par.array_sizes) #number of elements going to each array
            offset_array = np.empty(size,dtype='i4')
            for k in range(size): #Calculating start points (offsets) for each array to be scattered
                offset_array[k] = np.cumsum(NCL3*NCLX*self.par.array_sizes)[k] - NCL3*NCLX*self.par.array_sizes[k]
            offset_array = tuple(offset_array)
            #Scatter data
            comm.Scatterv([flow_info_par_scatter,sizes_array,offset_array,MPI.DOUBLE], flow_info_par[i],root=self.__root)
            
            i+=1
        # Array shape depends on parallelisation direction
        if self._par_dir == 'y': 
            array_shape=(4,NCL3,self.par.array_size,NCL1)
        else:
            array_shape=(4,NCL3,NCL2,self.par.array_size)
            
        flow_info_par = flow_info_par.reshape(array_shape)

        if self._par_dir == 'y': #Interpolate to centre the velocity vectors - originally staggered in CHAPSim
            flow_info1_par,self.par.array_size = self.__velo_interp(flow_info_par,NCL3,NCL1,self.par.array_size)
            flow_info1_par = flow_info1_par.reshape((4,NCL3*(NCL1-1)*self.par.array_size))
        else: #if parallelisation direction is x
            flow_info1_par,self.par.array_size = self.__velo_interp(flow_info_par,NCL3,NCL2,self.par.array_size)
            flow_info1_par = flow_info1_par.reshape((4,NCL3*NCL2*self.par.array_size))
        if self._par_dir =='x': # Correct array sizes back to original value
            self.par.array_sizes[size-1] -=1
    
        Phy_string = '%.10g' % PhyTime
        
        # creating dataframe index for the instantaneous flow fields
        index=pd.MultiIndex.from_arrays([[Phy_string,Phy_string,Phy_string,Phy_string],['u','v','w','P']])
        Instant_DF = pd.DataFrame(flow_info1_par,index=index)
        
        Instant_DF.index.name = (self.par.array_start,self.par.array_end) # So dataframe gives start and end points

        return Instant_DF        
    def __velo_interp(self,flow_info,NCL3, NCLX, array_size):
        """
        Interpolates the velocity field with two different methods depending of \
        parallelisation scheme. vector field scattered at h/2 in its coordinate \
        direction. Scalars e.g. pressure are already centred

        Parameters
        ----------
        flow_info : numpy array 
            Contains staggered flow data outer dimension is [u,v,w,P] respectively
        NCL3 : int4
            number of elements in the z direction
        NCLX : int
            number of elements in the non-parallelised direction
        array_size : int
            local size of the parallelised direction

        Returns
        -------
        flow_interp : numpy array 
            Contains centred flow data outer dimension is [u,v,w,P] respectively
        array_size_interp : int
            returns new size, if x direction parallelised then final process \
            will be an element shorter

        """
        comm=MPI.COMM_WORLD
        rank=comm.rank
        size = comm.Get_size()
        
        if self._par_dir == 'y':
            array_size_interp = array_size
            array_end = array_size_interp-1 #final value interpolated separately for each rank
            flow_interp = np.zeros((4,NCL3,array_size_interp,NCLX-1)) #initialisation of array
            
            for i in range(NCLX-1): #U velocity interpolation, missing last value due to interpolation
                flow_interp[0,:,:,i] = 0.5*(flow_info[0,:,:array_size_interp,i] + flow_info[0,:,:array_size_interp,i+1])
                
            for i in range(NCL3): #W velocity
                if i != NCL3-1: # Last value interpolated differently
                    flow_interp[2,i,:,:] = 0.5*(flow_info[2,i,:array_size_interp,:-1] + flow_info[2,i+1,:array_size_interp,:-1])
                else: #interpolation with first cell due to z periodicity BC
                    flow_interp[2,i,:,:] = 0.5*(flow_info[2,i,:array_size_interp,:-1] + flow_info[2,0,:array_size_interp,:-1])
            #Calculating array to send to previous rank
            
            send_array = flow_info[1,:,0,:-1].copy()
            recv_array = None
            # send first array to previous rank to allow interpolation of prev rank final value
            if rank > 0: #rank 0 has no previous rank
                comm.send(send_array,dest=rank-1,tag=(50+rank-1))
            # receive array from rank+1
            if rank < size-1: #rank size-1 has no rank+1
                recv_array = comm.recv(source=rank+1,tag=(50+rank))
            if rank==0: #send array to top rank because wall values only\
                #given on bottom hence required to interpolate with wall at the top
                comm.send(send_array,dest=size-1,tag=(50+rank-1))
            if rank == size-1: # Receive array from rank 0
                recv_array = comm.recv(source=0,tag=(50-1))
            #interpolate last velue with sent array
            flow_interp[1,:,array_end,:] = 0.5*(flow_info[1,:,array_end,:-1] + recv_array)
            for i in range(array_size_interp-1): #interpolate all except last value in paralleled direction
                flow_interp[1,:,i,:] = 0.5*(flow_info[1,:,i,:-1] + flow_info[1,:,i+1,:-1])
            flow_interp[3,:,:,:] = flow_info[3,:,:array_size_interp,:-1] #pressure already centred
        else:
            if rank == size-1: #final rank 1 element smaller in parallelisation direction
                array_size_interp = array_size-1
                #print(array_size_interp,flush=True)
            else:
                array_size_interp = array_size
            array_end = array_size_interp-1
            flow_interp = np.zeros((4,NCL3,NCLX,array_size_interp))
            for i in range(NCL3): #W velocity, comments same as above
                if i != NCL3-1:
                    flow_interp[2,i,:,:] = 0.5*(flow_info[2,i,:,:array_size_interp] + flow_info[2,i+1,:,:array_size_interp])
                else: #interpolation with first cell due to z periodicity BC
                    flow_interp[2,i,:,:] = 0.5*(flow_info[2,i,:,:array_size_interp] + flow_info[2,0,:,:array_size_interp])
            for i in range(NCLX):
                if i !=NCLX-1: #interpolate with neighbouring value
                    flow_interp[1,:,i,:] = 0.5*(flow_info[1,:,i,:array_size_interp] + flow_info[1,:,i+1,:array_size_interp])
                else: #final value interpolates with first due to reasons explained above
                    flow_interp[1,:,i,:] = 0.5*(flow_info[1,:,i,:array_size_interp] + flow_info[1,:,0,:array_size_interp])
            recv_array=None        
            #Send first value of array to previous rank
            if rank > 0:
                send_array=flow_info[0,:,:,0].copy()
                comm.send(send_array,dest=rank-1)
            if rank < size-1:
                recv_array=comm.recv(source=rank+1)
                #interpolate end values
                flow_interp[0,:,:,array_end] = 0.5*(flow_info[0,:,:,array_end] + recv_array)
                for i in range(array_size_interp-1): #interpolate remaining values
                    flow_interp[0,:,:,i] = 0.5*(flow_info[0,:,:,i] + flow_info[0,:,:,i+1])

            else: # Interpolate final rank 
                for i in range(array_size_interp):
                    flow_interp[0,:,:,i] = 0.5*(flow_info[0,:,:,i] + flow_info[0,:,:,i+1])
            flow_interp[3,:,:,:] = flow_info[3,:,:,:array_size_interp]
        
        return flow_interp, array_size_interp
    def inst_contour(self,axis1,axis2,axis3_value,flow_field,PhyTime,fig='',ax=''):
        """
        Creates contour plot primarily for testing the parallelisation of the code,
        Recommended to use the version in CHAPSim_post.py

        Parameters
        ----------
        axis1 : string
            Axis for creating 2-D contour out of 3-D data
        axis2 : string
            Axis for creating 2-D contour out of 3-D data
        axis3_value : int
            index along third dimension along which to postprocess 2-D plane 
        flow_field : string
            string of [u,v,w,P,mag], whether to plot a component from the 
            DataFrame or calculate velocity magnitude
        PhyTime : float (or string)
            Time to be postprocessed
        fig : TYPE, optional
            figures can be passed and hence updated 
        ax : TYPE, optional
            axes can be passed and hence updated 

        Raises
        ------
        ValueError
            Ensuring correct values are chosen for axes and flow_field variables.

        Returns
        -------
        fig : TYPE
            modified figure
        ax : TYPE
            modified axis 

        """
        comm=MPI.COMM_WORLD
        rank=comm.rank
        size = comm.Get_size()
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)
        
        if axis1 == axis2: 
            raise ValueError("Axis 1 cannot equal axis 2")
    
            
        axes = ['x','y','z']
        if axis1 not in axes or axis2 not in axes:
            raise ValueError("axis 1 and axis 2 must have values of %s, %s or %s" % axes)
        #Swap axes to ensure image always the appropriate orientation
        if axis1 != 'x' and axis1 != 'z':
            axis_temp = axis2
            axis2 = axis1
            axis1 = axis_temp
        elif axis1 == 'z' and axis2 == 'x':
            axis_temp = axis2
            axis2 = axis1
            axis1 = axis_temp
        #Extract field from data frame
        if flow_field == 'u' or flow_field =='v' or flow_field =='w' or flow_field =='P':
            local_velo = self.InstDF.loc[PhyTime,flow_field].values
        #Calculate velocity magnitude            
        elif flow_field == 'mag':            
            index = pd.MultiIndex.from_arrays([[PhyTime,PhyTime,\
                                                PhyTime],['u','v','w']])
            local_velo = np.sqrt(np.square(self.InstDF.loc[index]).sum(axis=0)).values
        else:
            raise ValueError("Not a valid argument")
        #Reshape into appropriate size
        if self._par_dir=='y':
            local_velo = local_velo.reshape((self.NCL[2],self.par.array_size,self.NCL[0]))
        else:
            local_velo = local_velo.reshape((self.NCL[2],self.NCL[1],self.par.array_size))
        #In these condiitons the arrays from different processes need to be concatenated
        if axis1 =='x' and axis2 == 'y':
            velo_post = local_velo[axis3_value,:,:].copy()
            NCLX = self.NCL[0]
            
        elif axis1 =='z' and axis2 == 'y' and self._par_dir=='y':
            velo_post = local_velo[:,:,axis3_value].copy()
           
            NCLX = self.NCL[2]
        elif axis1 =='x' and axis2 == 'z' and self._par_dir=='x':
            velo_post = local_velo[:,axis3_value,:].copy()
            NCLX = self.NCL[1]
            
        if 'NCLX' in locals(): 
            if rank != 0: # sending relevant array slice to rank 0 for concatenation
                send_array = velo_post.copy()
                comm.send(send_array,dest=0,tag=60+rank)
                
            else: # Receiving and stacking array on rank 0
                for i in range(1,size):
                    recv_array = None
                    recv_array = comm.recv(source=i,tag=60+i)
                    #print(velo_post.shape,recv_array.shape)
                    if self._par_dir == 'y':
                        if axis1=='z':
                            velo_post = np.hstack((velo_post,recv_array))
                                
                        else:
                            velo_post = np.vstack((velo_post,recv_array))
                            if i==size-1:
                                velo_post=velo_post.T
                    else:
                        velo_post = np.hstack((velo_post,recv_array))
                        if i==size-1:
                                velo_post=velo_post.T
                        
                    
        else: #all the required information is on a single rank
            x_range = self.InstDF.index.name
            if axis3_value < self.par.array_end and axis3_value >= self.par.array_start and rank != 0:
                if axis1 =='x' and axis2=='z':
                    velo_post = local_velo[:,axis3_value-self.par.array_start,:].copy()
                    velo_post = velo_post.T
                else:
                    velo_post = local_velo[:,:,axis3_value-self.par.array_start].copy()
                #If the relevant rank and rank !=0 send to rank 0
                comm.send(velo_post,dest=0,tag=77)
            elif axis3_value >= self.par.array_end or axis3_value < self.par.array_start and rank == 0:
                #If rank 0 is not the relevnt rank receive array from relevant rank
                recv_array = None
                recv_array=comm.recv(tag=77)
                #print('checkpoint1b',flush=True)
                velo_post=recv_array
            elif rank==0: #If the relevant rank is rank 0
                velo_post=local_velo[:,:,axis3_value-self.par.array_start]
        
            

        if rank ==0: # create figure on rank 0
            #print('checkpoint0a',flush=True)
            axis1_coords = self.CoordDF[axis1].dropna().values
            axis2_coords = self.CoordDF[axis2].dropna().values
            axis1_mesh, axis2_mesh = np.meshgrid(axis1_coords,axis2_coords)
            #print('checkpoint1a',flush=True)
            if not fig:
                fig,ax = plt.subplots(figsize=[10,5])
            elif not ax:
                ax = fig.add_subplot(1,1,1)
            #print('checkpoint2a',flush=True)
            #print(axis1_mesh.shape,axis2_mesh.shape,velo_post.shape)
            ax1 = ax.pcolormesh(axis1_coords,axis2_coords,velo_post.T,cmap='nipy_spectral')
            #print('checkpoint3a',flush=True)
            ax = ax1.axes
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            fig.colorbar(ax1,ax=ax)
            ax.set_xlabel(r"$%s/\delta$" % axis1,fontsize=18)
            ax.set_ylabel(r"$%s/\delta$" % axis2,fontsize=16)
            #print('checkpoint4a',flush=True)
        else:
            fig = None
            ax = None
        
        return fig, ax
                
class CHAPSim_AVG():
    def __init__(self,time,meta_data='',path_to_folder='',time0='',abs_path=True,par_dir='y',root=0):
        """
        A class for extracting averaged data from CHAPSim's averaged data files 
        Also has limited postprocessing abilities for more capabilities view 
        the serial version of the code CHAPSim_post.

        Parameters
        ----------
        time : float or list
            Time to be extracted
        meta_data : CHAPSim_meta, optional
            meta data from the results such as input parameters and coordinates.
            The default is ''.
        path_to_folder : string, optional
            path to the results folder. The default is ''.
        time0 : float or string, optional
            Initial time between which the average is calculated time0<time. 
            The default is ''.
        abs_path : bool, optional
            Whether the given path is an absolute or relative path.
            The default is True.
        par_dir : string, optional
            Parallelisation direction. The default is 'y'.
        root : int, optional
            which rank will be root during __init__. The default is 0.

        Raises
        ------
        TypeError
            `time' can only be a float or a list

        Returns
        -------
        None.

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        
        #self.NCL = np.zeros(3,dtype='int32')
        if not meta_data: #Calculate meta data if not present
            meta_data = cp.CHAPSim_meta(path_to_folder,abs_path)
        self._meta_data = meta_data
        self.CoordDF = meta_data.CoordDF
        self._metaDF = meta_data.metaDF
        self.NCL = meta_data.NCL
        
        if par_dir =='y':
            self.par = parallel(self.NCL[1])
        else:
            self.par = parallel(self.NCL[0])
        self.par_dir = par_dir
        assert(root<size and root>=0)
        self.__root = root
        
        if isinstance(time,float):
            DF_list = self.__AVG_extract(time,time0,path_to_folder,abs_path)
            #Each DataFrame is a different object in the list
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
                    DF_list = self.__AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                else:
                    DF_temp=[]
                    local_DF_list = self.__AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                    for DF, local_DF in zip(DF_list, local_DF_list): # concatenate each element of the list elementwise
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
            raise TypeError("`time' can only be a float or a list")
    
        
    def __AVG_extract(self,Time_input,time0,path_to_folder,abs_path):
        """
        Extract averaged flow information. Note this can be a memory intensive 
        function on large meshes hence there are frequent attempts to delete
        variables and call the garbage collector

        Parameters
        ----------
        Time_input : float
            Input time for extraction.
        time0 : float
            Initial time: average calculated between time0 and time
        path_to_folder : string
            path to the results folder.
        abs_path : bool
            Whether the given path is an absolute or relative path..

        Returns
        -------
        DF_list : list of pandas DataFrames
            List containing the averaged data 

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        
        #Loop to reduce peak memory use of the root process
        for i in range(7):
            if rank ==self.__root:
                if time0:
                    if i ==0:
                        #Extract time0 averaged data file
                        instant = "%0.9E" % time0
                        file_string = "DNS_perioz_AVERAGD_T" + instant + "_FLOW.D"
                        file_folder = "2_averagd_D"
                        if not abs_path:
                            file_path = os.path.abspath(os.path.join(path_to_folder, \
                                                     file_folder, file_string))
                        else:
                            file_path = os.path.join(path_to_folder, \
                                                     file_folder, file_string)
                                
                        file0 = open(file_path,'rb')
                        
                        int_info = np.zeros(4)
                        r_info = np.zeros(3)
                        int_info = np.fromfile(file0,dtype='int32',count=4)    
                        
                        NCL1 = int_info[0]
                        NCL2 = int_info[1]
                        NSTATIS0 = int_info[3] # Number of data points average is comprised of
                        dummy_size = NCL1*NCL2*50*3
                        r_info = np.fromfile(file0,dtype='float64',count=3)
                        
                        PhyTime = r_info[0]
                        #REN = r_info[1]
                        #DT = r_info[2]
                        AVG_info0 = np.zeros(dummy_size)
                    AVG_info0 = np.fromfile(file0,dtype='float64',count=dummy_size)
                
                if i== 0:
                    #Extract time averaged data file
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
                    
                    dummy_size = NCL1*NCL2*50*3
                    r_info = np.fromfile(file,dtype='float64',count=3)
                    
                    PhyTime = r_info[0]
                    #REN = r_info[1]
                    #DT = r_info[2]
                    AVG_info = np.zeros(dummy_size)
                AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)
                if time0:
                    #Calculating average between time0 and time
                    #Slightly convoluted calculation due to unexplained iceberg 
                    #error
                    NSTAT=(NSTATIS1-NSTATIS0)
                    NSTAT_AVG=AVG_info*NSTATIS1
                    NSTAT_AVG0=AVG_info0
                    NSTAT_AVG0*=NSTATIS0
                    AVG_info = (NSTAT_AVG - NSTAT_AVG0)/NSTAT
                    del AVG_info0
                AVG_info = AVG_info.reshape(3,50,NCL2,NCL1)
                #Setting up parallelisation of averaged data by splitting array along relevant axis
                if self.par_dir == 'y':
                    AVG_info_par_chunks = np.split(AVG_info,np.cumsum(self.par.array_sizes),axis=2)
                    NCLX=NCL1
                else:
                    AVG_info_par_chunks = np.split(AVG_info,np.cumsum(self.par.array_sizes),axis=3)
                    NCLX=NCL2
                del AVG_info
                for block, process in zip(AVG_info_par_chunks,range(size)):
                    block=block.reshape(3*50*NCLX*self.par.array_sizes[process])
                    #Reforming array to be scattered
                    if process == 0:
                        AVG_info_par_scatter = block
                    else:
                        AVG_info_par_scatter = np.hstack((AVG_info_par_scatter,block))
                del AVG_info_par_chunks #release memory
                gc.collect() #Call the garbage collector to try and free more memory
            else:
                AVG_info_par_scatter=None
                NCLX=None
                NCL1=None
                NCL2=None
                PhyTime=None
            #broadcasting information from root
            NCLX = comm.bcast(NCLX,root=self.__root)
            NCL1 = comm.bcast(NCL1,root=self.__root)
            NCL2 = comm.bcast(NCL2,root=self.__root)
            PhyTime = comm.bcast(PhyTime,root=self.__root)
            #creating empty array to filled by scattered array
            AVG_info_par_stack = np.empty((50*3*NCLX*self.par.array_size))
            sizes_array= tuple(50*3*NCLX*self.par.array_sizes) # size of array going to each process
            offset_array = np.empty(size,dtype='i4')
            #Creating array for start values of each process
            for k in range(size):
                offset_array[k] = np.cumsum(50*3*NCLX*self.par.array_sizes)[k] - 50*3*NCLX*self.par.array_sizes[k]
            offset_array = tuple(offset_array)
            #Scattering array
            comm.Scatterv([AVG_info_par_scatter,sizes_array,offset_array,MPI.DOUBLE], AVG_info_par_stack,root=self.__root)
            AVG_info_par_stack = AVG_info_par_stack.reshape((3,50,NCLX*self.par.array_size))
            #Collecting the different parts of the final array as there is progression through each loop
            if i ==0 :
                AVG_info_par = AVG_info_par_stack.copy()
            else:
                #print(AVG_info_par.shape,AVG_info_par_stack.shape,flush=True)
                AVG_info_par = np.concatenate((AVG_info_par,AVG_info_par_stack))
            del AVG_info_par_scatter
            gc.collect()
        if rank ==self.__root:            
            file.close()
            if time0:
                file0.close()
        array_size = self.par.array_size
        #Extracting arrays to form each DataFrame
        if self.par_dir=='y':
            AVG_info_par = AVG_info_par.reshape((21,50,array_size,NCL1))
            # mean velocity gradient tensor
            Velo_grad_tensor = np.zeros((9,array_size,NCL1))
            #mean presure*velocity gradient tensor
            Pr_Velo_grad_tensor = np.zeros((9,array_size,NCL1))
            #mean(velocity gradient tensor squared)
            DUDX2_tensor = np.zeros((81,array_size,NCL1))
        else:
            AVG_info_par = AVG_info_par.reshape((21,50,NCL2,array_size))
            Velo_grad_tensor = np.zeros((9,NCL2,array_size))
            Pr_Velo_grad_tensor = np.zeros((9,NCL2,array_size))
            DUDX2_tensor = np.zeros((81,NCL2,array_size))
            
        flow_AVG = AVG_info_par[0,:4,:,:] # mean flow variables
        
        PU_vector = AVG_info_par[2,:3,:,:] #mean pressure*velocity vector
        UU_tensor = AVG_info_par[3,:6,:,:] #velocity squared tensor
        #print(UU_tensor[0],flush=True)
        UUU_tensor = AVG_info_par[5,:10,:,:] #velcity cubed tensor
        
        for i in range(3):
            for j in range(3):
                Velo_grad_tensor[i*3+j,:,:] = AVG_info_par[6+j,i,:,:]
                Pr_Velo_grad_tensor[i*3+j,:,:] = AVG_info_par[9+j,i,:,:]
        for i in range(9):
            for j in range(9):
                DUDX2_tensor[i*9+j] = AVG_info_par[12+j,i,:,:] 
        del AVG_info_par
        gc.collect()   
        #======================================================================
        #Rearrage into a form for the 2-D dataframe
        if self.par_dir == 'y':
            flow_AVG = flow_AVG.reshape((4,NCL1*array_size))
            
            PU_vector = PU_vector.reshape((3,array_size*NCL1))
            UU_tensor = UU_tensor.reshape((6,array_size*NCL1))
            UUU_tensor = UUU_tensor.reshape((10,array_size*NCL1))
            Velo_grad_tensor = Velo_grad_tensor.reshape((9,array_size*NCL1))
            Pr_Velo_grad_tensor = Pr_Velo_grad_tensor.reshape((9,array_size*NCL1))
            DUDX2_tensor = DUDX2_tensor.reshape((81,array_size*NCL1))
        else:
            flow_AVG = flow_AVG.reshape((4,NCL2*array_size))
            
            PU_vector = PU_vector.reshape((3,NCL2*array_size))
            UU_tensor = UU_tensor.reshape((6,NCL2*array_size))
            UUU_tensor = UUU_tensor.reshape((10,NCL2*array_size))
            Velo_grad_tensor = Velo_grad_tensor.reshape((9,NCL2*array_size))
            Pr_Velo_grad_tensor = Pr_Velo_grad_tensor.reshape((9,NCL2*array_size))
            DUDX2_tensor = DUDX2_tensor.reshape((81,NCL2*array_size))
        #======================================================================
        #Set up of pandas dataframes
        #Create index
        Phy_string = '%.10g' % PhyTime
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
        #Create list of dataframes to be extracted by calling process
        DF_list = [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]
        
        #print(UU_tensorDF.loc[Phy_string,'uu'].values,flush=True)
        for DF in DF_list:
            DF.index.name = (self.par.array_start,self.par.array_end)
            
        return DF_list
    def AVG_flow_contour(self,flow_field,PhyTime,fig='',ax=''):
        """
        Creates contour plot primarily for testing the parallelisation of the code,
        Recommended to use the version in CHAPSim_post.py

        Parameters
        ----------
        flow_field : string
            string of [u,v,w,P,mag], whether to plot a component from the 
            DataFrame or calculate velocity magnitude.
        PhyTime : float
            Time to be postprocessed
        fig : matlab figure, optional
            Pre-created figures can be used for the post processing. The default is ''.
        ax : TYPE, optional
            Pre-created axes can be used for the post processing. The default is ''.

        Raises
        ------
        Exception
            Ensuring correct value are chosen flow_field variable.

        Returns
        -------
        fig : matplot figure
            figure with contour plot on it.
        ax : matplotlib axis
            axes which can be modified by calling process.

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        
        if type(PhyTime) == float:
            PhyTime = "{:.10g}".format(PhyTime)
        #Extracting data from dataframe
        if flow_field == 'u' or flow_field =='v' or flow_field =='w' or flow_field =='P':
            local_velo = self.flow_AVGDF.loc[PhyTime,flow_field].values
        elif flow_field == 'mag':
            index = pd.MultiIndex.from_arrays([[PhyTime,PhyTime,\
                                                PhyTime],['u','v','w']])
            local_velo = np.sqrt(np.square(self.flow_AVGDF.loc[index]).sum(axis=0)).values
        else:
            raise Exception
        #Resahpe array into the appropriate shape
        if self.par_dir == 'y':
            local_velo = local_velo.reshape(self.par.array_size,self.NCL[0])
        else:
            local_velo = local_velo.reshape(self.NCL[1],self.par.array_size)
        #Send arrays to the root process and concatenate to form output array
        if rank != 0:
            send_array = local_velo.copy()
            comm.send(send_array,dest=0,tag=70+rank)
        else:
            for i in range(1,size):
                recv_array=None
                recv_array = comm.recv(source=i,tag=70+i)
                if self.par_dir =='y':
                    local_velo =np.vstack((local_velo,recv_array))
                else:
                    local_velo =np.hstack((local_velo,recv_array))
        
        if rank == 0:
            #Post-processing on rank 0
            x_coords = self.CoordDF['x'].dropna().values
            y_coords = self.CoordDF['y'].dropna().values
            X, Y = np.meshgrid(x_coords,y_coords)
            #If fig and ax are not arguments to the functions create them
            if not fig:
                fig,ax = plt.subplots(figsize=[10,5])
            elif not ax:
                ax = fig.add_subplot(1,1,1) 
                
            ax1 = ax.pcolormesh(X,Y,local_velo,cmap='nipy_spectral')
            ax = ax1.axes
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            fig.colorbar(ax1,ax=ax)
            ax.set_xlabel("$x/\delta$",fontsize=18)
            ax.set_ylabel("$y/\delta$",fontsize=16)
        else:
            fig = None
            ax = None
        return fig, ax
    def fluct_contour_plot(self,comp,PhyTime,fig='',ax=''):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        #assert comp =='uu' or comp =='vv' or comp =='ww' or comp =='uw' or comp =='uv' or comp =='vw', ("Incorrect component selected: %s"%comp) 
        if type(PhyTime) == float:
            PhyTime = "{:.10g}".format(PhyTime)
        #Extracting data from dataframe
        
        UU = self.UU_tensorDF.loc[PhyTime,comp+comp].values
        U = self.flow_AVGDF.loc[PhyTime,comp].values
        local_velo = np.sqrt(UU-U*U)
        #Resahpe array into the appropriate shape
        if self.par_dir == 'y':
            local_velo = local_velo.reshape(self.par.array_size,self.NCL[0])
        else:
            local_velo = local_velo.reshape(self.NCL[1],self.par.array_size)
        #Send arrays to the root process and concatenate to form output array
        if rank != 0:
            send_array = local_velo.copy()
            comm.send(send_array,dest=0,tag=70+rank)
        else:
            for i in range(1,size):
                recv_array=None
                recv_array = comm.recv(source=i,tag=70+i)
                if self.par_dir =='y':
                    local_velo =np.vstack((local_velo,recv_array))
                else:
                    local_velo =np.hstack((local_velo,recv_array))
        
        if rank == 0:
            #Post-processing on rank 0
            x_coords = self.CoordDF['x'].dropna().values
            y_coords = self.CoordDF['y'].dropna().values
            X, Y = np.meshgrid(x_coords,y_coords)
            #If fig and ax are not arguments to the functions create them
            if not fig:
                fig,ax = plt.subplots(figsize=[10,5])
            elif not ax:
                ax = fig.add_subplot(1,1,1) 
                
            ax1 = ax.pcolormesh(X,Y,local_velo,cmap='nipy_spectral')
            ax = ax1.axes
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
            fig.colorbar(ax1,ax=ax)
            ax.set_xlabel("$x/\delta$",fontsize=18)
            ax.set_ylabel("$y/\delta$",fontsize=16)
        else:
            fig = None
            ax = None
        return fig, ax
class CHAPSim_autocov():
    def __init__(self,comp1,comp2,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True,root=0):
        """
        Class to create the the 

        Parameters
        ----------
        comp1 : string
            velocity component 1.
        comp2 : string
            velocity component 2.
        x_split_list : list, optional
            If present the x direction is split with the autocovariance being 
            calculated within each split. The default is ''.
        path_to_folder : string, optional
            path to the results folder. The default is ''.
        time0 : float, optional
            Initial time. The default is ''.
        abs_path : bool, optional
            Whether the path is absolute or relative. The default is True.
        homogen : bool, optional
            Whether to treat the flow as locally homogeneous in the streamwise 
            direction. The default is True.
        root : int, optional
            The root rank, must be less than the number of MPI processes. 
            The default is 0.

        Returns
        -------
        None.

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        #Extracts all the filenames from the results folder
        file_names = cp.time_extract(path_to_folder,abs_path)
        #Extracts the times from the file names
        time_list =[]
        for file in file_names:
            time_list.append(float(file[20:35]))
        #Removing duplicates
        times = list(dict.fromkeys(time_list))
        #Removes times below time0
        if time0:
            times = list(filter(lambda x: x >= time0, times))
        #times=times[:2]
            
        self.comp = (comp1,comp2)
        self._meta_data = cp.CHAPSim_meta(path_to_folder)
        self._par = parallel(self._meta_data.NCL[1])
        
        assert(root<size and root>=0) #root must be less than the number of processes and greater than 0
        if x_split_list:
            self.x_split_list = x_split_list
        try:
            self._AVG_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder,time0,abs_path,root=root)
        except Exception: #If a results file is still writing error can be caused hence this removes the largest value
            times_temp= times
            times_temp.remove(max(times))
            self._AVG_data = CHAPSim_AVG(max(times_temp),self._meta_data,path_to_folder,time0,root=root)
        i=1
        #Calculates the autocovariance for each time step and then performs an incremental average
        for timing in times:       
            self._inst_data = CHAPSim_Inst(timing,self._meta_data,path_to_folder,root=root)
            coe3 = (i-1)/i #Coefficents for the incremental average
            coe2 = 1/i
            if i==1: # if i is 1 autocorr assigned
                autocorr, index = self.__autocorr(timing,max(times),comp1,comp2,x_split_list,homogen)
            else: # performs incremental average above i=1
                local_autocorr, index = self.__autocorr(timing,max(times),comp1,comp2,homogen,x_split_list)
                assert(local_autocorr.shape == autocorr.shape)
                autocorr = autocorr*coe3 + local_autocorr*coe2
            if rank ==0:
                print("Completed loop %d of %d" %(i,len(times)),flush=True)
            i += 1
        autocorr = autocorr.T

        #Creates datafrmae
        self.autocorrDF = pd.DataFrame(autocorr,index=index)
        del self._inst_data #releases memory
    def __autocorr(self,time,AVG_time,comp1,comp2,homogen,x_split_list=''):
        """
        A functions to create the format of the pandas dataframe and calls 
        routines to determine the autocovariance

        Parameters
        ----------
        time : float
            time to be extracted from instDF.
        AVG_time : float
            To extract averaged data to enable the calculation of fluctuations.
        comp1 : string
            velocity component to be extracted for fluctuation 1.
        comp2 : string
            velocity component to be extracted for fluctuation 2.
        x_split_list : list, optional
            list of index of the splits in the x direction. The default is ''.
        homogen : bool, optional
            Whether to treat the flow as local homogeneous in the streamwise 
            direction. The default is True.

        Raises
        ------
        ValueError
            Ensures the values in x_split_list are not larger than the length 
            of the x direction.

        Returns
        -------
        autocorrDF.values : numpy array
            Values of the autocovariance in the appropriate form to be averaged 
            in the __init__ function
        index : list
            For the creation of the dataframe in the __init__ function.

        """
        if x_split_list:
            split_index = []
            direction_index = []
            for x in x_split_list:
                if x > self._meta_data.NCL[0]:
                    raise ValueError("value in x_split_list cannot be larger"\
                                     +"than x_size: %d, %d" %(x,self._meta_data.NCL[0]))
            #Create DataFrame for each split then concatenate
            for i in range(len(x_split_list)-1): 
                x_point1 = x_split_list[i]
                x_point2 = x_split_list[i+1]
                #Calculate the autocovariance
                autocorr_tempDF = self.__autocorr_calc(time,AVG_time,comp1,comp2,x_point1, x_point2,homogen)
                if i==0:
                    autocorrDF = autocorr_tempDF   
                else:
                    concatDF =[autocorrDF,autocorr_tempDF]
                    autocorrDF = pd.concat(concatDF, axis=1)
                #Create index incrementally through each part of the loop
                split_list = ['Split ' + str(i+1),'Split ' + str(i+1)]
                direction_list = ['x','z']
                split_index.extend(split_list)
                direction_index.extend(direction_list)
            index = [split_index,direction_index]
                
        else: #if there is no x_split_list
            x_point1 = 0
            x_point2 = self._meta_data.NCL[0]
            autocorrDF = self.__autocorr_calc(time,AVG_time,comp1,comp2,x_point1, x_point2)
            index = ['x','z']
        return autocorrDF.values, index
    def __autocorr_calc(self,PhyTime,AVG_time,comp1,comp2,x_point1, x_point2,homogen):
        """
        Function to calculate the autocovariance includes discrimination between 
        homogeneous and non-homogeneous averages in the streamwise direction. 
        Uses numba to accelerate loop heavy component.

        Parameters
        ----------
        PhyTime : float
            time to be extracted from CHAPSim_Inst class
        AVG_time : float
            time to be extracted from CHAPSim_AVG class. Corresponds to max(times)
            in __init__
        comp1 : string
            velocity component for fluctuation representing the start of the 
            two-point correlation.
        comp2 : string
            velocity component for fluctuation representing the end point of the 
            two-point correlation.
        x_point1 : int
            Allows the use of the x_split_list giving a restricted view of the
            fluctuation arrays simplifying loops. Provides the view start point
        x_point2 : int
            Allows the use of the x_split_list giving a restricted view of the
            fluctuation arrays simplifying loops. Provides the view end point.
        homogen : bool, optional
            Whether the xsplits or the whole domain should be treated as locally
            homogeneous in the streamwise direction. The default is True.

        Returns
        -------
        autocorr : Pandas DataFrame
            DataFrame of the correlation for a particular x_split (which is the 
            whole domain if not x_split_list has been provided).

        """
        # comm = MPI.COMM_WORLD
        # rank=comm.rank
        if type(PhyTime) == float:
            PhyTime = "{:.10g}".format(PhyTime)
        if type(AVG_time) == float:
            AVG_time = "{:.10g}".format(AVG_time)
        
        NCL_local = self._meta_data.NCL
        #Calculate the fluctuations 
        velo1 = self._inst_data.InstDF.loc[PhyTime,comp1].values.\
                    reshape((NCL_local[2],self._par.array_size,NCL_local[0]))
        AVG1 = self._AVG_data.flow_AVGDF.loc[AVG_time,comp1].values\
                    .reshape((self._par.array_size,NCL_local[0]))
        fluct1 = np.zeros_like(velo1)
        for i in range(NCL_local[2]):
            fluct1[i] = velo1[i] - AVG1
        
        velo2 = self._inst_data.InstDF.loc[PhyTime,comp2].values.\
                    reshape((NCL_local[2],self._par.array_size,NCL_local[0]))
        AVG2 = self._AVG_data.flow_AVGDF.loc[AVG_time,comp2].values\
                    .reshape((self._par.array_size,NCL_local[0]))
        fluct2 = np.zeros_like(velo2)
        
        for i in range(NCL_local[2]):
            fluct2[i] = velo2[i] - AVG2
        #Restrict range of fluctuation to the relevant part of the array
        fluct1_0 = fluct1[:,:,x_point1:x_point2]
        fluct2_0 = fluct2[:,:,x_point1:x_point2]
        
        #determining size of the x and z direction. Half because the maximum 
        #separation is the same length
        x_size = int(np.trunc(0.5*fluct1_0.shape[2]))
        z_size = int(np.trunc(0.5*fluct2_0.shape[0]))
        R_x = np.zeros((self._par.array_size,x_size))

        R_z = np.zeros((self._par.array_size,z_size))

        #t0 = time.time()
        #Using external funcion due to use of acceleration with the numba module
        if homogen:
            R_x = _loop_accelerator_x(fluct1_0,fluct2_0,R_x,x_size)
            R_z = _loop_accelerator_z(fluct1_0,fluct2_0,R_z,z_size)
            R_x = R_x/(fluct2_0.shape[0]*x_size) #Divide by the number of values
            R_z = R_z/(fluct1_0.shape[2]*z_size)
        else:
            R_x = _loop_accelerator_x_non_homogen(fluct1_0,fluct2_0,R_x,x_size)
            R_z = _loop_accelerator_z_non_homogen(fluct1_0,fluct2_0,R_z,z_size)
            R_x = R_x/(fluct1_0.shape[0])
            R_z = R_z/(z_size)
        #t1 = time.time()
        
        #Reshape into appropriate size for the DataFrame
        R_x = R_x.reshape((self._par.array_size*x_size))
        R_z = R_z.reshape(z_size*self._par.array_size)
        #Create DataFrame from easy concatenation in the calling routines
        R_xDF = pd.DataFrame(R_x)
        R_zDF = pd.DataFrame(R_z)
        #Concatenate the z and x autocovariances into one dataFrame
        autocorr = pd.concat([R_xDF,R_zDF],axis=1)
        #print(rank,t1-t0,flush=True)
        return autocorr
    def autocorr_contour(self,comp,Y_plus=False,Y_plus_max ='', which_split='',norm=True,fig='',ax=''):
        """
        Produces a contour plot of the autocovariance (which can be normalised 
        for the ACF). Can account for whether a restricted view is desired in 
        terms of the maximum Y+ value 

        Parameters
        ----------
        comp : string
            Can be x or z. The direction of the autocorrelation to be retrived 
            from DataFrame.
        Y_plus : bool, optional
            Whether the wall units are used in the  y direction. 
            The default is False.
        Y_plus_max : float, optional
            Gives the maximum Y+ value to be printed. The default is ''.
        which_split : int, optional
            If there is an x_split_list which split to be printed by the function.
            If not present and there is an x_split_list attribute than all are
            printed. The default is ''.
        norm : bool, optional
            Whether autocovariance is normalised to produce the ACF. The default 
            is True.
        fig : matplotlib figure, optional
            If the plot is to be added to a pre-existing figure. The default is ''.
        ax : TYPE, optional
            If the plot is to be added to a pre-existing axis. The default is ''.

        Returns
        -------
        fig, ax on the root rank (0), None otherwise.

        """
        comm = MPI.COMM_WORLD
        rank=comm.rank
        size_mpi=comm.Get_size()
        if rank ==0:
            print("###  PLOTTING CONTOUR OF AUTOCORRELATION  of %s%s ###" % self.comp, flush=True)
        assert(comp=='x' or comp =='z') #Only component x and z computed
        if Y_plus_max and Y_plus == False: 
            warnings.warn("Rank %d: ignoring `Y_plus_max' value: Y_plus == False" %rank)
        NCL_local = self._meta_data.NCL.copy()
        NCL_local[1] = self._par.array_size
        if (hasattr(self,'x_split_list') and which_split) or not hasattr(self,'x_split_list'):
            if which_split:
                assert(type(which_split)==int)
                split_string = "Split "+ str(which_split)
                #Calculating sizes
                x_point2 = self.x_split_list[which_split]
                x_point1 = point1 = self.x_split_list[which_split-1]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                    #Extract autocovariance
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
                fig,ax = plt.subplots(figsize=[10,5])
            elif not ax:
                ax = fig.add_subplot(1,1,1)
                
            #Extracting coordinates for the plot and Y+ if necessary
            coord = self._meta_data.CoordDF[comp].copy().dropna()\
                    .values[point1:point1+size]
            y_coord = self._meta_data.CoordDF['y'].copy().dropna()\
                    .values
            if Y_plus:
                #Calculating Y+ 
                avg_time = self._AVG_data.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                u_tau_star, delta_v_star = wall_unit_calc(self._AVG_data,avg_time)
                y_coord = y_coord[:int(y_coord.size/2)]
                y_coord = (1-np.abs(y_coord))/delta_v_star[point1]
#==============================================================================
#               THIS SECTION CAN BE SIMPLIFIED 
                if Y_plus_max:
                    #Telling the root rank which is the final sending rank
                    y_coord = y_coord[y_coord<Y_plus_max]
                    if len(y_coord)>=self._par.array_start:
                        if len(y_coord)<self._par.array_end:
                            Ruu = Ruu[:(len(y_coord)-self._par.array_start)]
                            comm.send(rank,dest=0,tag=66)
                        elif rank==0:
                            stoprank=comm.recv(tag=66)
                    else:
                        Ruu=None
                else:
                    if y_coord.size>=self._par.array_start:
                        if y_coord.size<self._par.array_end:
                            Ruu = Ruu[:(y_coord.size-self._par.array_start)]
                            comm.send(rank,dest=0,tag=66)
                        elif rank==0:
                            stoprank=comm.recv(tag=66)
                    else:
                        Ruu=None
#==============================================================================
            else: # If  Y+ isn't used  the stoprank is the final rank
                    stoprank=size_mpi-1
            if rank > 0 and Ruu is not None: # Sending Ruu to root rank if rank <= stoprank
                comm.send(Ruu,dest=0,tag=(50+rank))
            elif rank==0: #Root rank receiving arrays
                for i in range(1,stoprank+1):
                    recv_array=None
                    recv_array=comm.recv(source=i,tag=50+i)
                    if comp=='z': #Array concatenation on root
                        Ruu = np.vstack((Ruu,recv_array))
                            
                    else:
                        Ruu = np.vstack((Ruu,recv_array))
                        if i==size-1:
                            Ruu=Ruu.T
                
            
                X,Y = np.meshgrid(coord,y_coord)
                ax1 = ax.pcolormesh(X,Y,Ruu,cmap='nipy_spectral')
                ax = ax1.axes
                if Y_plus:
                    ax.set_ylabel(r"$Y^{+0}$", fontsize=16)
                else:
                    ax.set_ylabel(r"$y/\delta$", fontsize=16)
                ax.set_xlabel(r"$\Delta %s/\delta$" %comp, fontsize=18)
                fig.colorbar(ax1,ax=ax)
        elif hasattr(self,'x_split_list') and not which_split:
            if rank==0:
                # This results in complex plots as result it overides the provided
                #fig and ax
                if fig or ax: 
                    warnings.warn("fig and ax are overridden in this case for all processes")
                ax_size = len(self.x_split_list)-1
                fig,ax = plt.subplots(ax_size,figsize=[8,ax_size*2.6])
            else:
                if not fig:
                    fig = None
                if not ax:
                    ax = None
            point1 = 0
            #Looping over all the splits
            for i in range(1,len(self.x_split_list)):
                #Extracting autocov DataFrame
                split_string = "Split "+ str(i)
                #Determining array sizes
                if comp =='x':
                    x_point2 = self.x_split_list[i]   
                    x_point1 = point1 = self.x_split_list[i-1]
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(0.5*NCL_local[2]))
                #Extracting array from DataFrame
                Ruu = self.autocorrDF.loc[split_string,comp].dropna().values\
                        .reshape((NCL_local[1],size)) 
                #Noramlise if this is indicated
                if norm:
                    Ruu_0 = Ruu[:,0].copy()
                    for j in range(size):
                        Ruu[:,j] = Ruu[:,j]/Ruu_0
                #Extracting coordinates 
                coord = self._meta_data.CoordDF[comp].dropna()\
                        .values[point1:point1+size]
                y_coord = self._meta_data.CoordDF['y'].dropna()\
                        .values
                        
                if Y_plus: 
                    #Calculating Y+
                    avg_time = self._AVG_data.flow_AVGDF.index[0][0]
                    # Calculating viscous lengthscales
                    u_tau_star, delta_v_star = wall_unit_calc(self._AVG_data,avg_time)
                    y_coord = y_coord[:int(y_coord.size/2)]
                    #Y+ should be 0 at the wall, ycoord varies from -1 to 1 \delta
                    y_coord = (1-np.abs(y_coord))/delta_v_star[point1]
                    if Y_plus_max:
#==============================================================================
                        #To be simplified like above
                        #potentially bugs related to stoprank being 0 - rank 0 cannot send to 0
                        #Future testing to be done on a single process
                        y_coord = y_coord[y_coord<Y_plus_max] #boolean indexing to extract Y+<Y+max
                        if len(y_coord)>=self._par.array_start: #Whether rank needs to be extracted
                            if len(y_coord)<self._par.array_end:
                                # determine relevant part of Ruu on the final rank
                                Ruu = Ruu[:(len(y_coord)-self._par.array_start)] 
                                comm.send(rank,dest=0,tag=66) #send the stop rank to root
                            elif rank==0:
                                stoprank=comm.recv(tag=66)
                        else:
                            Ruu=None #Ruu not required above stoprank
                    else:
                        if int(y_coord.size)>=self._par.array_start:
                            if int(y_coord.size)<self._par.array_end:
                                Ruu = Ruu[:(int(y_coord.size)-self._par.array_start)]
                                comm.send(rank,dest=0,tag=66)
                            elif rank==0:
                                stoprank=comm.recv(tag=66)
                        else:
                            Ruu=None
#==============================================================================                        
                else: #If y+ not present stoprank is the largest rank
                    stoprank=size_mpi-1
                if rank > 0 and Ruu is not None: 
                    comm.send(Ruu,dest=0,tag=(50+rank))
                elif rank==0:
                    for j in range(1,stoprank+1):
                        recv_array=None
                        #print(j,flush=True)
                        recv_array=comm.recv(source=j,tag=50+j)
                        if comp=='z':
                            #print(Ruu.shape,recv_array.shape,flush=True)
                            Ruu = np.vstack((Ruu,recv_array))
                                
                        else:
                            Ruu = np.vstack((Ruu,recv_array))
                            if i==size-1:
                                Ruu=Ruu.T
                    X,Y = np.meshgrid(coord,y_coord)
                    ax1 = ax[i-1].pcolormesh(X,Y,Ruu,cmap='nipy_spectral')
                    ax[i-1] = ax1.axes
                    if i==len(self.x_split_list)-1:
                        ax[i-1].set_xlabel(r"$\Delta %s/\delta$" %comp, fontsize=18)
                        
                    if Y_plus:
                        ax[i-1].set_ylabel(r"$Y^{+0}$", fontsize=16)
                    else:
                        ax[i-1].set_ylabel(r"$y/\delta$", fontsize=16)
                    fig.colorbar(ax1,ax=ax[i-1])
            if rank == 0:
                fig.subplots_adjust(hspace = 0.4)
        
        return fig,ax
    def Spectra_calc(self,comp,y_values,which_split='',fig='',ax=''):
        """
        Create spectra along particular line in the autocovariance plane

        Parameters
        ----------
        comp : string
            Can be x or z. The direction of the autocorrelation to be retrieved 
            from DataFrame.
        y_values : int
            index in the y direction to be extracted.
        which_split : int, optional
            If there is an x_split_list which split to be printed by the function.
            If not present and there is an x_split_list attribute than all are
            printed. The default is ''.
        fig : matplotlib figure, optional
            If the plot is to be added to a pre-existing figure. The default is ''.
        ax : TYPE, optional
            If the plot is to be added to a pre-existing axis. The default is ''.

        Returns
        -------
        fig : matplotlib figure
            Figure containg the axes with the spectra on them.
        ax : matplotlib axis
            axis containing the spectrum plots, allows for external modification 
            of axis labels, title etc..

        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        NCL_local = self._meta_data.NCL
        NCL_local[1] = self._par.array_size
        if rank == 0:
            print("###  PLOTTING SPECTRUM of %s%s ###"% self.comp,flush=True)
            if not fig:
                fig,ax = plt.subplots()
            elif not ax:
                ax = fig.add_subplot(1,1,1)
        else:
            fig = None ; ax = None
        
        if (hasattr(self,'x_split_list') and which_split) or not hasattr(self,'x_split_list'):
            if which_split:
                #Much of this is commented similarly in the above funtions
                assert(type(which_split)==int)
                split_string = "Split "+ str(which_split)
                x_point2 = self.x_split_list[which_split]
                x_point1 = point1 = self.x_split_list[which_split-1]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(NCL_local[2]))
                if y_values >= self._par.array_start and y_values<self._par.array_end:
                    Ruu = self.autocorrDF.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    if rank != 0:
                        comm.send(Ruu,dest=0,tag=77)
                elif rank ==0:
                    Ruu = None
                    Ruu = comm.recv(tag=77)

            else:
                x_point1 = point1 = 0
                x_point2 = NCL_local[0]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(NCL_local[2]))
                if y_values >= self._par.array_start and y_values<self._par.array_end:
                    Ruu = self.autocorrDF.loc[comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    if rank != 0:
                        comm.send(Ruu,dest=0,tag=77)
                elif rank ==0:
                    Ruu = None
                    Ruu = comm.recv(tag=77)
            if rank ==0:    
                #Calculate FFT of the autocovariance to find the spectrum
                wavenumber_spectra = fftpack.rfft(Ruu)
                coord = self._meta_data.CoordDF[comp].dropna()\
                            .values[point1:point1+size]
                #Calculate chnge in spacing, note that the 1/\Delta x is constant
                delta_comp = coord[1]-coord[0]
                #Calculate sample frequency
                Fs = (2.0*np.pi)/delta_comp
                comp_size= wavenumber_spectra.size
                #Calclate wavenumber distribution
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

                if y_values >= self._par.array_start and y_values<self._par.array_end:
                    Ruu = self.autocorrDF.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    if rank != 0:
                        comm.send(Ruu,dest=0,tag=77)
                elif rank ==0:
                    Ruu = None
                    Ruu = comm.recv(tag=77)
                if  rank == 0:    
                    wavenumber_spectra = fftpack.rfft(Ruu)
                    coord = self._meta_data.CoordDF[comp].dropna()\
                            .values[point1:point1+size]
                    delta_comp = coord[1]-coord[0]
                    Fs = (2.0*np.pi)/delta_comp
                    comp_size= wavenumber_spectra.size
                    wavenumber_comp = np.arange(comp_size)*Fs/comp_size
                
                    ax.plot(wavenumber_comp,np.abs(wavenumber_spectra))
                    #ax.plot(wavenumber_comp,wavenumber_comp**(-5/3))
                    if i == len(self.x_split_list)-1:
                        x_coord = self._meta_data.CoordDF['x'].dropna()\
                            .values[self.x_split_list[:-1]]
                        ax.legend([r"$x/\delta = %.3f$" %x for x in x_coord])
        if rank ==0:
            ax.set_xlabel(r"$\kappa_%s$"%comp,fontsize=18)
            ax.set_ylabel(r"$E(\kappa_%s)$"%comp,fontsize=16)
            ax.grid()
            ax.set_xscale('log')
            #ax.set_yscale('log')
        return fig, ax
        

                
                
class CHAPSim_Ruu(CHAPSim_autocov):
    """ 
    An instance of the parent class with components ('u','u').
    For help see use help(CHAPSim_autocov) the parent class
    """
    def __init_(self,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True,root=0):
        super().__init__('u','u',x_split_list,path_to_folder,time0,abs_path,homogen,root)
class CHAPSim_Rvv(CHAPSim_autocov):
    """ 
    An instance of the parent class with components ('v','v').
    For help see use help(CHAPSim_autocov) the parent class
    """
    def __init_(self,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True,root=0):
        super().__init__('v','v',x_split_list,path_to_folder,time0,abs_path,homogen,root)
class CHAPSim_Rww(CHAPSim_autocov):
    """ 
    An instance of the parent class with components ('w','w').
    For help see use help(CHAPSim_autocov) the parent class
    """
    def __init_(self,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True,root=0):
        super().__init__('w','w',x_split_list,path_to_folder,time0,abs_path,homogen,root)
class CHAPSim_Ruv(CHAPSim_autocov):
    """ 
    An instance of the parent class with components ('u','v').
    For help see use help(CHAPSim_autocov) the parent class
    """
    def __init_(self,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True,root=0):
        super().__init__('u','v',x_split_list,path_to_folder,time0,abs_path,homogen,root)
class CHAPSim_k_spectra(CHAPSim_autocov):
    def __init__(self,x_split_list='',path_to_folder='',time0='',abs_path=True,homogen=True,root=0):
        """
        Calls __init__ of the parent class for each of the normal terms in the 
        two-point correlation tensor <u_i(x1)u_j(x1 + r)>, sums and divides by 2

        For more details of the parent class use help(CHAPSim_autocov

        """
        
        super().__init__('u','u',x_split_list,path_to_folder,time0,abs_path,homogen,root)
        self.autocorr_uu = self.autocorrDF
        super().__init__('v','v',x_split_list,path_to_folder,time0,abs_path,homogen,root)
        self.autocorr_vv = self.autocorrDF
        super().__init__('w','w',x_split_list,path_to_folder,time0,abs_path,homogen,root)
        self.autocorr_ww = self.autocorrDF
    def Spectra_calc(self,comp,y_values,which_split='',fig='',ax=''):
        """
        Computes and plots the energy spectrum along particular line in the
        autocovariance plane

        Parameters
        ----------
       comp : string
            Can be x or z. The direction of the autocorrelation to be retrieved 
        y_values : int
            index in the y direction to be extracted.
        which_split : int, optional
            If there is an x_split_list which split to be printed by the function.
            If not present and there is an x_split_list attribute than all are
            printed. The default is ''.
        fig : matplotlib figure, optional
            If the plot is to be added to a pre-existing figure. The default is ''.
        ax : TYPE, optional
            If the plot is to be added to a pre-existing axis. The default is ''.

        Returns
        -------
        fig : matplotlib figure
            Figure containg the axes with the spectra on them.
        ax : matplotlib axis
            axis containing the spectrum plots, allows for external modification 
            of axis labels, title etc..

        """
        #Much of this function is the same as the for the equivalent in the 
        #parent class with the exception of extracting the three normal autocorrelation 
        #components
        
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        NCL_local = self._meta_data.NCL
        NCL_local[1] = self._par.array_size
        if rank == 0:
            if not fig:
                fig,ax = plt.subplots()
            elif not ax:
                ax = fig.add_subplot(1,1,1)
        else:
            fig = None ; ax = None
        
        if (hasattr(self,'x_split_list') and which_split) or not hasattr(self,'x_split_list'):
            if which_split:
                assert(type(which_split)==int)
                split_string = "Split "+ str(which_split)
                x_point2 = self.x_split_list[which_split]
                x_point1 = point1 = self.x_split_list[which_split-1]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(NCL_local[2]))
                if y_values >= self._par.array_start and y_values<self._par.array_end:
                    Ruu = self.autocorr_uu.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    Rvv = self.autocorr_vv.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    Rww = self.autocorr_ww.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    if rank != 0:
                        comm.send(Ruu,dest=0,tag=77)
                        comm.send(Rvv,dest=0,tag=88)
                        comm.send(Rww,dest=0,tag=99)
                elif rank ==0:
                    Ruu = None ; Rvv = None ; Rww = None
                    Ruu = comm.recv(tag=77)
                    Rvv = comm.recv(tag=88)
                    Rww = comm.recv(tag=99)

            else:
                x_point1 = point1 = 0
                x_point2 = NCL_local[0]
                if comp =='x':
                    size = int(np.trunc(0.5*(x_point2-x_point1)))
                else:
                    size = int(np.trunc(NCL_local[2]))
                if y_values >= self._par.array_start and y_values<self._par.array_end:
                    Ruu = self.autocorr_uu.loc[comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    Rvv = self.autocorr_vv.loc[comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    Rww = self.autocorr_ww.loc[comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    if rank != 0:
                        comm.send(Ruu,dest=0,tag=77)
                        comm.send(Rvv,dest=0,tag=88)
                        comm.send(Rww,dest=0,tag=99)
                elif rank ==0:
                    Ruu = None ; Rvv = None ; Rww = None
                    Ruu = comm.recv(tag=77)
                    Rvv = comm.recv(tag=88)
                    Rww = comm.recv(tag=99)
            if rank ==0:      
                Phi_uu = fftpack.rfft(Ruu)
                Phi_vv = fftpack.rfft(Rvv)
                Phi_ww = fftpack.rfft(Rww)
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

                if y_values >= self._par.array_start and y_values<self._par.array_end:
                    Ruu = self.autocorr_uu.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    Rvv = self.autocorr_vv.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    Rww = self.autocorr_ww.loc[split_string,comp].copy().dropna().values\
                        .reshape((NCL_local[1],size))[y_values-self._par.array_start]
                    if rank != 0:
                        comm.send(Ruu,dest=0,tag=77)
                        comm.send(Rvv,dest=0,tag=88)
                        comm.send(Rww,dest=0,tag=99)
                elif rank ==0:
                    Ruu = None ; Rvv = None ; Rww = None
                    Ruu = comm.recv(tag=77)
                    Rvv = comm.recv(tag=88)
                    Rww = comm.recv(tag=99)
                if  rank == 0:    
                    
                    Phi_uu = fftpack.rfft(Ruu)
                    Phi_vv = fftpack.rfft(Rvv)
                    Phi_ww = fftpack.rfft(Rww)
                    wavenumber_spectra = 0.5*(Phi_uu + Phi_vv + Phi_ww)
                    coord = self._meta_data.CoordDF[comp].dropna()\
                            .values[point1:point1+size]
                    delta_comp = coord[1]-coord[0]
                    Fs = (2.0*np.pi)/delta_comp
                    comp_size= wavenumber_spectra.size
                    wavenumber_comp = np.arange(comp_size)*Fs/comp_size
                
                    ax.plot(wavenumber_comp,np.abs(wavenumber_spectra))
                    if i == len(self.x_split_list)-1:
                        x_coord = self._meta_data.CoordDF['x'].dropna()\
                            .values[self.x_split_list[:-1]]
                        ax.legend([r"$x/\delta = %.3f$" %x for x in x_coord])
                    #ax.plot(wavenumber_comp,wavenumber_comp**(-5/3))
        if rank ==0:
            ax.set_xlabel(r"$\kappa_%s$"%comp,fontsize=18)
            ax.set_ylabel(r"$E(\kappa_%s)$"%comp,fontsize=16)
            ax.grid()
            ax.set_xscale('log')
            #ax.set_yscale('log')
        return fig, ax
class CHAPSim_uv_pdf():
    """ Class for creating and plotting the joint probability distribution of u and v"""
    def __init__(self,xy_indices,u_points,v_points,path_to_folder='',abs_path=True,time0='',root=0):
        """
        Initialises class to create joint probability distribution

        Parameters
        ----------
        xy_indices : list of tuples
            Pairs of x,y indices where the JPD will be calculated.
        u_points : int
            Starting number of points in the u axis, this will increase.
        v_points : int
            Starting number of points in the v axis, this will increase..
        path_to_folder : string, optional
            Results location. The default is the current directory.
        abs_path : bool, optional
            Whether the provided path is absolute or not. The default is True.
        time0 : float or string, optional
            Starting time point for the averaged calculation. The default is ''.
        root : int, optional
            The root MPI rank. The default is 0.

        Returns
        -------
        None.

        """
        
        comm=MPI.COMM_WORLD
        size=comm.Get_size()
        rank=comm.rank
        assert(root<size and root>=0)
        self.__root=root
        if rank ==self.__root:
            print("###  CREATING JOINT PROBABILITY DISTRIBUTIONS  ###\n",flush=True)
            file_names = cp.time_extract(path_to_folder,abs_path)
            #Extraction of times and removal of duplicates
            time_list =[]
            for file in file_names:
                time_list.append(float(file[20:35]))
            times = list(dict.fromkeys(time_list))
            if time0:
                times = list(filter(lambda x: x > time0, times))
            #times=times[:10]
        else:
            times=None
        #Broadcasting data from root rank
        self._meta_data=cp.CHAPSim_meta(path_to_folder)
        times=comm.bcast(times,root=self.__root)
        NCLX = self._meta_data.CoordDF['y'].dropna().values.size
        #Setting up the parallelising the y direction
        self._par=parallel(NCLX)
        
        #Extracting averaged data into class CHAPSim_AVG
        try:
            self._AVG_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder=path_to_folder,\
                                         time0=time0,par_dir='y',root=self.__root)
        except Exception: #if exception raised if a file is read while being written 
            times_temp= times
            times_temp.remove(max(times))
            self._AVG_data = CHAPSim_AVG(max(times_temp),self._meta_data,path_to_folder=path_to_folder,\
                                         time0=time0,par_dir='y',root=self.__root)
        
        NCL=self._AVG_data.NCL
        time_string = "{:.10g}".format(max(times))
        U_AVG = self._AVG_data.flow_AVGDF.loc[time_string,'u'].values\
                            .reshape((self._par.array_size,NCL[0]))
        V_AVG = self._AVG_data.flow_AVGDF.loc[time_string,'v'].values\
                            .reshape((self._par.array_size,NCL[0]))
        UU = self._AVG_data.UU_tensorDF.loc[time_string,'uu'].values\
                            .reshape((self._par.array_size,NCL[0]))
        VV = self._AVG_data.UU_tensorDF.loc[time_string,'vv'].values\
                            .reshape((self._par.array_size,NCL[0]))
        UV = self._AVG_data.UU_tensorDF.loc[time_string,'uv'].values\
                            .reshape((self._par.array_size,NCL[0]))
        
        #Averaged velocity for determining the fluctuation
        u_avg=[] 
        v_avg=[]
        #For calculating the covariance matrix (for mahalanobis distance FUTURE)
        uu_list=[] 
        vv_list = []
        uv_list = []
        # xy_coord=[]
        #So every rank has the averaged velocity values at all xy combinations specified
        self.xy_indices = xy_indices
        for xy in xy_indices:
            y_coord = xy[1]-self._par.array_start
            if y_coord <self._par.array_size and y_coord>=0: # if rank contains pair `xy'
                u_avg.append(U_AVG[y_coord,xy[0]])
                v_avg.append(V_AVG[y_coord,xy[0]])
                uu = UU[y_coord,xy[0]] - U_AVG[y_coord,xy[0]]*U_AVG[y_coord,xy[0]]
                vv = VV[y_coord,xy[0]] - V_AVG[y_coord,xy[0]]*V_AVG[y_coord,xy[0]]
                uv = UV[y_coord,xy[0]] - U_AVG[y_coord,xy[0]]*V_AVG[y_coord,xy[0]]
                uu_list.append(uu) #So list contains all values plus the new value
                vv_list.append(vv)
                uv_list.append(uv)
                # xy_coord.append((xy[0],xy[1]))
                if rank != 0: #if not root send entire list to root
                    comm.send(u_avg,dest=0,tag=66)
                    comm.send(v_avg,dest=0,tag=67)
                    comm.send(uu_list,dest=0,tag=68)
                    comm.send(vv_list,dest=0,tag=69)
                    comm.send(uv_list,dest=0,tag=70)
            elif rank ==0: # if root rank doesn't contain pair receive from sending rank
                u_avg = comm.recv(tag=66)
                v_avg = comm.recv(tag=67)
                uu_list = comm.recv(tag=68)
                vv_list = comm.recv(tag=69)
                uv_list = comm.recv(tag=70)
            #Broadcast current list to all ranks before repeating for rest of xy_indices
            u_avg = comm.bcast(u_avg,root=0)
            v_avg = comm.bcast(v_avg,root=0)
            uu_list = comm.bcast(uu_list,root=0)
            vv_list = comm.bcast(vv_list,root=0)
            uv_list = comm.bcast(uv_list,root=0)
        #Parallelise the for length of xy_indices so each rank has different JPDs to calculate
        self._par_var_list = parallel(len(xy_indices))
        u_avg = u_avg[self._par_var_list.array_start:self._par_var_list.array_end].copy()
        v_avg = v_avg[self._par_var_list.array_start:self._par_var_list.array_end].copy()
        uu_list = uu_list[self._par_var_list.array_start:self._par_var_list.array_end].copy()
        vv_list = vv_list[self._par_var_list.array_start:self._par_var_list.array_end].copy()
        uv_list = uv_list[self._par_var_list.array_start:self._par_var_list.array_end].copy()
        
        #Calculate covariances locally on each rank
        cov_list = []
        for i in range(len(uu_list)):
            cov_list.append(np.array([[uu_list[i],uv_list[i]],[uv_list[i],vv_list[i]]]))
        #Decrement reference count to free memory
        del uu_list ;  del vv_list ;  del uv_list ; del UU ; del VV ; del UV
        del U_AVG ; del V_AVG ; del self._AVG_data
        gc.collect()
        comm.barrier()
        #Calculate local xy_coords to be determined
        xy_coord = xy_indices[self._par_var_list.array_start:self._par_var_list.array_end]
        i=0
        #Initialising arrays
        pre_CDF_list = [None]*len(xy_coord)
        u_vals =  [None]*len(xy_coord)
        v_vals = [None]*len(xy_coord)
        
        if rank ==0:
            print("Entering Extraction loop!!",flush=True)
        for timing in times:
            #Extracting instantaneous data for each time in times
            self._inst_data = CHAPSim_Inst(timing,self._meta_data,path_to_folder=path_to_folder,\
                                         par_dir="y",root=self.__root)
            U_inst = self._inst_data.InstDF.loc["{:.10g}".format(timing),'u'].values\
                            .reshape((NCL[2],self._par.array_size,NCL[0]))
            V_inst = self._inst_data.InstDF.loc["{:.10g}".format(timing),'v'].values\
                            .reshape((NCL[2],self._par.array_size,NCL[0]))
            
            #Creating lists of the instantaneous data on all ranks - similar process to above
            u_inst_list=[]
            v_inst_list=[]
            for xy in xy_indices:
                y_coord = xy[1]-self._par.array_start
                if y_coord <self._par.array_size and y_coord>=0:
                    u_inst_list.append(U_inst[:,y_coord,xy[0]])
                    v_inst_list.append(V_inst[:,y_coord,xy[0]])
                    if rank != 0:
                        comm.send(u_inst_list,dest=0,tag=66)
                        comm.send(v_inst_list,dest=0,tag=67)
                elif rank ==0:
                    u_inst_list = comm.recv(tag=66)
                    v_inst_list = comm.recv(tag=67)
                u_inst_list = comm.bcast(u_inst_list,root=0)
                v_inst_list = comm.bcast(v_inst_list,root=0)
            #index list for local JPD to be calculated
            u_inst_list = u_inst_list[self._par_var_list.array_start:self._par_var_list.array_end]
            v_inst_list = v_inst_list[self._par_var_list.array_start:self._par_var_list.array_end]
            
            #Calculate fluctuations if rank has values to calculate
            #There will be no values to calculate if rank id exceeds the length of xy_indices-1
            if self._par_var_list.array_size != 0: 
                fluct_uv_tuple = self.__fluct_calc(u_inst_list, v_inst_list, u_avg, v_avg)
                
                #Calculate pre-CDF and the u and v velocity arrays
                pre_CDF_list, u_vals, v_vals = self.__CDF_calc(fluct_uv_tuple, pre_CDF_list, u_points, v_points,
                                                          u_vals,v_vals)
            
            i+=1
            if rank ==0:
                print("Completed %d of %d" % (i,len(times)),flush=True)
            gc.collect()
            comm.barrier()
            #print(i,flush=True)
        del u_inst_list ; del v_inst_list ; del U_inst ; del V_inst
        gc.collect()
        comm.barrier()
        if self._par_var_list.array_size != 0:
            del fluct_uv_tuple #decrement reference
            No_elements = len(times)*NCL[2]
            #print(pre_CDF_list,flush=True)
            for i in range(len(pre_CDF_list)):
                #Normalising so CDF at max u and max v is 1
                pre_CDF_list[i] = pre_CDF_list[i]/No_elements
                
            #Calculating Joint probability distribution
            JPD_list = self.__JPD_calc(pre_CDF_list, u_vals, v_vals)
            #Creating index for DataFrame
            xy_index = [str(x) for x in xy_coord]
            
            self.Joint_pdf_DF = pd.DataFrame(np.stack(JPD_list),index=xy_index)
            
            self.u_valsDF = pd.DataFrame(np.stack(u_vals),index = xy_index)
            self.v_valsDF = pd.DataFrame(np.stack(v_vals),index = xy_index)
            
        else: #If rank had no xy vals to calculate object equals None
            self.Joint_pdf_DF=None
            self.u_valsDF=None
            self.v_valsDF=None
    
        
            
    def __fluct_calc(self,u_inst_z_list,v_inst_z_list,u_avg_xy_list,v_avg_xy_list):
        """ Calculates the velocity function of u and v at each timestep 
            returning a list of tuples for each point in the z direction        
        """
        fluct_u_list =[]
        #Calculating u fluctuation for each xy combination creating list of ndarrays
        fluct_u = np.zeros_like(u_inst_z_list[0])
        for u_inst, u_avg in zip(u_inst_z_list,u_avg_xy_list):
            for i in range(u_inst.size):
                fluct_u[i] = u_inst[i] - u_avg
            fluct_u_list.append(fluct_u)
            
        fluct_v_list =[]
        #Calculating v fluctuation for each xy combination creating list of ndarrays
        fluct_v = np.zeros_like(v_inst_z_list[0])
        for v_inst, v_avg in zip(v_inst_z_list,v_avg_xy_list):
            for i in range(v_inst.size):
                fluct_v[i] = v_inst[i] - v_avg
            fluct_v_list.append(fluct_v)
            
        #Create list of a list of tuples, where each tuple is a u,v pair, outer 
        #list is each x,y pair and inner list for each z in x,y pair
        fluct_uv_tuple = []
        for fluct_u, fluct_v in zip(fluct_u_list,fluct_v_list):
            fluct_uv_tuple.append(list(zip(fluct_u,fluct_v)))
        
            
        return fluct_uv_tuple
        
    def __CDF_calc(self,fluct_uv_tuple,CDF_master,u_points_start,v_points_start,\
                   u_vals,v_vals):
        """ Calculate the cumulative distribution function for each xy conbination
            MORE DETAIL TO BE ADDED LATER
        
        """

        for k in range(len(fluct_uv_tuple)):
            if u_vals[k] is not None: #Not the first run through
                #Calculates the maxima and minima from previous passes through function
                u_max_old = u_vals[k][-1]
                v_max_old = v_vals[k][-1]
                u_min_old = u_vals[k][0]
                v_min_old = v_vals[k][0]
            else:
                #Ensures that the first pass always results in u_min,u_max etc. being changed 
                u_max_old = -float("inf") ; v_max_old= -float("inf")
                u_min_old = float("inf") ; v_min_old = float("inf")
            #determining new values for the current time
            u_max = max(max([x[0] for x in fluct_uv_tuple[k]]),u_max_old)
            u_min = min(min([x[0] for x in fluct_uv_tuple[k]]),u_min_old)
            v_max = max(max([x[1] for x in fluct_uv_tuple[k]]),v_max_old)
            v_min = min(min([x[1] for x in fluct_uv_tuple[k]]),v_min_old)
            if u_vals[k] is None:#First run through
                #Determines u_inc/v_inc remains approx constant throughout
                u_inc = (u_max-u_min)/u_points_start
                v_inc = (v_max-v_min)/v_points_start
                #widens range to ensure all values are caught in first pass
                u_min -= u_inc
                v_min -= v_inc
                u_max += u_inc
                v_max += v_inc
                #Not needed
                old_u_min = old_v_min =  float("inf")
                old_u_max = old_v_max = -float("inf")
            else: #Subsequent passes
                #Recalculates u_inc/ vinc from the previous u_vals array
                u_inc = (u_max_old-u_min_old)/CDF_master[k].shape[0]
                v_inc = (v_max_old-v_min_old)/CDF_master[k].shape[1]
                
                #Determines the changes required for minima and maxima in terms of u_inc/v_inc
                u_min_delta = int(np.floor((u_min_old-u_min)/u_inc))
                u_min=u_min_old - (u_min_delta+1)*u_inc
                u_max_delta = int(np.ceil(u_max-u_max_old)/u_inc)
                u_max = u_max_old + (u_max_delta+1)*u_inc 
                
                v_min_delta = int(np.floor((v_min_old-v_min)/v_inc))
                v_min = v_min_old - (v_min_delta+1)*v_inc
                v_max_delta = int(np.ceil(v_max-v_max_old)/v_inc)
                v_max = v_max_old + (v_max_delta+1)*v_inc
                
                #calculates the indices where the old CDF array will fit into the new one
                old_u_min = int(np.floor((u_min_old-u_min)/u_inc))
                old_v_min = int(np.floor((v_min_old-v_min)/v_inc))
                old_v_max = old_v_min + CDF_master[k].shape[1]
                old_u_max = old_u_min + CDF_master[k].shape[0]
                
            #Recalculating u_vals and v_vals based on quantities from this pass
            u_vals[k] = np.arange(u_min,u_max,u_inc,dtype='f8')
            v_vals[k] = np.arange(v_min,v_max,v_inc,dtype='f8')
            
            CDF = np.zeros((u_vals[k].size,v_vals[k].size),dtype='i4') #Initialising CDF
            
            #On subsequent pass place old CDF array inside new one and propagate 
            #the values to subsequent indices
            if CDF_master[k] is  not None:
                CDF[old_u_min:old_u_max,old_v_min:old_v_max]= CDF_master[k]
                if CDF_master[k] is not None:
                    for i in range(old_v_max,v_vals[k].size):
                        CDF[old_u_min:old_u_max,i] = CDF_master[k][:,-1]
                    for i in range(old_u_max,u_vals[k].size):
                        CDF[i,old_v_min:old_v_max] = CDF_master[k][-1,:]
                    CDF[old_u_max:,old_v_max:] = CDF_master[k][-1,-1]
            
            #Adding to CDF for values on the new pass
            for i in range(CDF.shape[1]):
                #Calculating the list of the values of u whose pairing v meets the condition
                local_u = np.array([x[0] for x in fluct_uv_tuple[k] if x[1]<=v_vals[k][i]])
                for j in range(CDF.shape[0]):
                    #The pre_CDF is the size of the resulting array
                    CDF[j,i] += local_u[local_u<=u_vals[k][j]].size
            #Check values are physical
            self.__CDF_check(CDF, CDF.shape[0], CDF.shape[1])        
            CDF_master[k] = CDF.copy()

        return CDF_master, u_vals, v_vals

    def __CDF_check(self,CDF,u_size,v_size):
        """ Ensures that all CDF values are physical, i.e they must increase 
            with increasing u and v """
        err = 1e-6
        for i in range(1,u_size):
            for j in range(1,v_size):
                try:
                    assert(CDF[i,j]+err>=CDF[i-1,j] and CDF[i,j]+err>=CDF[i,j-1])
                except AssertionError:
                    traceback.print_stack()
                    raise ValueError("The value of the cumulative distribution function" \
                                     + "must always increase or the stay the same size with" \
                                     + " increasing index Indices (%d,%d), PreCDF has values %g %g %g" \
                                         % (i,j,CDF[i,j],CDF[i-1,j],CDF[i,j-1]))
                        
    def __mahalanobis_calc(self,cov,fluct_uv_tuple):
        """ Calculates the mahalanobis distance currently unused may be added 
            to scale the plot """
        mahal_list = []
        
        for fluct_uv in fluct_uv_tuple:
            mahal=[]
            for uv in fluct_uv:
                uv_array = np.array(uv)
                mahal.append(np.sqrt((uv_array.T).dot(cov).dot(uv_array)))
            mahal_list.append(mahal)
        return mahal_list
    
    def __uv_lim_calc(self,cov,fluct_uv_tuple,u_vals,v_vals):
        mahal_list = self.__mahalanobis_calc(cov,fluct_uv_tuple)
        
        for i in range(len(mahal_list)):
            for l in range(len(mahal_list[i])):
                pass
    def __JPD_calc(self,CDF_master,u_vals,v_vals):
        """ Computes the Joint probability distribution function by calculating 
            the mixed derivative of the CDF wrt u and v"""
        PDF = [None]*len(CDF_master)
        for k in range(len(CDF_master)):
            PDF[k] = d_CDF_du = np.zeros_like(CDF_master[k])
            for i in range(CDF_master[k].shape[1]):
                d_CDF_du[:,i] = CT.Gen_Grad_calc(u_vals[k],CDF_master[k][:,i])
            for j in range(CDF_master[k].shape[0]):
                PDF[k][j,:] = CT.Gen_Grad_calc(v_vals[k],d_CDF_du[j,:])
            PDF[k]=PDF[k].reshape((CDF_master[k].shape[1]*CDF_master[k].shape[0]))
        return PDF
    def Joint_pdf_plot(self,coord_xy_list=''):
        """
        Plots the joint probability distribution

        Parameters
        ----------
        coord_xy_list : list of tuples, optional
            The list of x,y pairs to be postprocessed. The default is all the 
            pairs that have been calculated.

        Returns
        -------
        fig : matplotlib figure
            Figure which JPD on
        ax : matplotlib axis
            Axis with JPD on.

        """
        comm=MPI.COMM_WORLD
        rank=comm.rank
        size=comm.Get_size()
        if rank ==0:
            print("###  PLOTTING JOINT PROBABILITY DISTRIBUTIONS  ###",flush=True)
        if self.Joint_pdf_DF is None:
            fig=None ; ax = None
            return fig, ax
            
        if not coord_xy_list:
            coord_xy_list = self.xy_indices
        
        coord_xy_list = [str(x) for x in coord_xy_list]
        local_xy_coord = self.Joint_pdf_DF.index
        extract_xy = [x for x in local_xy_coord if x in coord_xy_list]
        
        #Extract from DataFrame the relevant JPDs
        extractDF = self.Joint_pdf_DF.loc[extract_xy]
        uvals_extract = self.u_valsDF.loc[extract_xy]
        vvals_extract = self.v_valsDF.loc[extract_xy]
        #Getting all the DataFrames onto the root rank for postprocessing
        if  rank !=0: 
            comm.send(extractDF,dest=0, tag=1)
            comm.send(uvals_extract,dest=0, tag=2)
            comm.send(vvals_extract,dest=0, tag=3)
            
        else:
            #Determining stop_rank for when size>len(xy_indices)
            if len(self.xy_indices) < size:
                stop_rank = len(self.xy_indices)
            else:
                stop_rank=size
            for i in range(1,stop_rank):
                #print(i,stop_rank)
                recvDF = None
                recvDF = comm.recv(source=i,tag=1)
                extractDF = pd.concat([extractDF,recvDF])
                recvDF = comm.recv(source=i,tag=2)
                uvals_extract = pd.concat([uvals_extract,recvDF])
                recvDF = comm.recv(source=i,tag=3)
                vvals_extract = pd.concat([vvals_extract,recvDF])
                #print(extractDF,flush=True)
                
        if rank ==0:
            fig, ax = plt.subplots(len(coord_xy_list),figsize=[10,3*len(coord_xy_list)])
            i=0
            for coord_xy in coord_xy_list: # A different subplot for each xy pair selected for post processing
                extract = extractDF.loc[coord_xy].dropna().values
                u_vals = uvals_extract.loc[coord_xy].dropna().values
                v_vals = vvals_extract.loc[coord_xy].dropna().values
                extract= extract.reshape((u_vals.size,v_vals.size))
                u_mesh, v_mesh = np.meshgrid(u_vals,v_vals)
                
        
                if len(coord_xy_list) > 1: #To account for behaviour of subplots which multiple subplots
                    #Plotting contour plot of the JPD
                    ax1 = ax[i].pcolormesh(u_mesh,v_mesh, extract.T,cmap='YlGn')
                    ax[i] = ax1.axes
                    ax[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
                    ax[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator('auto'))
                    fig.colorbar(ax1,ax=ax[i])
                    if i == len(coord_xy_list)-1:
                        ax[i].set_xlabel(r"$u'/U_b$",fontsize=18)
                    ax[i].set_ylabel(r"$v'/U_b$",fontsize=16)
                    
                else:
                    #needs to be updated
                    CS = ax.contour(u_mesh,v_mesh, extract.T,extent=(min(u_vals),
                                                                        max(u_vals),min(v_vals),max(v_vals)))
                    ax.clabel(CS,inline=1,fontsize=10)
                    
                i+=1
        else:
            fig = None ; ax = None
            
        return fig, ax
                    
class CHAPSim_Quad_Anal():
    """ Class to calculate quadrant analysis using Willmarth and Lu 1972"""
    def __init__(self,prop_direction,other_coord_list,h_list,path_to_folder='',time0='',abs_path=True,root=0):
        """
        Initialises self producing a process distributed spatially parallelised
        DataFrame containing uv_q for each quadrant along a line in the direcion 
        of prop_direction at index in the other direction specified by other_coord_list

        Parameters
        ----------
        prop_direction : string
            propagation direction for the quadrant analysis, either 'x' or 'y'.
        other_coord_list : int or list of int
            index where lines in the propagation direction are taken.
        h_list : float or list of floats
            Value used as PDF for Willmarth-Lu method I_n = 1 if uv > h*u_rms*v_rms, 0 otherwise.
        path_to_folder : string, optional
            path to the results folder. The default is '' meaning current directory.
        time0 : float or string, optional
            Initial time between which the average is calculated time0<time. 
            The default is '' meaning no lower bound on the average.
        abs_path : bool, optional
            Whether the given path is an absolute or relative path.
            The default is True.
        root : int, optional
            which rank will be root during the calling of class CHAPSim_Inst 
            and CHAPSim_AVG. The default is 0.

        Raises
        ------
        ValueError
            Ensures other_coord_list, h_list are the appropriate type.

        Returns
        -------
        None.

        """
        comm=MPI.COMM_WORLD
        rank=comm.rank
        size=comm.Get_size()
        assert(root<size and root>=0)
        self.__root=root
        assert(prop_direction=='x' or prop_direction=='y')
        if rank == self.__root:
            #Extraction of files
            print("###  BEGINNING QUADRANT ANALYSIS  ###\n",flush=True)
            file_names = cp.time_extract(path_to_folder,abs_path)
            #Extraction of times and removal of duplicates
            time_list =[]
            for file in file_names:
                time_list.append(float(file[20:35]))
            times = list(dict.fromkeys(time_list))
            if time0:
                times = list(filter(lambda x: x > time0, times))
            #times=times[:2]
            if isinstance(other_coord_list,int): #if just int is given converts to single element list
                other_coord_list=[other_coord_list]
            elif not isinstance(other_coord_list,list):
                raise ValueError("other_coord_list must be list or int")
                
            if isinstance(h_list,float): #if just float is given converts to single element list
                h_list=[h_list]
            elif not isinstance(h_list,list):
                raise ValueError("h_list must be list or float")
            coord_h_list=[]
            for coord in other_coord_list:
                for h in h_list:
                    coord_h_list.append((coord,h))
            
        else:
            coord_h_list=None
            times=None
        #Broadcast variables
        coord_h_list=comm.bcast(coord_h_list,root=self.__root)
        self.coord_h_list=coord_h_list
        self._meta_data=cp.CHAPSim_meta(path_to_folder)
        times=comm.bcast(times,root=self.__root)
        #Parallelisation in the propagation direction
        NCLX = self._meta_data.CoordDF[prop_direction].dropna().values.size
        self._par=parallel(NCLX)
        self._prop_dir=prop_direction
        #Creating average class at the latest time in the results
        try:
            self._AVG_data = CHAPSim_AVG(max(times),self._meta_data,path_to_folder=path_to_folder,\
                                         time0=time0,par_dir=self._prop_dir,root=self.__root)
        except Exception:
            times_temp= times
            times_temp.remove(max(times))
            self._AVG_data = CHAPSim_AVG(max(times_temp),self._meta_data,path_to_folder=path_to_folder,\
                                         time0=time0,par_dir=self._prop_dir,root=self.__root)
        i=1
        if rank == self.__root:
            print("Entering Extraction Loop!!",flush=True)
        #Performing incremental average for the quadrant analysis assuming 
        #homogeneous in z direction and stationary between max(times) and time0
        for timing in times:
            #Extracted instanteous data
            self._inst_data=CHAPSim_Inst(timing,self._meta_data,path_to_folder=path_to_folder,\
                                         par_dir=self._prop_dir,root=self.__root)
            NCL=self._inst_data.NCL
            #Extracting DataFrames at different coords in other_coord_list of (u-\bar{u})(v-\bar{v})
            #u_rms*v_rms at each line from other_coord_list and DataFrame of which quadrant each fluct_uv
            fluct_uv_DF, uv_rmsDF, QuadDF, uvDF = self._fluct_extract(self._inst_data,self._AVG_data,\
                             other_coord_list,timing,max(times),prop_direction)
        
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1: # creates array on the first pass
                #Creates array at different coords and values of h for each quadrant at each point along each line
                quad_anal, index = self.__quad_anal_calc(fluct_uv_DF,uv_rmsDF,QuadDF,uvDF,coord_h_list,NCL,prop_direction)
            else: # increntally averages array on subsequent passes
                quad_anal_local, index = self.__quad_anal_calc(fluct_uv_DF,uv_rmsDF,QuadDF,uvDF,coord_h_list,NCL,prop_direction)
                assert(quad_anal.shape == quad_anal_local.shape)
                quad_anal = quad_anal*coe3 + quad_anal_local*coe2
            
            if rank==self.__root:
                print("Completed loop %d of %d" % (i,len(times)),flush=True)
            i+=1
        #Creates DataFrame based on index and arrays produced by __quad_anal_calc
        self.QuadAnalDF=pd.DataFrame(quad_anal,index=pd.MultiIndex.from_arrays(index))
        del self._inst_data #free memory
        
    def __quad_anal_calc(self,fluct_uv_DF,uv_rmsDF,QuadDF,uvDF,coord_h_list,NCL_local,prop_dir):
        """
        Calculates the quadrant analysis according to the Willmarth Lu 1972

        Parameters
        ----------
        fluct_uv_DF : pandas DataFrame
            DataFrame with the (u-\bar{u})(v-\bar{v}) in the planes specified by 
            other_coord_list and prop_direction.
        uv_rmsDF : pandas DataFrame
            DataFrame with lines specified by specified by other_coord_list and prop_direction
            of u_rms*v_rms.
        QuadDF : pandas DataFrame
            DataFrame with the quadrants associated with fluct_uv_DF in the 
            planes specified by other_coord_list and prop_direction.
        coord_h_list : list of tuples
            Associated with all the possible conbinations of other_coord and h.
        NCL_local : numpy array
            Array with coordinate dimensions, this would vary depending 
            on parallelisation direction
        prop_dir : string
            Propagation direction.

        Returns
        -------
        uv_qDF.values : numpy array
            array at a particular time step of the quadrant analysis
        [coord_index,quadrant_index] : list
            index to be used on the DataFrame self.QuadAnalDF in __init__.

        """
        array_size_par = self._par.array_end -self._par.array_start
        
        array_size=(NCL_local[2],array_size_par)
        
        
        uv_q=np.zeros((4,array_size_par))
        quadrant_index =[]
        coord_index=[]
        uv_qDF =None
        for coord_h in coord_h_list:
            coord=coord_h[0]
            h=coord_h[1] 
            
            #Extracting the plane and line associated with coord
            fluct_plane = fluct_uv_DF.loc[coord].values\
                .reshape(array_size)
            rms_plane = uv_rmsDF.loc[coord].values
            for j in range(1,5):
                #return boolean array corresponding whether location is part of quadrant
                quad_array=QuadDF.loc[coord].values\
                    .reshape(array_size) == j
               
                uv_array = uvDF.loc[coord].values
                for i in range(array_size_par):
                    #Multiplication by boolean array results in 0 when false
                    #Hence will only sum those greater than h*rms_plane[i] within the quadrant
                    fluct_array = np.abs(fluct_plane[quad_array[:,i],i]) >= h*rms_plane[i]
                    #print(fluct_plane)
                    quant=np.sum(fluct_plane[quad_array[:,i],i]*fluct_array)#/uv_array[i]
                    #print(np.sign(fluct_plane[:,i]))
                    uv_q[j-1,i]=quant/(NCL_local[2]*uv_array[i]) #Normalise by number of elements in z dir
                coord_index.append((coord,h))
            quadrant_index.extend(range(1,5))
            uv_qDF_temp=pd.DataFrame(uv_q) # create DataFrame of particular coord h combo
            if uv_qDF is not None:
                uv_qDF=pd.concat([uv_qDF,uv_qDF_temp]) # concatenate fo all combos
            else:
                uv_qDF=uv_qDF_temp.copy()
            
        return uv_qDF.values, [coord_index,quadrant_index]
                
    def _fluct_extract(self,inst_data,AVG_data,other_coord_list,PhyTime,AVG_time,prop_dir):
        
        NCL_local=inst_data.NCL
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)
        if type(AVG_time) == float:
            AVG_time = "{:.10g}".format(AVG_time)
        
        array_size_par = self._par.array_size
        #Extracting and reshaping necessary arrays
        if self._prop_dir=='y':
            array_size_inst=(NCL_local[2],array_size_par,NCL_local[0])
            array_size_avg = (array_size_par,NCL_local[0])
        else:
            array_size_inst=(NCL_local[2],NCL_local[1],array_size_par)
            array_size_avg = (NCL_local[1],array_size_par)
            
        u_velo_inst=inst_data.InstDF.loc[PhyTime,'u'].values\
                    .reshape(array_size_inst)
        v_velo_inst=inst_data.InstDF.loc[PhyTime,'v'].values\
                    .reshape(array_size_inst)  
        u_velo_avg=AVG_data.flow_AVGDF.loc[AVG_time,'u'].values\
                    .reshape(array_size_avg)  
        v_velo_avg=AVG_data.flow_AVGDF.loc[AVG_time,'v'].values\
                    .reshape(array_size_avg)
        U2_velo=AVG_data.UU_tensorDF.loc[AVG_time,'uu'].values\
                    .reshape(array_size_avg)
        V2_velo=AVG_data.UU_tensorDF.loc[AVG_time,'vv'].values\
                    .reshape(array_size_avg)
        
        UV_velo = AVG_data.UU_tensorDF.loc[AVG_time,'uv'].values\
                    .reshape(array_size_avg)
                    
        fluct_u=np.zeros_like(u_velo_inst)
        fluct_v=np.zeros_like(v_velo_inst)
        #Computing mean square velocity
        u_rms2=U2_velo - u_velo_avg*u_velo_avg
        v_rms2=V2_velo - v_velo_avg*v_velo_avg
        uv = UV_velo - u_velo_avg*v_velo_avg
        for i in range(NCL_local[2]):
            #Calculating velocity fluctuations
            fluct_u[i] = u_velo_inst[i] - u_velo_avg
            fluct_v[i] = v_velo_inst[i] - v_velo_avg
        del u_velo_inst ; del v_velo_inst ; del u_velo_avg ; del v_velo_avg
        del U2_velo ; del V2_velo ; del UV_velo
        #Different propagation directions have different array shapes
        if prop_dir=='x':
            fluct_u_plane = fluct_u[:,other_coord_list,:]
            fluct_v_plane = fluct_v[:,other_coord_list,:]
            del fluct_u ; del fluct_v
            #To determine which quadrant each uv is in with array indicating sign 
            fluct_u_isneg = fluct_u_plane < 0
            fluct_v_isneg = fluct_v_plane < 0
            #Square root and slice
            u_rms_plane = np.sqrt(u_rms2[other_coord_list,:])
            v_rms_plane = np.sqrt(v_rms2[other_coord_list,:])
            
            uv_plane = uv[other_coord_list,:]
            del u_rms2 ; del v_rms2 ; del uv
            
            quadrant_array = np.zeros_like(fluct_v_isneg,dtype='i4')
            
            for i in range(1,5): #determining quadrant
                if i ==1:
                    quadrant_array_temp = np.logical_and(~fluct_u_isneg,~fluct_v_isneg)#not fluct_u_isneg and not fluct_v_isneg
                    quadrant_array += quadrant_array_temp*1
                elif i==2:
                    quadrant_array_temp = np.logical_and(fluct_u_isneg,~fluct_v_isneg)#not fluct_u_isneg and fluct_v_isneg
                    quadrant_array += quadrant_array_temp*2
                elif i==3:
                    quadrant_array_temp =  np.logical_and(fluct_u_isneg,fluct_v_isneg)
                    quadrant_array += quadrant_array_temp*3
                elif i==4:
                    quadrant_array_temp =  np.logical_and(~fluct_u_isneg,fluct_v_isneg)#fluct_u_isneg and not fluct_v_isneg
                    quadrant_array += quadrant_array_temp*4
            assert(quadrant_array.all()<=4 and quadrant_array.all()>=1)    
            
            output_fluct = np.zeros((len(other_coord_list),NCL_local[2],array_size_par))    
            output_quadrant = np.zeros_like(output_fluct,dtype='i4')
            for j in range(len(other_coord_list)):
                #Rearranging and multiplying fluct_u*fluct_v
                output_fluct[j,:,:] = fluct_u_plane[:,j,:]*fluct_v_plane[:,j,:]   
                output_quadrant[j,:,:] = quadrant_array[:,j,:] #rearranging
            del fluct_u_plane ; del fluct_v_plane
            #Reshaping and determing output_rms
            output_fluct = output_fluct.reshape((len(other_coord_list),NCL_local[2]*array_size_par))
            output_rms=u_rms_plane*v_rms_plane
            del u_rms_plane ; del v_rms_plane
            output_quadrant = output_quadrant.reshape((len(other_coord_list),NCL_local[2]*array_size_par))
        else:
            #Similar to above
            fluct_u_plane = fluct_u[:,:,other_coord_list]
            fluct_v_plane = fluct_v[:,:,other_coord_list]
            del fluct_u ; del fluct_v
            
            fluct_u_isneg = fluct_u_plane < 0
            fluct_v_isneg = fluct_v_plane < 0
            u_rms_plane = np.sqrt(u_rms2[:,other_coord_list])
            v_rms_plane = np.sqrt(v_rms2[:,other_coord_list])
            uv_plane = uv[:,other_coord_list]
            del u_rms2 ; del v_rms2
            
            output_fluct = np.zeros((len(other_coord_list),NCL_local[2],array_size_par))  
            
            quadrant_array = np.zeros_like(fluct_v_isneg,dtype='i4')
            
            for i in range(1,5):
                if i ==1:
                    quadrant_array_temp = np.logical_and(~fluct_u_isneg,~fluct_v_isneg)#not fluct_u_isneg and not fluct_v_isneg
                    quadrant_array += quadrant_array_temp*1
                elif i==2:
                    quadrant_array_temp = np.logical_and(fluct_u_isneg,~fluct_v_isneg)#not fluct_u_isneg and fluct_v_isneg
                    quadrant_array += quadrant_array_temp*2
                elif i==3:
                    quadrant_array_temp =  np.logical_and(fluct_u_isneg,fluct_v_isneg)
                    quadrant_array += quadrant_array_temp*3
                elif i==4:
                    quadrant_array_temp =  np.logical_and(~fluct_u_isneg,fluct_v_isneg)#fluct_u_isneg and not fluct_v_isneg
                    quadrant_array += quadrant_array_temp*4
            assert(quadrant_array.all()<=4 and quadrant_array.all()>=1)     
            
            output_rms = np.zeros((len(other_coord_list),array_size_par))
            output_quadrant = np.zeros_like(output_fluct,dtype='i4')
            for j in range(len(other_coord_list)):
                output_fluct[j,:,:] = fluct_u_plane[:,:,j]*fluct_v_plane[:,:,j]
                output_rms[j,:] = u_rms_plane[:,j]*v_rms_plane[:,j]
                uv_plane[j,:] = uv_plane[:,j]
                output_quadrant[j,:,:]=quadrant_array[:,:,j]
            del u_rms_plane ; del v_rms_plane
            #Reshaping
            output_fluct = output_fluct.reshape((len(other_coord_list),NCL_local[2]*array_size_par))    
            output_quadrant = output_quadrant.reshape((len(other_coord_list),NCL_local[2]*array_size_par))
        
        #Creating DataFrames
        fluctDF = pd.DataFrame(output_fluct,index=other_coord_list)
        rmsDF=pd.DataFrame(output_rms,index=other_coord_list)
        quadrantDF=pd.DataFrame(output_quadrant,index=other_coord_list)
        uvDF = pd.DataFrame(uv_plane,index=other_coord_list)
        gc.collect()
        return fluctDF, rmsDF, quadrantDF, uvDF
    def line_plot(self,coord_h_list=''):
        """
        Create line plot(s) with the different coord values of different axes 
        and h on the same axes

        Parameters
        ----------
        coord_h_list : list of tuples, optional
            If present will only extract the info from the relevant DataFrames
            else it will extract all. The default is ''.

        Returns
        -------
        fig : matplotlib figure
            Figure containing the axes 
        ax : matplotlib axis
            axes which contain the plots.

        """
        comm=MPI.COMM_WORLD
        rank=comm.rank
        size=comm.Get_size()
        if rank ==0:
            print("### PLOTTING QUADRANT ANALYSIS ###",flush=True)
        if not coord_h_list: #coord_h_list is not given
            coord_h_list=self.coord_h_list
        #Extracting coords and creating ordered sets of coord and h
        coord_prop=self._meta_data.CoordDF[self._prop_dir].dropna().values    
        coord_set=list(set([x[0] for x in coord_h_list]))
        h_set=list(set([x[1] for x in coord_h_list]))
        if rank ==0:
            fig,ax = plt.subplots(4,len(coord_set),figsize=[12,5*len(coord_set)])
            
            #print(coord_set,h_set)
        if self._prop_dir =='x':
            other_dir = 'y'
        else:
            other_dir = 'x'
        #Extract other coords
        coord_other = self._meta_data.CoordDF[other_dir].dropna().values    
        if other_dir =='y': #Use Y+ for legend if the propagation direction x 
            avg_time = self._AVG_data.flow_AVGDF.index[0][0]
            u_tau_star, delta_v_star = wall_unit_calc(self._AVG_data,avg_time,par_dir='x')
            coord_other = (1-np.abs(coord_other))/delta_v_star[0]
            
        j=0
        cum_sizes = tuple(self._par.array_sizes)
        start_points = np.cumsum(self._par.array_sizes)
        for k in range(size):
            start_points[k] -= self._par.array_sizes[k]
        start_points = tuple(start_points)
        #Outputting graph simplest way to understand this 
        for coord in coord_set:
            for h in h_set:
                QuadArray=self.QuadAnalDF.loc[(coord,h)].values
                for i in range(4):
                    quad_local = QuadArray[i].copy()  

                    if rank == 0:
                        concat_array=np.empty(np.sum(self._par.array_sizes))
                    else:
                        concat_array = None
                    #Use gather to cellect the separate 1-D arrays onto the root process
                    comm.Gatherv(quad_local,[concat_array,cum_sizes,start_points,MPI.DOUBLE])
                    if rank ==0:
                        #Plot quadrant analysis
                        ax[i,j].plot(coord_prop[10:-1],concat_array[10:-1])
                        if h==h_set[-1] and coord==coord_set[-1]:
                            ax[i,0].set_ylabel(r"Q%d"% (i+1),fontsize=16)
            if rank ==0:
                #Setting titles, lengends and labels
                coord_value=coord_other[coord]
                if other_dir == 'y':        
                    ax[0,j].set_title(r"$Y^{+0} = %.3f$" % coord_value,loc='right')
                else:
                    ax[0,j].set_title(r"$x/\delta = %.3f$" % coord_value,loc='right')
                ax[3,j].set_xlabel(r"$%s/\delta$"% self._prop_dir,fontsize=18)        
            j+=1
        if rank ==0:
            ax[0,0].legend(["h ="+str(x) for x in  h_set])
            fig.tight_layout()
        #ax.grid()
            
                
        else:
            fig, ax = (None,None)
            
        return fig, ax
                    
                        
@numba.jit(nopython=True)
def _loop_accelerator_x(fluct1,fluct2,R_x,x_size):
    for ix0 in range(x_size):
        for z in range(fluct1.shape[0]):
            for ix in range(x_size):
                R_x[:,ix0] += fluct1[z,:,ix]*fluct2[z,:,ix0+ix]
    return R_x
@numba.jit(nopython=True)
def _loop_accelerator_z(fluct1,fluct2,R_z,z_size):
    for iz0 in range(z_size):
        for ix in range(fluct1.shape[2]):
            for iz in range(z_size):
                R_z[:,iz0] += fluct1[iz,:,ix]*fluct2[iz+iz0,:,ix]
    return R_z
@numba.jit(nopython=True)
def _loop_accelerator_x_non_homogen(fluct1,fluct2,R_x,x_size):
    for ix0 in range(x_size):
        for z in range(fluct1.shape[0]):
            R_x[:,ix0] += fluct1[z,:,0]*fluct2[z,:,ix0]
    return R_x
@numba.jit(nopython=True)
def _loop_accelerator_z_non_homogen(fluct1,fluct2,R_z,z_size):
    for iz0 in range(z_size):
        for iz in range(z_size):
            R_z[:,iz0] += fluct1[iz,:,0]*fluct2[iz+iz0,:,0]
    return R_z
def wall_unit_calc(AVG_DF,PhyTime,par_dir='y'):
    comm = MPI.COMM_WORLD
    rank=comm.rank
    size= comm.Get_size()
    assert(par_dir =='x' or par_dir=='y')
    if par_dir =='y':
        if rank==0:
            y_coords = AVG_DF.CoordDF['y'].dropna().values
            NCL = AVG_DF.NCL
            
            u_array = AVG_DF.flow_AVGDF.loc[PhyTime,'u'].values.reshape((AVG_DF.par.array_size,NCL[0]))
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
        else:
            u_tau_star, delta_v_star = (None,None)
        u_tau_star = comm.bcast(u_tau_star,root=0)
        delta_v_star = comm.bcast(delta_v_star,root=0)
    else:
        y_coords = AVG_DF.CoordDF['y'].dropna().values
        NCL = AVG_DF.NCL
        
        u_array = AVG_DF.flow_AVGDF.loc[PhyTime,'u'].values.reshape((NCL[1],AVG_DF.par.array_size))
        #x_coords = coordsDF['x'].dropna().values
        
        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star
        umean_grad = np.zeros_like(u_array[1])
        REN = AVG_DF._metaDF.loc['REN'].values[0]
        
        wall_velo = AVG_DF._meta_data.moving_wall_calc()
        for i in range(AVG_DF.par.array_size):
            umean_grad[i] = mu_star*(u_array[0,i]-wall_velo[AVG_DF.par.array_start+i])/(y_coords[0]--1.0)
            #print(u_array[0,i])
        #The scaled viscosity \mu* is 1 for isothermal flow
        tau_w = umean_grad
        
        u_tau_star_local = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star_local = (nu_star/u_tau_star_local)/REN
        cum_sizes = tuple(AVG_DF.par.array_sizes)
        start_points = np.empty(size)
        for k in range(size):
            start_points[k] = np.cumsum(AVG_DF.par.array_sizes)[k] - AVG_DF.par.array_sizes[k]
        start_points = tuple(start_points)
        if rank == 0:
            u_tau_star=np.empty(np.sum(AVG_DF.par.array_sizes))
            delta_v_star=np.empty(np.sum(AVG_DF.par.array_sizes))
        else:
            u_tau_star = None
            delta_v_star = None
        comm.Gatherv(u_tau_star_local,[u_tau_star,cum_sizes,start_points,MPI.DOUBLE])
        comm.Gatherv(delta_v_star_local,[delta_v_star,cum_sizes,start_points,MPI.DOUBLE])
            
        delta_v_star = comm.bcast(delta_v_star,root=0)
        u_tau_star = comm.bcast(u_tau_star,root=0)
        
    return u_tau_star, delta_v_star
