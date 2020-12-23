"""
# _instant.py
File contains the implementation of the classes to processes the 
instantaneous results from the CHAPSim DNS solver 
"""
import numpy as np
import matplotlib as mpl

import sys
import os
import warnings
import gc

from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd
from .. import CHAPSim_plot as cplt

from ._meta import CHAPSim_meta
_meta_class=CHAPSim_meta

class CHAPSim_Inst():
    """
    ## CHAPSim_Inst
    This is a module for processing and visualising instantaneous data from CHAPSim
    """
    _module = sys.modules[__name__]
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
            meta_data = self._module._meta_class(path_to_folder,abs_path,tgpost)
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
                    # concat_DF = [self.InstDF,local_DF]
                    InstDF.concat(local_DF)
        else:
            raise TypeError("\033[1;32 `time' must be either float or list")
        shape = (NCL[2],NCL[1],NCL[0])
        return meta_data,CoordDF,NCL,InstDF, shape

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)
    
    def _hdf_extract(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Inst'
        meta_data = self._module._meta_class.from_hdf(file_name,base_name+'/meta_data')
        CoordDF=meta_data.CoordDF
        NCL=meta_data.NCL

        shape = (NCL[2],NCL[1],NCL[0])
        
        InstDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/InstDF')#pd.read_hdf(file_name,base_name+'/InstDF').data(shape)
        return meta_data, CoordDF, NCL, InstDF, shape

    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Inst'
        self._meta_data.save_hdf(file_name,write_mode,group_name=base_name+'/meta_data')
        self.InstDF.to_hdf(file_name,key=base_name+'/InstDF',mode='a')
    # @profile(stream=open("mem_check.txt",'w'))
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
        flow_info = self.__velo_interp(flow_info,NCL3,NCL2,NCL1)
        gc.collect()
        # flow_info1 = flow_info1.reshape((4,dummy_size-NCL3*NCL2))
        Phy_string = '%.9g' % PhyTime
        # creating dataframe index
        index = [[Phy_string]*4,['u','v','w','P']]
        # index=pd.MultiIndex.from_arrays([[Phy_string]*4,['u','v','w','P']])
        # creating dataframe so that data can be easily accessible elsewhere
        Instant_DF = cd.datastruct(flow_info,index=index,copy=False)# pd.DataFrame(flow_info1,index=index)

        for file in open_list:
            file.close()
            
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
        
        axis1_coords = self.CoordDF[axis1]
        axis2_coords = self.CoordDF[axis2]
        if flow_field == 'u' or flow_field =='v' or flow_field =='w' or flow_field =='P':
            local_velo = self.InstDF[PhyTime,flow_field]
        elif flow_field == 'mag':
            local_velo = np.sqrt(np.square(self.InstDF.values[:3]).sum(axis=0))
        else:
            raise ValueError("\033[1;32 Not a valid argument")
        
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
            fig,ax = cplt.subplots(**kwargs)
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
            fig = cplt.figure()
        if not ax:
            ax = fig.add_subplot(1,1,1)
        fig, ax = _vector_plot(self.CoordDF,self.InstDF[PhyTime],self.NCL,\
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
                velo_field1 = self.InstDF[PhyTime,velo1][:,:arr_len_y,x_start_index:x_end_index]
                velo_field2 = self.InstDF[PhyTime,velo2][:,:arr_len_y,x_start_index:x_end_index]
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
        # if PhyTime:
        #     if type(PhyTime) == float:
        #         PhyTime = "{:.9g}".format(PhyTime)
                
        # if len(set([x[0] for x in self.InstDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.InstDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.InstDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        
        
        if not hasattr(vals_list,'__iter__'):
            vals_list = [vals_list]
        X = self._meta_data.CoordDF['x']
        Y = self._meta_data.CoordDF['y']
        Z = self._meta_data.CoordDF['z']
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
        
        
        return fig, ax

    def vorticity_calc(self,PhyTime=''):
        # if PhyTime:
        #     if type(PhyTime) == float:
        #         PhyTime = "{:.9g}".format(PhyTime)
                
        # if len(set([x[0] for x in self.InstDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.InstDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.InstDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        vorticity = np.zeros((3,*self.shape),dtype='f8')
        u_velo = self.InstDF[PhyTime,'u']
        v_velo = self.InstDF[PhyTime,'v']
        w_velo = self.InstDF[PhyTime,'w']

        vorticity[0] = Grad_calc(self.CoordDF,w_velo,'y',False) - Grad_calc(self.CoordDF,v_velo,'z',False)      
        vorticity[1] = Grad_calc(self.CoordDF,u_velo,'z',False) - Grad_calc(self.CoordDF,w_velo,'x',False)      
        vorticity[2] = Grad_calc(self.CoordDF,v_velo,'x',False) - Grad_calc(self.CoordDF,u_velo,'y',False)     

        # vorticity = vorticity.reshape(3,np.prod(self.shape))
        return cd.datastruct(vorticity,index=['x','y','z'])

    def plot_vorticity_contour(self,comp,slice,ax_val,PhyTime=None,avg_data=None,y_mode='half_channel',pcolor_kw=None,fig=None,ax=None,**kwargs):
        # if PhyTime is not None:
        #     if type(PhyTime) == float:
        #         PhyTime = "{:.9g}".format(PhyTime)
                
        # if len(set([x[0] for x in self.InstDF.index])) == 1:
        #     fluct_time = list(set([x[0] for x in self.InstDF.index]))[0]
        #     if PhyTime is not None and PhyTime != fluct_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
        #     PhyTime = fluct_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.InstDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        # if slice not in ['xy','zy','xz']:
        #     slice = slice[::-1]
        #     if slice not in ['xy','yz','xz']:
        #         msg = "The contour slice must be either %s"%['xy','yz','xz']
        #         raise KeyError(msg)
        
        # slice = list(slice)
        # slice_set = set(slice)
        # coord_set = set(list('xyz'))
        # coord = "".join(coord_set.difference(slice_set))

        # if not hasattr(ax_val,'__iter__'):
        #     if isinstance(ax_val,float) or isinstance(ax_val,int):
        #         ax_val = [ax_val] 
        #     else:
        #         msg = "ax_val must be an interable or a float or an int not a %s"%type(ax_val)
        #         raise TypeError(msg)

        # if coord == 'y':
        #     norm_vals = [0]*len(ax_val)
        #     if avg_data is None:
        #         msg = f'For contour slice {slice}, avg_data must be provided'
        #         raise ValueError(msg)
        #     axis_index = CT.y_coord_index_norm(avg_data,self.CoordDF,ax_val,norm_vals,y_mode)
        # else:
        #     axis_index = CT.coord_index_calc(self.CoordDF,coord,ax_val)
        #     if not hasattr(axis_index,'__iter__'):
        #         axis_index = [axis_index]

        if not hasattr(ax_val,'__iter__'):
            if isinstance(ax_val,float) or isinstance(ax_val,int):
                ax_val = [ax_val] 
            else:
                msg = "ax_val must be an interable or a float or an int not a %s"%type(ax_val)
                raise TypeError(msg)

        slice, coord,axis_index = CT.contour_plane(slice,ax_val,avg_data,y_mode,PhyTime)

        if fig is None:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [8,4*len(ax_val)]
            else:
                warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(ax_val),**kwargs)
        elif ax is None:
            ax=fig.subplots(len(ax_val),squeeze=False)      
        ax = ax.flatten()

        VorticityDF = self.vorticity_calc(PhyTime=PhyTime)
        vort_array = VorticityDF[comp]

        coord_1 = self.CoordDF[slice[0]]
        coord_2 = self.CoordDF[slice[1]]
    
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        ax_out=[]
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        for i in range(len(ax_val)):
            contour_array = CT.contour_indexer(vort_array,axis_index[i],coord)
            # if coord == 'x':
            #     contour_array = vort_array[:,:,axis_index[i]].squeeze().T
            # elif coord == 'y':
            #     contour_array = vort_array[:,axis_index[i]].squeeze()
            # else:
            #     contour_array = vort_array[axis_index[i]].squeeze()
            mesh = ax[i].pcolormesh(coord_1_mesh,coord_2_mesh,contour_array,**pcolor_kw)
            ax[i].set_xlabel(r"$%s^*$"%slice[0])
            ax[i].set_ylabel(r"$%s$"%slice[1])
            ax[i].set_title(r"$%s = %.3g$"%(coord,ax_val[i]))
            cbar=fig.colorbar(mesh,ax=ax[i])
            cbar.set_label(r"$\omega_%s$"%comp)# ,fontsize=12)
            ax_out.append(mesh)

        return fig, np.array(ax_out)

    def plot_entrophy(self):
        pass
    def __str__(self):
        return self.InstDF.__str__()
    def __iadd__(self,inst_data):
        assert self.CoordDF.equals(inst_data.CoordDF), "CHAPSim_Inst are not from the same case"
        assert self.NCL == inst_data.NCL, "CHAPSim_Inst are not from the same case"

        self.InstDF.concat(inst_data.InstDF)
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
        
        NCL1_io = meta_data.metaDF['NCL1_tg_io'][1]
        ioflowflg = True if NCL1_io > 2 else False
        
        if ioflowflg:
            NCL[0] -= 1
        shape = (NCL[2],NCL[1],NCL[0])

        return meta_data,CoordDF,NCL,InstDF, shape