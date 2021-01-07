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

from CHAPSim_post.utils import docstring, gradient, indexing, misc_utils

import CHAPSim_post.CHAPSim_plot as cplt
import CHAPSim_post.CHAPSim_Tools as CT
import CHAPSim_post.CHAPSim_dtypes as cd

# import CHAPSim_post.utils as utils

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
            self._inst_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)

    def _inst_extract(self,time,meta_data=None,path_to_folder='.',abs_path = True,tgpost=False):
        if meta_data is None:
            meta_data = self._module._meta_class(path_to_folder,abs_path,tgpost)
        self._meta_data = meta_data
        self.CoordDF = meta_data.CoordDF
        self.NCL = meta_data.NCL

        #Give capacity for both float and lists
        if isinstance(time,(float,int)): 
            self.InstDF = self._flow_extract(time,path_to_folder,abs_path,tgpost)
        elif isinstance(time,(list,tuple)):
            for PhyTime in time:
                if not hasattr(self, 'InstDF'):
                    self.InstDF = self._flow_extract(PhyTime,path_to_folder,abs_path,tgpost)
                else: #Variable already exists
                    local_DF = self._flow_extract(PhyTime,path_to_folder,abs_path,tgpost)
                    # concat_DF = [self.InstDF,local_DF]
                    self.InstDF.concat(local_DF)
        else:
            raise TypeError("\033[1;32 `time' must be either float or list")
        self.shape = (self.NCL[2],self.NCL[1],self.NCL[0])

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)
    
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = 'CHAPSim_Inst'
        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')
        self.CoordDF=self._meta_data.CoordDF
        self.NCL=self._meta_data.NCL

        self.shape = (self.NCL[2],self.NCL[1],self.NCL[0])
        self.InstDF = cd.datastruct.from_hdf(file_name,shapes=self.shape,key=key+'/InstDF')#pd.read_hdf(file_name,base_name+'/InstDF').data(shape)

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = 'CHAPSim_Inst'
        self._meta_data.save_hdf(file_name,write_mode,key=key+'/meta_data')
        self.InstDF.to_hdf(file_name,key=key+'/InstDF',mode='a')

    def _flow_extract(self,Time_input,path_to_folder,abs_path,tgpost):
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
        flow_info = self.__velo_interp(flow_info,NCL3,NCL2,NCL1)
        gc.collect()

        Phy_string = '%.9g' % PhyTime

        # creating datastruct index
        index = [[Phy_string]*4,['u','v','w','P']]

        # creating datastruct so that data can be easily accessible elsewhere
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

    def check_PhyTime(self,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = "PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible"
        with warnings.catch_warnings(record=True) as w:
            key = self.InstDF.check_index(PhyTime,err_msg=err_msg,warn_msg=warn_msg,outer=True)
            a = w
        if PhyTime is not None and len(a)>0:
            for warn in a:
                warnings.warn(a.message)
        return key[0]

    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime=None,x_split_list=None,y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
                
        PhyTime = self.check_PhyTime(PhyTime)        
        
        axis_vals = misc_utils.check_list_vals(axis_vals)
            
        plane , coord, axis_index = indexing.contour_plane(plane,axis_vals,self.avg_data,y_mode,PhyTime)


        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.InstDF[PhyTime,comp]
                
        if x_split_list is None:
            x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        
        ax_layout = (len(x_split_list)-1,len(axis_vals))
        figsize=[10*len(axis_vals),3*(len(x_split_list)-1)]
        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(*ax_layout,fig=fig,ax=ax,**kwargs)

        ax=ax.flatten()

        title_symbol = CT.get_title_symbol(coord,y_mode,False)
        
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        X, Z = np.meshgrid(x_coords, z_coords)

        ax=ax.flatten()
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        for j,_ in enumerate(x_split_list[:-1]):
            for i,_ in enumerate(axis_vals):
                fluct_slice = CT.contour_indexer(fluct,axis_index[i],coord)

                ax1 = ax[j*len(axis_vals)+i].pcolormesh(X,Z,fluct_slice,**pcolor_kw)
                ax1.set_clim(min_val,max_val)

                ax1.axes.set_xlabel(r"$%s/\delta$" % 'x')
                ax1.axes.set_ylabel(r"$%s/\delta$" % 'z')
                ax1.axes.set_xlim([x_split_list[j],x_split_list[j+1]])
                ax1.axes.set_title(r"$%s=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                
                cbar=fig.colorbar(ax1,ax=ax[j*len(axis_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)

                ax[j*len(axis_vals)+i]=ax1
                ax[j*len(axis_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()

        if ax.size == 1:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_vector(self,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        
        ax_val = misc_utils.check_list_vals(ax_val)

        slice, coord, axis_index = indexing.contour_plane(slice,ax_val,self.avg_data,y_mode,PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[8,4*len(ax_val)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(ax_val),fig=fig,ax=ax,**kwargs)
        ax = ax.flatten()


        coord_1 = self.CoordDF[slice[0]][::spacing[0]]
        coord_2 = self.CoordDF[slice[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in slice]
        U = self.InstDF[PhyTime,UV_slice[0]]
        V = self.InstDF[PhyTime,UV_slice[1]]

        U_space, V_space = CT.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        ax_out=[]
        quiver_kw = cplt.update_quiver_kw(quiver_kw)
        for i in range(len(ax_val)):
            scale = np.amax(U_space[:,:,i])*coord_1.size/np.amax(coord_1)/scaling
            extent_array = (np.amin(coord_1),np.amax(coord_1),np.amin(coord_2),np.amax(coord_1))

            Q = ax[i].quiver(coord_1_mesh, coord_2_mesh,U_space[:,:,i].T,V_space[:,:,i].T,angles='uv',scale_units='xy', scale=scale,**quiver_kw)
            ax[i].set_xlabel(r"$%s^*$"%slice[0])
            ax[i].set_ylabel(r"$%s$"%slice[1])
            ax[i].set_title(r"$%s = %.3g$"%(coord,ax_val[i]),loc='right')
            ax[i].set_title(r"$t^*=%s$"%PhyTime,loc='left')
            ax_out.append(Q)
        if len(ax_out) ==1:
            return fig, ax_out[0]
        else:
            return fig, np.array(ax_out)
    
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
                strain_rate[:,:,:,i,j] = 0.5*(gradient.Grad_calc(self.CoordDF,velo_field1,coord2,False) \
                                        + gradient.Grad_calc(self.CoordDF,velo_field2,coord1,False))
                rot_rate[:,:,:,i,j] = 0.5*(gradient.Grad_calc(self.CoordDF,velo_field1,coord2,False) \
                                        - gradient.Grad_calc(self.CoordDF,velo_field2,coord1,False))
                j+=1
            i+=1
        del velo_field1 ; del velo_field2
        S2_Omega2 = strain_rate**2 + rot_rate**2
        del strain_rate ; del rot_rate

        S2_Omega2_eigvals, e_vecs = np.linalg.eigh(S2_Omega2)
        del e_vecs; del S2_Omega2
        
        lambda2 = np.sort(S2_Omega2_eigvals,axis=3)[:,:,:,1]
        
        return lambda2
    def plot_lambda2(self,vals_list,x_split_list=None,PhyTime=None,ylim=None,Y_plus=True,avg_data=None,colors=None,fig=None,ax=None,**kwargs):
                
        PhyTime = self.check_PhyTime(PhyTime)            
        
        
        if not hasattr(vals_list,'__iter__'):
            vals_list = [vals_list]
        X = self._meta_data.CoordDF['x']
        Y = self._meta_data.CoordDF['y']
        Z = self._meta_data.CoordDF['z']

        if ylim is not None:
            if Y_plus:
                if avg_data is None:
                    msg = "If Y_plus is selected, the avg_data keyword argument needs to be given"
                    raise ValueError(msg)
                y_index= CT.Y_plus_index_calc(avg_data,self.CoordDF,ylim)
            else:
                y_index=CT.coord_index_calc(self.CoordDF,'y',ylim)
            Y=Y[:y_index]
            # lambda2=lambda2[:,:y_index,:]
        if x_split_list is None:
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
                
                color = colors[i%len(colors)] if colors is not None else ''
                patch = ax[j].plot_isosurface(Y,Z,X[x_start:x_end],lambda2,val,color)
        
        
        return fig, ax

    def vorticity_calc(self,PhyTime=None):

        self.check_PhyTime(PhyTime)

        vorticity = np.zeros((3,*self.shape),dtype='f8')
        u_velo = self.InstDF[PhyTime,'u']
        v_velo = self.InstDF[PhyTime,'v']
        w_velo = self.InstDF[PhyTime,'w']

        vorticity[0] = CT.Grad_calc(self.CoordDF,w_velo,'y',False) - CT.Grad_calc(self.CoordDF,v_velo,'z',False)      
        vorticity[1] = CT.Grad_calc(self.CoordDF,u_velo,'z',False) - CT.Grad_calc(self.CoordDF,w_velo,'x',False)      
        vorticity[2] = CT.Grad_calc(self.CoordDF,v_velo,'x',False) - CT.Grad_calc(self.CoordDF,u_velo,'y',False)     

        return cd.datastruct(vorticity,index=['x','y','z'])

    def plot_vorticity_contour(self,comp,slice,ax_val,PhyTime=None,avg_data=None,y_mode='half_channel',pcolor_kw=None,fig=None,ax=None,**kwargs):
        
        if isinstance(ax_val,(int,float)):
            ax_val = [ax_val]        
        elif not isinstance(ax_val,(list,tuple)):
            msg = "ax_val must be an interable or a float or an int not a %s"%type(ax_val)
            raise TypeError(msg)
        
        PhyTime = self.check_PhyTime(PhyTime)

        slice, coord,axis_index = indexing.contour_plane(slice,ax_val,avg_data,y_mode,PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[8,4*len(ax_val)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(ax_val),fig=fig,ax=ax,**kwargs)   
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
            mesh = ax[i].pcolormesh(coord_1_mesh,coord_2_mesh,contour_array,**pcolor_kw)
            ax[i].set_xlabel(r"$%s^*$"%slice[0])
            ax[i].set_ylabel(r"$%s$"%slice[1])
            ax[i].set_title(r"$%s = %.3g$"%(coord,ax_val[i]))
            cbar=fig.colorbar(mesh,ax=ax[i])
            cbar.set_label(r"$\omega_%s$"%comp)
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
        super()._inst_extract(*args,**kwargs)

class CHAPSim_Inst_tg(CHAPSim_Inst):
    tgpost = True
    def _inst_extract(self,*args,**kwargs):
        kwargs['tgpost'] = self.tgpost
        super()._inst_extract(*args,**kwargs)
        
        NCL1_io = self._meta_data.metaDF['NCL1_tg_io'][1]
        ioflowflg = True if NCL1_io > 2 else False
        
        if ioflowflg:
            self.NCL[0] -= 1
        self.shape = (self.NCL[2],self.NCL[1],self.NCL[0])
