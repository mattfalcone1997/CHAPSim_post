"""
# _fluct.py
File contains the implementation of the classes to visualise the 
fluctuations of the velocity and pressure fields. Also is used 
to process the additional statistics including autocovarianace
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from scipy import integrate

import sys
import os
import warnings
import gc
import itertools

from .. import CHAPSim_plot as cplt
from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd

import CHAPSim_post as cp

from ._meta import CHAPSim_meta
_meta_class=CHAPSim_meta

from ._average import CHAPSim_AVG,CHAPSim_AVG_io,CHAPSim_AVG_tg_base
_avg_io_class = CHAPSim_AVG_io
_avg_class = CHAPSim_AVG
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._instant import CHAPSim_Inst
_instant_class = CHAPSim_Inst

class CHAPSim_fluct_base():
    _module = sys.modules[__name__]
    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_fluct'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.create_dataset("NCL",data=np.array(self.shape))
        hdf_file.close()
        self.meta_data.save_hdf(file_name,'a',base_name+'/meta_data')
        self.fluctDF.to_hdf(file_name,key=base_name+'/fluctDF',mode='a')#,format='fixed',data_columns=True)

    
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime='',x_split_list='',y_mode='wall',fig='',ax='',**kwargs):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        
        
        if not hasattr(axis_vals,'__iter__'):
            if isinstance(axis_vals,float) or isinstance(axis_vals,int):
                axis_vals = [axis_vals] 
            else:
                msg = "axis_vals must be an interable or a float or an int not a %s"%type(axis_vals)
                raise TypeError(msg)
        
        plane , coord, axis_index = CT.contour_plane(plane,axis_vals,self.avg_data,y_mode,PhyTime)


        # y_index = CT.y_coord_index_norm(self.avg_data,self.avg_data.CoordDF,
        #                                 y_vals,x_vals=0,mode='wall')
        # y_index = list(itertools.chain(*y_index))
        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF[PhyTime,comp]
        
        #fluct=fluct.reshape((self.NCL[2],self.NCL[0]))
        # if isinstance(y_vals,int):
        #     y_vals=[y_vals]
        # elif not isinstance(y_vals,list):
        #     raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
        
        if not x_split_list:
            x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        
        kwargs['squeeze'] = False
        single_input=False
        if not fig:
            if  'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10*len(axis_vals),3*(len(x_split_list)-1)]
            else:
                warnings.warn('Figure size algorithm overridden', stacklevel=2)
            fig, ax = cplt.subplots(len(x_split_list)-1,len(axis_vals),**kwargs)
        elif not ax:
            ax=fig.subplots(len(x_split_list)-1,len(axis_vals),squeeze=False)      
        else:
            if isinstance(ax,mpl.axes.Axes):
                single_input=True
                ax = np.array([ax]).reshape((1,1))
            elif not isinstance(ax,np.ndarray):
                msg = "Input axes must be an instance %s or %s"%(mpl.axes,np.ndarray)
                raise TypeError(msg)
        ax=ax.flatten()

        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)

        title_symbol = CT.get_title_symbol(coord,y_mode,False)
        
        X, Z = np.meshgrid(x_coords, z_coords)
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        if len(x_split_list)==2:
            for i in range(len(axis_vals)):
                # print(X.shape,Z.shape,fluct.shape)
                fluct_slice = CT.contour_indexer(fluct,axis_index[i],coord)
                ax1 = ax[i].pcolormesh(X,Z,fluct_slice,cmap='jet',shading='auto')
                ax2 = ax1.axes
                ax1.set_clim(min_val,max_val)
                cbar=fig.colorbar(ax1,ax=ax[i])
                cbar.set_label(r"$%s^\prime$"%comp)# ,fontsize=12)
                ax2.set_xlim([x_split_list[0],x_split_list[1]])
                ax2.set_xlabel(r"$%s/\delta$" % plane[0])# ,fontsize=20)
                ax2.set_ylabel(r"$%s/\delta$" % plane[1])# ,fontsize=20)
                ax2.set_title(r"$%s=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                ax2.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                ax[i]=ax1
        else:
            ax=ax.flatten()
            max_val = np.amax(fluct); min_val=np.amin(fluct)
            for j in range(len(x_split_list)-1):
                for i in range(len(axis_vals)):
                    fluct_slice = CT.contour_indexer(fluct,axis_index[i],coord)
                    # print(X.shape,Z.shape,fluct.shape)
                    ax1 = ax[j*len(axis_vals)+i].pcolormesh(X,Z,fluct_slice,cmap='jet',shading='auto')
                    ax2 = ax1.axes
                    ax1.set_clim(min_val,max_val)
                    cbar=fig.colorbar(ax1,ax=ax[j*len(axis_vals)+i])
                    cbar.set_label(r"$%s^\prime$"%comp)# ,fontsize=12)
                    ax2.set_xlabel(r"$%s/\delta$" % 'x')# ,fontsize=20)
                    ax2.set_ylabel(r"$%s/\delta$" % 'z')# ,fontsize=20)
                    ax2.set_xlim([x_split_list[j],x_split_list[j+1]])
                    ax2.set_title(r"$%s=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                    ax2.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                    ax[j*len(axis_vals)+i]=ax1
                    ax[j*len(axis_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()
        if single_input:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_streaks(self,comp,vals_list,x_split_list='',PhyTime='',ylim='',Y_plus=True,*args,colors='',fig=None,ax=None,**kwargs):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        fluct = self.fluctDF[PhyTime,comp]
        
        if ylim:
            if Y_plus:
                y_index= CT.Y_plus_index_calc(self.avg_data,self.CoordDF,ylim)
            else:
                y_index=CT.coord_index_calc(self.CoordDF,'y',ylim)
            # y_coords=y_coords[:(y_index+1)]
            fluct=fluct[:,:y_index,:]
        else:
            y_index = self.shape[1]
        # x_coords, y_coords , z_coords = self.meta_data.return_edge_data()
        # y_coords=y_coords[:(y_index+1)]

        # Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)
        
        # if not fig:
        #     fig = cplt.vtkFigure()
        # for val in vals_list:
        #     fig.plot_isosurface(X,Z,Y,fluct,val)
        X = self.meta_data.CoordDF['x']
        Y = self.meta_data.CoordDF['y'][:y_index]
        Z = self.meta_data.CoordDF['z']
        # print(X.shape,Y.shape,Z.shape,fluct.shape)
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
            for val,i in zip(vals_list,range(len(vals_list))):
                
                
                color = colors[i%len(colors)] if colors else ''
                patch = ax[j].plot_isosurface(Y,Z,X[x_start:x_end],fluct[:,:,x_start:x_end],val,color)
                ax[j].add_lighting()
                # patch.set_color(colors[i%len(colors)])

        return fig, ax
    def plot_fluct3D_xz(self,comp,y_vals,PhyTime='',x_split_list='',fig='',ax=''):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        if isinstance(y_vals,int):
            y_vals=[y_vals]
        elif not isinstance(y_vals,list):
            raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
            
        x_coords = self.CoordDF['x']
        z_coords = self.CoordDF['z']
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF[PhyTime,comp][:,y_vals,:]
        
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
        
        ax=ax.flatten()
        max_val = -np.float('inf'); min_val = np.float('inf')
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
                ax[j*len(y_vals)+i].set_ylabel(r'$x/\delta$')# ,fontsize=20)
                ax[j*len(y_vals)+i].set_xlabel(r'$z/\delta$')# ,fontsize=20)
                ax[j*len(y_vals)+i].set_zlabel(r'$%s^\prime$'%comp)# ,fontsize=20)
                ax[j*len(y_vals)+i].set_zlim(min_val,max_val)
                surf.set_clim(min_val,max_val)
                cbar=fig.colorbar(surf,ax=ax[j*len(y_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)# ,fontsize=12)
                ax[j*len(y_vals)+i]=surf
        fig.tight_layout()
        return fig, ax

    def plot_vector(self,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        # if PhyTime is not None:
        #     if type(PhyTime) == float:
        #         PhyTime = "{:.9g}".format(PhyTime)
                
        # if len(set([x[0] for x in self.fluctDF.index])) == 1:
        #     fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
        #     if PhyTime is not None and PhyTime != fluct_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
        #     PhyTime = fluct_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        # if slice not in ['xy','zy','xz']:
        #     slice = slice[::-1]
        #     if slice not in ['xy','yz','xz']:
        #         msg = "The contour slice must be either %s"%['xy','yz','xz']
        #         raise KeyError(msg)
        
        # slice = list(slice)
        # slice_set = set(slice)
        # coord_set = set(list('xyz'))
        # coord = "".join(coord_set.difference(slice_set))

        if not hasattr(ax_val,'__iter__'):
            if isinstance(ax_val,float) or isinstance(ax_val,int):
                ax_val = [ax_val] 
            else:
                msg = "ax_val must be an interable or a float or an int not a %s"%type(ax_val)
                raise TypeError(msg)

        # if not x_split_list:
        #     x_split_list=[np.amin(x_coords),np.amax(x_coords)]

        slice, coord, axis_index = CT.contour_plane(slice,ax_val,self.avg_data,y_mode,PhyTime)
        # if coord == 'y':
        #     norm_vals = [0]*len(ax_val)
        #     axis_index = CT.y_coord_index_norm(self.avg_data,self.CoordDF,ax_val,norm_vals,y_mode)
        # else:
        #     axis_index = CT.coord_index_calc(self.CoordDF,coord,ax_val)
        #     if not hasattr(axis_index,'__iter__'):
        #         axis_index = [axis_index]

        if fig is None:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [8,4*len(ax_val)]
            else:
                warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(ax_val),**kwargs)
        elif ax is None:
            ax=fig.subplots(len(ax_val),squeeze=False)   
        else:
            if isinstance(ax,mpl.axes.Axes):
                ax = np.array([ax]).reshape((1,1))
            elif not isinstance(ax,np.ndarray):
                msg = "Input axes must be an instance %s or %s"%(mpl.axes,np.ndarray)
                raise TypeError(msg)
        ax = ax.flatten()


        coord_1 = self.CoordDF[slice[0]][::spacing[0]]
        coord_2 = self.CoordDF[slice[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in slice]
        U = self.fluctDF[PhyTime,UV_slice[0]]
        V = self.fluctDF[PhyTime,UV_slice[1]]


  
        U_space, V_space = CT.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        ax_out=[]
        quiver_kw = cplt.update_quiver_kw(quiver_kw)
        for i in range(len(ax_val)):
            scale = np.amax(U_space[:,:,i])*coord_1.size/np.amax(coord_1)/scaling
            extent_array = (np.amin(coord_1),np.amax(coord_1),np.amin(coord_2),np.amax(coord_1))
            # im=ax[i].imshow(U_space[:,:,i].astype('f4').T,cmap='jet',extent=)
            # ax[i].pcolormesh(coord_1_mesh,coord_2_mesh,U_space[:,:,i].T,cmap='jet')
            # im.axes.set_xticks(coord_1)
            # im.axes.set_yticks(coord_2)
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


    @classmethod
    def create_video(cls,axis_vals,comp,contour=True,plane='xz',meta_data='',path_to_folder='',time0='',
                            abs_path=True,tgpost=False,x_split_list='',lim_min=None,lim_max=None,
                            ax_func=None,fluct_func=None,fluct_args=(),fluct_kw={}, fig='',ax='',**kwargs):

        module = sys.modules[cls.__module__]
        times= CT.time_extract(path_to_folder,abs_path)
        
        if time0:
            times = sorted(list(filter(lambda x: x > time0, times)))

        if cp.TEST:
            times = times[-10:]

        max_time = np.amax(times)
        avg_data=cls._module._avg_class(max_time,meta_data,path_to_folder,time0,abs_path,tgpost=cls.tgpost)
        if isinstance(axis_vals,(int,float)):
            axis_vals=[axis_vals]
        elif not isinstance(axis_vals,list):
            raise TypeError("\033[1;32 axis_vals must be type int or list but is type %s"%type(axis_vals))
        
        x_coords = avg_data.CoordDF['x']
        if not x_split_list:
            x_split_list=[np.min(x_coords),np.max(x_coords)]
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7*len(axis_vals),3*(len(x_split_list)-1)]
            fig = cplt.figure(**kwargs)
        fig_list = [fig]*len(times)

        frames=list(zip(times,fig_list))
        def animate(frames):
            time=frames[0]; fig=frames[1]
            ax_list=fig.axes
            for ax in ax_list:
                ax.remove()
            fluct_data = cls(time,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)

            if contour:
                fig, ax = fluct_data.plot_contour(comp,axis_vals,plane=plane,PhyTime=time,x_split_list=x_split_list,fig=fig)
            else:
                fig,ax = fluct_data.plot_fluct3D_xz(axis_vals,comp,time,x_split_list,fig)
            ax[0].axes.set_title(r"$t^*=%.3g$"%time,loc='left')

            if fluct_func is not None:
                fluct_func(fig,ax,time,*fluct_args,**fluct_kw)

            if ax_func is not None:
                ax = ax_func(ax)

            for im in ax:
                im.set_clim(vmin=lim_min)
                im.set_clim(vmax=lim_max)

            fig.tight_layout()
            return ax
        anim = mpl.animation.FuncAnimation(fig,animate,frames=frames)
        
        # print("\033[1;32m The colorbar limits are set to (%.3g,%.3g) the maximum limits are (%.3g,%.3g)\033[0;37m\n"%(lim_min,lim_max,lims[0],lims[1]))
        # del lims
        return anim

        
    def __str__(self):
        return self.fluctDF.__str__()

class CHAPSim_fluct_io(CHAPSim_fluct_base):
    tgpost = False
    def __init__(self,time_inst_data_list,avg_data='',path_to_folder='',abs_path=True,*args,**kwargs):
        if not avg_data:
            time = CT.max_time_calc(path_to_folder,abs_path)
            avg_data = self._module._avg_io_class(time,path_to_folder=path_to_folder,abs_path=abs_path,*args,**kwargs)
        if not hasattr(time_inst_data_list,'__iter__'):
            time_inst_data_list = [time_inst_data_list]
        for time_inst_data in time_inst_data_list:
            if isinstance(time_inst_data,self._module._instant_class):
                if 'inst_data' not in locals():
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if 'inst_data' not in locals():
                    inst_data = self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=False)
                else:
                    inst_data += self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=False)

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)

        self.avg_data = avg_data
        self.meta_data = avg_data._meta_data
        self.NCL = self.meta_data.NCL
        self.CoordDF = self.meta_data.CoordDF
        self.shape = inst_data.shape
    # def _fluct_calc(self,inst_data,avg_data):
    #     inst_shape = inst_data.shape
    #     avg_shape = avg_data.shape
    #     fluct_array = np.zeros((4,*inst_shape))
    #     j=0
    #     for (index, inst), (index_avg, avg) in zip(inst_data.InstDF,avg_data.flow_AVGDF):
    #         # inst = inst.values.reshape(inst_data.shape)
    #         # avg = avg.values.reshpe(avg_data.shape)
    #         for i in range(inst_shape[0]):
    #             fluct_array[j,i] = inst[i] - avg
    #         j+=1
    #     return fluct_array.reshape((4,np.prod(inst_shape)))
        
    def _fluctDF_calc(self, inst_data, avg_data):
        
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape[:]))
        avg_time = list(set([x[0] for x in avg_data.flow_AVGDF.index]))
        
        assert len(avg_time) == 1, "In this context therecan only be one time in avg_data"
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape[:]))
        j=0
        
        for j, (time, comp) in enumerate(inst_data.InstDF.index):
            avg_values = avg_data.flow_AVGDF[avg_time[0],comp]
            inst_values = inst_data.InstDF[time,comp]

            for i in range(inst_data.shape[0]):
                fluct[j,i] = inst_values[i] -avg_values
            del inst_values
        
        # fluct = fluct.reshape((len(inst_data.InstDF.index),np.prod(inst_data.shape)))
        # print(inst_times,u_comp)
        return cd.datastruct(fluct,index=inst_data.InstDF.index.copy())
    
class CHAPSim_fluct_tg(CHAPSim_fluct_base):
    tgpost = True
    def __init__(self,time_inst_data_list,avg_data='',path_to_folder='',abs_path=True,*args,**kwargs):
        if not hasattr(time_inst_data_list,'__iter__'):
            time_inst_data_list = [time_inst_data_list]
        for time_inst_data in time_inst_data_list:
            if isinstance(time_inst_data,self._module._instant_class):
                if 'inst_data' not in locals():
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if 'inst_data' not in locals():
                    inst_data = self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=True)
                else:
                    inst_data += self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=True)
        
        if not avg_data:
            times = [float(x[0]) for x in inst_data.InstDF.index]
            avg_data = self._module._avg_tg_base_class(times,path_to_folder=path_to_folder,abs_path=abs_path)

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
                    avg_values = avg_data.flow_AVGDF[None,comp]
                    inst_values = inst_data.InstDF[time,comp]

                    for i in range(inst_data.shape[0]):
                        for k in range(inst_data.shape[2]):
                            fluct[j,i,:,k] = inst_values[i,:,k] -avg_values[:,avg_index]
                    j+=1
        elif all(time in avg_times for time in inst_times):
            for j,index in enumerate(indices):
                avg_index=avg_data._return_index(index[0])
                avg_values = avg_data.flow_AVGDF[None,index[1]]
                inst_values = inst_data.InstDF[index]
                for i in range(inst_data.shape[0]):
                    for k in range(inst_data.shape[2]):
                        fluct[j,i,:,k] = inst_values[i,:,k] -avg_values[:,avg_index]
        else:
            raise ValueError("avg_data must either be length 1 or the same length as inst_data")
        # DF_shape = (len(indices),np.prod(inst_data.shape))
        return cd.datastruct(fluct,index=inst_data.InstDF.index)#.data(inst_data.shape)
    
    def plot_vector(self,*args,**kwargs):
        PhyTime = None
        return super().plot_vector(*args,PhyTime=PhyTime,**kwargs)
