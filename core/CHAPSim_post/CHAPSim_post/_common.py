"""
## _common.py
A module to create base level visualisation classes 
providing functionality common to several classes
"""
import numpy as np
import matplotlib as mpl
import sys

import itertools
from abc import abstractmethod, ABC

import CHAPSim_post.CHAPSim_plot as cplt
from CHAPSim_post.utils import indexing, misc_utils,gradient

_cart_to_cylind_str = {
    'x' : 'z',
    'y' : 'r',
    'z' : r'\theta',
    'u' : r'u_z',
    'v' : r'u_r',
    'w' : r'u_\theta'
}

_cylind_to_cart ={
    'z' : 'x',
    'r' : 'y',
    'theta' : 'z'
}

class DomainHandler():

    def __init__(self,meta_data):
        if meta_data.metaDF['iCase'] in [1,4]:
            self.coord_sys = 'cart'
        elif meta_data.metaDF['iCase'] in [2,3]:
            self.coord_sys = 'cylind'
        else:
            msg = "CHAPSim case type invalid"
            raise ValueError(msg)

        self.Grad_calc = gradient.Grad_calc

    @property
    def is_cylind(self):
        return self.coord_sys == 'cylind'

    def __str__(self):
        if self.coord_sys == 'cart':
            coord = "cartesian"
        else:
            coord = "cylindrical"

        return f"{self.__class__.__name__} with %s coordinate system"%coord

    def __repr__(self):
        return self.__str__()

    def _alter_item(self,char,to_out=True):
        convert_dict = _cart_to_cylind_str if to_out else _cylind_to_cart
        if self.coord_sys == 'cylind' and char in convert_dict.keys():
            return convert_dict[char]
        else:
            return char

    def in_to_out(self,key):
        return "".join([self._alter_item(x,True) for x in key])

    def out_to_in(self,key):
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

            factor_out = 1/coordDF['y'] if self.is_cylind else 1.0
            grad_vector[2] = np.multiply(grad_vector[2],factor_out)

        return grad_vector

    def Vector_div_io(self,coordDF,vector_array):
        if vector_array.ndim not in (3,4):
            msg = "The number of dimension of the vector array must be 3 or 4"
            raise ValueError(msg)

        grad_vector = np.zeros_like(vector_array)
        grad_vector[0] = self.Grad_calc(coordDF,vector_array[0],'x')
        
        factor_in = coordDF['y'] if self.is_cylind else 1.0
        factor_out = 1/coordDF['y'] if self.is_cylind else 1.0

        grad_vector[1] = self.Grad_calc(coordDF,np.multiply(vector_array[1],factor_in),'y')
        grad_vector[1] = np.multiply(grad_vector[1],factor_out)
        
        if vector_array.ndim == 4:
            grad_vector[2] = self.Grad_calc(coordDF,
                                    np.multiply(vector_array[2],factor_in),'z')
            grad_vector[1] = np.multiply(grad_vector[1],factor_out)

        div_scalar = np.sum(grad_vector,axis=0)
        return div_scalar

    def Scalar_laplacian(self,coordDF,flow_array):
        grad_vector = self.Scalar_grad_io(coordDF,flow_array)
        lap_scalar = self.Vector_div_io(coordDF,grad_vector)
        return lap_scalar

    def Scalar_laplacian_tg(self,coordDF,flow_array):
        factor_in = coordDF['y'] if self.is_cylind else 1.0
        factor_out = 1/coordDF['y'] if self.is_cylind else 1.0
        dflow_dy = np.multiply(self.Grad_calc(coordDF,flow_array,'y'),
                                factor_in)
        lap_scalar = np.multiply(self.Grad_calc(coordDF,dflow_dy,'y'),
                                factor_out)
        return lap_scalar

    
class classproperty():
    def __init__(self,func):
        self.f = func
    def __get__(self,obj,cls):
        return self.f(cls)
    

class Common(ABC):
    def __init__(self,meta_data):
        self.Domain = DomainHandler(meta_data)

    @classproperty
    def _module(cls):
        return sys.modules[cls.__module__]

    def _check_outer(self,processDF,outer,err_msg,warn_msg):

        err = ValueError(err_msg)
        warn = UserWarning(warn_msg)
        return processDF.check_outer(outer,err,warn)

    def check_comp(self,processDF,comp):
        err = Exception()
        try:
            processDF.check_inner(comp,err)
        except Exception:
            return False
        else:
            return True

    
class common3D(Common):

    @abstractmethod
    def _check_outer(self,processDF,outer,err_msg,warn_msg):
        return super()._check_outer(processDF,outer,err_msg,warn_msg)

    def contour_plane(self,plane,axis_vals,avg_data,y_mode,PhyTime):

        plane = self.Domain.out_to_in(plane)

        if plane not in ['xy','zy','xz']:
            plane = plane[::-1]
            if plane not in ['xy','zy','xz']:
                msg = "The contour slice must be either %s"%['xy','yz','xz']
                raise KeyError(msg)
        slice_set = set(plane)
        coord_set = set(list('xyz'))
        coord = "".join(coord_set.difference(slice_set))

        if coord == 'y':
            tg_post = True if all([x == 'None' for x in avg_data.flow_AVGDF.times]) else False
            if not tg_post:
                norm_val = 0
            elif tg_post:
                norm_val = PhyTime
            else:
                raise ValueError("problems")
            norm_vals = [norm_val]*len(axis_vals)
            if avg_data is None:
                msg = f'For contour slice {slice}, avg_data must be provided'
                raise ValueError(msg)
            axis_index = indexing.y_coord_index_norm(avg_data,axis_vals,norm_vals,y_mode)
        else:
            axis_index = indexing.coord_index_calc(avg_data.CoordDF,coord,axis_vals)
            if not hasattr(axis_index,'__iter__'):
                axis_index = [axis_index]
        # print(axis_index)
        return plane, coord, axis_index

    def contour_indexer(self,array,axis_index,coord):

        if coord == 'x':
            indexed_array = array[:,:,axis_index].squeeze().T
        elif coord == 'y':
            indexed_array = array[:,axis_index].squeeze()
        else:
            indexed_array = array[axis_index].squeeze()
        return indexed_array

    def vector_indexer(self,U,V,axis_index,coord,spacing_1,spacing_2):
        if isinstance(axis_index[0],list):
            ax_index = list(itertools.chain(*axis_index))
        else:
            ax_index = axis_index[:]
        if coord == 'x':
            U_space = U[::spacing_1,::spacing_2,ax_index]
            V_space = V[::spacing_1,::spacing_2,ax_index]
        elif coord == 'y':
            U_space = U[::spacing_2,ax_index,::spacing_1]
            V_space = V[::spacing_2,ax_index,::spacing_1]
            U_space = np.swapaxes(U_space,1,2)
            U_space = np.swapaxes(U_space,1,0)
            V_space = np.swapaxes(V_space,1,2)
            V_space = np.swapaxes(V_space,1,0)

        else:
            U_space = U[ax_index,::spacing_2,::spacing_1]
            V_space = V[ax_index,::spacing_2,::spacing_1]
            U_space = np.swapaxes(U_space,2,0)
            V_space = np.swapaxes(V_space,0,2)
            
        return U_space, V_space

    def _check_attr(self):
        attr_list = ["CoordDF","meta_data"]
        for attr in attr_list:
            if not hasattr(self,attr):
                msg = (f"The class {self.__class__.__name__}, requires the attribute"
                        f" {attr} to use base class Common3D's methods")
                raise AttributeError(msg)

    def plot_contour(self,ProcessDF,avg_data,comp,axis_vals,plane='xz',PhyTime=None,x_split_list=None,y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        self._check_attr()
        
        PhyTime = self._check_outer(ProcessDF,PhyTime)          

        axis_vals = misc_utils.check_list_vals(axis_vals)
            
        plane , coord, axis_index = indexing.contour_plane(plane,axis_vals,avg_data,y_mode,PhyTime)

        if coord == 'y':
            axis_vals = indexing.ycoords_from_coords(avg_data,axis_vals,mode=y_mode)[0]
        else:
            axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)

        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = ProcessDF[PhyTime,comp]
                
        if x_split_list is None:
            x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        
        x_size = 1.5*(np.amax(x_coords) - np.amin(x_coords))
        z_size = 1.2*(np.amax(z_coords) - np.amin(z_coords))

        ax_layout = (len(x_split_list)-1,len(axis_vals))
        figsize=[x_size*len(axis_vals),z_size*(len(x_split_list)-1)]

        
        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(*ax_layout,fig=fig,ax=ax,**kwargs)

        ax=ax.flatten()

        # x_coords_split=indexing.coord_index_calc(self.CoordDF,'x',x_split_list)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)
        
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        X, Z = np.meshgrid(x_coords, z_coords)

        ax=ax.flatten()
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        for j,_ in enumerate(x_split_list[:-1]):
            for i,_ in enumerate(axis_vals):
                fluct_slice = indexing.contour_indexer(fluct,axis_index[i],coord)
                # print(fluct_slice.shape)
                ax1 = ax[j*len(axis_vals)+i].pcolormesh(X,Z,fluct_slice,**pcolor_kw)
                ax1.set_clim(min_val,max_val)

                ax1.axes.set_xlabel(r"$%s/\delta$" % plane[0])
                ax1.axes.set_ylabel(r"$%s/\delta$" % plane[1])
                ax1.axes.set_xlim([x_split_list[j],x_split_list[j+1]])
                ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
                ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                
                cbar=fig.colorbar(ax1,ax=ax[j*len(axis_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)

                ax[j*len(axis_vals)+i]=ax1
                ax[j*len(axis_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
            
    def plot_streaks(self,ProcessDF,avg_data,comp,vals_list,x_split_list=None,PhyTime=None,ylim='',Y_plus=True,*args,colors='',fig=None,ax=None,**kwargs):

        self._check_attr()
        PhyTime = self._check_outer(ProcessDF,PhyTime)   

        fluct = ProcessDF[PhyTime,comp]
        
        if ylim:
            if Y_plus:
                y_index= indexing.Y_plus_index_calc(avg_data,self.CoordDF,ylim)
            else:
                y_index=indexing.coord_index_calc(self.CoordDF,'y',ylim)
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
            x_start = indexing.coord_index_calc(self.CoordDF,'x',x_split_list[j])
            x_end = indexing.coord_index_calc(self.CoordDF,'x',x_split_list[j+1])
            for val,i in zip(vals_list,range(len(vals_list))):
                
                
                color = colors[i%len(colors)] if colors else ''
                patch = ax[j].plot_isosurface(Y,Z,X[x_start:x_end],fluct[:,:,x_start:x_end],val,color)
                ax[j].add_lighting()
                # patch.set_color(colors[i%len(colors)])

        return fig, ax

    def plot3D_xz(self,ProcessDF,avg_data,comp,y_vals,y_mode='half-channel',PhyTime=None,x_split_list=None,fig=None,ax=None,surf_kw=None,**kwargs):

        self._check_attr()
        
        PhyTime = self._check_outer(ProcessDF,PhyTime)     

        y_vals = misc_utils.check_list_vals(y_vals)    

        axis_index = indexing.y_coord_index_norm(avg_data,y_vals,mode=y_mode)[0]
        # axis_index = np.diag(axis_index)
        y_vals = indexing.ycoords_from_coords(avg_data,y_vals,mode=y_mode)[0]

        CoordDF = self.meta_data.CoordDF

        x_coords, z_coords = (CoordDF['x'],CoordDF['z'])

        Z, X = np.meshgrid(z_coords,x_coords)

        fluct = ProcessDF[PhyTime,comp][:,axis_index,:]
                              
        surf_kw = cplt.update_mesh_kw(surf_kw,rcount=100,ccount=150)

        single_input=False
        if not isinstance(ax,np.ndarray):
            single_input=True

        if x_split_list is None:
            x_split_list = [np.amin(x_coords),np.amax(x_coords)]

        x_index_list = indexing.coord_index_calc(CoordDF,'x',x_split_list)

        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'},antialiased=True)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(x_index_list)-1,len(y_vals),fig=fig,ax=ax,**kwargs)
        ax = ax.flatten()
        
        max_val = -np.float('inf'); min_val = np.float('inf')
        for j,_ in enumerate(x_index_list[:-1]):
            for i,_ in enumerate(y_vals):
                max_val = np.amax(fluct[:,i,:]); min_val=np.amin(fluct[:,i,:])
                
                z_vals = Z[x_index_list[j]:x_index_list[j+1]]
                x_vals = X[x_index_list[j]:x_index_list[j+1]]
                surf_vals = fluct[:,i,x_index_list[j]:x_index_list[j+1]].T

                ax[j*len(x_index_list[:-1])+i] = ax[j*len(x_index_list[:-1])+i].plot_surface( z_vals,x_vals,surf_vals,**surf_kw)

                ax[j*len(x_index_list[:-1])+i].axes.set_ylabel(r'$x/\delta$')
                ax[j*len(x_index_list[:-1])+i].axes.set_xlabel(r'$z/\delta$')
                ax[j*len(x_index_list[:-1])+i].axes.set_zlabel(r'$%s^\prime$'%comp)
                ax[j*len(x_index_list[:-1])+i].axes.invert_xaxis()
                # ax[j*len(x_split_list[:-1])+i].axes.set_ylim([x_split_list[j],x_split_list[j+1]])

        for a in ax.flatten():
            a.set_clim([min_val,max_val])
            a.axes.set_zlim([min_val,max_val])
        if single_input:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_vector(self,ProcessDF,avg_data,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        self._check_attr()
        PhyTime = self._check_outer(ProcessDF,PhyTime)    
        
        ax_val = misc_utils.check_list_vals(ax_val)

        slice, coord, axis_index = indexing.contour_plane(slice,ax_val,avg_data,y_mode,PhyTime)

        if coord == 'y':
            ax_val = indexing.ycoords_from_coords(avg_data,ax_val,mode=y_mode)[0]
        else:
            ax_val = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)


        kwargs = cplt.update_subplots_kw(kwargs,figsize=[8,4*len(ax_val)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(ax_val),fig=fig,ax=ax,**kwargs)

        ax = ax.flatten()


        coord_1 = self.CoordDF[slice[0]][::spacing[0]]
        coord_2 = self.CoordDF[slice[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in slice]
        U = ProcessDF[PhyTime,UV_slice[0]]
        V = ProcessDF[PhyTime,UV_slice[1]]

        U_space, V_space = indexing.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        ax_out=[]
        quiver_kw = cplt.update_quiver_kw(quiver_kw)
        for i in range(len(ax_val)):
            scale = np.amax(U_space[:,:,i])*coord_1.size/np.amax(coord_1)/scaling
            extent_array = (np.amin(coord_1),np.amax(coord_1),np.amin(coord_2),np.amax(coord_1))

            Q = ax[i].quiver(coord_1_mesh, coord_2_mesh,U_space[:,:,i].T,V_space[:,:,i].T,angles='uv',scale_units='xy', scale=scale,**quiver_kw)
            ax[i].set_xlabel(r"$%s/\delta$"%slice[0])
            ax[i].set_ylabel(r"$%s/\delta$"%slice[1])
            ax[i].set_title(r"$%s = %.2g$"%(coord,ax_val[i]),loc='right')
            ax[i].set_title(r"$t^*=%s$"%PhyTime,loc='left')
            ax_out.append(Q)
        if len(ax_out) ==1:
            return fig, ax_out[0]
        else:
            return fig, np.array(ax_out)