"""
## plot
This is a submodule of CHAPSim_plot extending matplotlib functionality
for simpler high-level use in this application
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from cycler import cycler

import itertools
import warnings
import copy
from shutil import which

import CHAPSim_post as cp
from CHAPSim_post.utils import misc_utils

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri
from numpy.core.fromnumeric import swapaxes
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interpn,interp1d

try:
    from skimage import measure
    _has_isosurface = True
except ImportError:
    _has_isosurface = False
    msg = ("An issue importing skimage for creating isosurfaces,"
           " this has been disabled")
    warnings.warn(msg)


if which('pdflatex') is not None:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] ='\\usepackage{amsmath}\n'

def _linekw_alias(**kwargs):
    alias_dict = {'aa':'antialiased',
                'c' : 'color',
                'ds':'drawstyle',
                'ls':'linestyle',
                'lw':'linewidth',
                'mec' : 'markeredgecolor',
                'mew' : 'markeredgewidth',
                'mfc' : 'markerfacecolor',
                'mfcalt' : 'markerfacecoloralt',
                'ms' : 'markersize'}
    
    new_dict = {}
    for key, val in kwargs.items():
        if key in alias_dict.keys():
            key = alias_dict[key]
        new_dict[key] = val
    return new_dict

def update_prop_cycle(**kwargs):
    avail_keys = [x[4:] for x in mpl.lines.Line2D.__dict__.keys() if x[0:3]=='get']

    if not all([key in avail_keys for key in kwargs.keys()]):
        msg = "The key is invalid for the matplotlib property cycler"
        raise ValueError(msg)

    kwargs = _linekw_alias(**kwargs)
    
    cycler_dict = mpl.rcParams['axes.prop_cycle'].by_key()
    for key, item in kwargs.items():
        if not hasattr(item,"__iter__"):
            item = [item]
        elif isinstance(item,str):
            if item == "" :
                item = [item]
        cycler_dict[key] = item

    item_length = [ len(item) for _,item in cycler_dict.items()]
    cycle_length = np.lcm.reduce(item_length)

    for key, val in cycler_dict.items():
        cycler_dict[key] = list(val)*int(cycle_length/len(val))
    mpl.rcParams['axes.prop_cycle'] = cycler(**cycler_dict)

default_prop_dict = dict(linestyle=['-','--','-.',':'],
                        marker=['x','.','v','^','+'],
                        color = 'bgrcmyk')

def reset_prop_cycler(**kwargs):
    update_prop_cycle(**default_prop_dict)
    update_prop_cycle(**kwargs)

reset_prop_cycler()

mpl.rcParams['lines.markerfacecolor'] = 'white'
# mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['legend.edgecolor'] = 'inherit'
mpl.rcParams['font.size'] = 17
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['axes.labelpad'] = 6.0
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['image.cmap'] = "jet"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


class CHAPSimFigure(mpl.figure.Figure):

    def clegend(self,*args, **kwargs):
        return super().legend(*args, **kwargs)
    def add_subplot(self,*args, **kwargs):
        if 'projection' not in kwargs.keys():
            kwargs['projection']='AxesCHAPSim'
        elif kwargs['projection'] == '3d':
            kwargs['projection'] = 'Axes3DCHAPSim'

        return super().add_subplot(*args,**kwargs)

    def c_add_subplot(self,*args, **kwargs):
        return self.add_subplot(*args,**kwargs)
    def get_legend(self):
        if len(self.legends) == 1:
            return self.legends[0]
        else:
            return self.legends
    def update_legend_fontsize(self, fontsize,tight_layout=True,**kwargs):
        if self.legends:
            texts=[legend.get_texts() for legend in self.legends]
            texts = itertools.chain(*texts)
            for text in texts:
                text.set_fontsize(fontsize)
        if tight_layout:
            self.get_axes()[0].get_gridspec().tight_layout(self,**kwargs)
    
    def set_axes_title_fontsize(self, fontsize):
        axes = self.get_axes()
        for ax in axes:
            ax.set_title_fontsize(fontsize)

    def tighter_layout(self,*args,**kwargs):
        gridspecs = self._gridspecs
        if gridspecs:
            for gs in gridspecs:
                gs.tight_layout(self,*args,**kwargs)
        else:
            super().tight_layout()

class AxesCHAPSim(mpl.axes.Axes):
    name='AxesCHAPSim'

    def cplot(self,*args, **kwargs):

        counter = self.count_lines()

        plot_kw = {}
        for key,val in mpl.rcParams['axes.prop_cycle'].by_key().items():
            plot_kw[key] = val[counter%len(val)]
        
        kwargs = _linekw_alias(**kwargs)
        for key,val in kwargs.items():
            plot_kw[key] = val

        if 'markevery' not in plot_kw:
            plot_kw['markevery'] = 10
        return super().plot(*args,**plot_kw)

    def get_shared_lines(self):

        attr_list = ['_shared_x_axes', '_shared_y_axes', '_twinned_axes']
        state = self.__getstate__()
        all_axes = { k: v for k,v in state.items() \
                    if k in attr_list and v is not None}.values()

        axes = set(itertools.chain(*all_axes))
        if axes == set():
            axes = set([self])

        shared_lines=[]
        for ax in axes:
            shared_lines.extend(ax.get_lines())
        return shared_lines

    def count_lines(self):

        lines = self.get_shared_lines()
        return len(lines)

    def clegend(self,*args, **kwargs):
        # if 'fontsize' not in kwargs.keys():
        #     kwargs['fontsize']=legend_fontsize
        if 'loc' not in kwargs.keys() and 'bbox_to_anchor' not in kwargs.keys():
            kwargs['loc'] = 'lower center'
            kwargs['bbox_to_anchor'] = (0.5,1.08)
        if not kwargs.pop('vertical',False):
            ncol = kwargs['ncol'] if 'ncol' in kwargs.keys() else 1
            if len(args)==2:
                args = (flip_leg_col(args[0],ncol),
                        flip_leg_col(args[1],ncol))
            elif len(args)==1:
                args = flip_leg_col(args,ncol)
            elif 'labels' in kwargs.keys() and 'handles' in kwargs.keys():
                kwargs['labels'] = flip_leg_col(kwargs['labels'],ncol)
                kwargs['handles'] = flip_leg_col(kwargs['handles'],ncol)
            elif 'labels' not in kwargs.keys() and 'handles' not in kwargs.keys() and len(args)==0:
                handles, labels = self.get_legend_handles_labels()
                kwargs['labels'] = flip_leg_col(labels,ncol)
                kwargs['handles'] = flip_leg_col(handles,ncol)
        return super().legend(*args, **kwargs)

    def set_label_fontsize(self ,fontsize,tight_layout=True,**kwargs):
        xlabel_str=self.get_xlabel()
        ylabel_str=self.get_ylabel()

        self.set_xlabel(xlabel_str ,fontsize=fontsize)
        self.set_ylabel(ylabel_str ,fontsize=fontsize)
        if tight_layout:
            self.get_figure().tight_layout(**kwargs)

    def set_title_fontsize(self ,fontsize,loc='center',tight_layout=True,**kwargs):
        title_str=self.get_title(loc=loc)
        self.set_title(title_str ,loc=loc,fontsize=fontsize)
        if tight_layout:
            self.get_gridspec().tight_layout(self.get_figure(),**kwargs)


    def update_legend_fontsize(self, fontsize,tight_layout=True,**kwargs):
        leg=self.get_legend()
        if leg is not None:
            texts=leg.get_texts()
            for text in texts:
                text.set_fontsize(fontsize)
        if tight_layout:
            self.get_gridspec().tight_layout(self.get_figure(),**kwargs)
        

    def set_line_markevery(self,every):
        lines = self.get_lines()
        for line in lines:
            line.set_markevery(every)
    def toggle_default_line_markers(self,*args, **kwargs):
        markers=['x','.','v','^','+']
        lines = self.get_lines()
        for line, i in zip(lines,range(len(lines))):
            if line.get_marker() is None:
                line.set_marker(markers[i%len(markers)])
            else:
                line.set_marker(None)
        if self.get_legend() is not None:
            self.clegend(*args,**kwargs)

    def normalise(self,axis,val,use_gcl=False):

        lines = self.get_lines()[-1:] if use_gcl else self.get_lines()

        for i,line in enumerate(lines):
            if hasattr(val,"__iter__"):
                if len(val) != len(self.get_lines()):
                    raise ValueError("There must be as many lines as normalisation values")
                norm_val = val[i]
            else:
                if hasattr(val,"__iter__"):
                    if len(val) != len(line.get_xdata()):
                        raise RuntimeError("The length of vals must be the same as the"+\
                                    "number of lines in an axis") 
                norm_val = val
            xdata=0; ydata=0
            xdata, ydata = line.get_data()
            
            if axis=='x':
                xdata =  np.array(xdata)/norm_val
            else:
                ydata =  np.array(ydata)/norm_val
            line.set_data(xdata, ydata)

        lim_val = max(val) if hasattr(val,"__iter__") else val

        def _other_logscale_lim(axis,lims):
            if axis == 'x':
                scale = self.get_xscale()
            else:
                scale = self.get_yscale()

            if scale == 'log' and any([lim < 0 for lim in lims]):
                return True
            else:
                return False

        if self.get_lines():
            if axis == 'x':
                xlims = [x/lim_val for x in self.get_xlim()]
                if _other_logscale_lim(axis,xlims):
                    data = np.array([line.get_xdata() for line in self.get_lines() ])
                    xlims = [np.amin(data),np.amax(data)]
                    
                self.set_xlim(xlims)
            else:
                ylims = [y/lim_val for y in self.get_ylim()]

                if _other_logscale_lim(axis,ylims):
                    data = np.array([line.get_ydata() for line in self.get_lines() ])
                    ylims = [np.amin(data),np.amax(data)]
                    
                self.set_ylim(ylims)

    def array_normalise(self,axis,val,use_gcl=False):
        lines = self.get_lines()[-1:] if use_gcl else self.get_lines()

        for line in lines:
            xdata, ydata = line.get_data()
            if xdata.size != len(val):
                raise ValueError("The size of val must be the same as the data")
            if axis=='x':
                xdata =  np.array(xdata)/val
            else:
                ydata =  np.array(ydata)/val
            line.set_data(xdata, ydata)

        self.relim()
        self.autoscale_view(True,True,True)

    def apply_func(self,axis,func,use_gcl=False):

        lines = self.get_lines()[-1:] if use_gcl else self.get_lines()
        for line in lines:
            xdata, ydata = line.get_data()
            if axis == 'y':
                ydata = func(ydata)
            elif axis == 'x':
                xdata = func(xdata)
            else:
                raise KeyError
            line.set_data(xdata,ydata)
        
        self.relim()
        self.autoscale_view(True,True,True)
        
    def toggle_default_linestyle(self,*args,**kwargs):
        linestyle=['-','--','-.',':']
        lines = self.get_lines()
        c_linestyle = [line.get_linestyle() for line in lines]
        if c_linestyle[:len(linestyle)] == linestyle:
            for line, i in zip(lines,range(len(lines))):
                line.set_linestyle('-')
        else:
            for line, i in zip(lines,range(len(lines))):
                line.set_linestyle(linestyle[i%len(linestyle)])
        
        if self.get_legend() is not None:
            self.clegend(*args,**kwargs)

    def shift_xaxis(self,val,shared=True,use_gcl=False):
        if shared and not use_gcl:
            lines = self.get_shared_lines()
        else:
            lines = self.get_lines()[-1:] if use_gcl else self.get_lines()

        if lines:
            for line in lines:
                x_data = np.array(line.get_xdata())
                x_data += val
                line.set_xdata(x_data)

        quadmesh_list = [x for x in self.get_children()\
                            if isinstance(x,mpl.collections.QuadMesh)]
        quiver_list = [x for x in self.get_children()\
                            if isinstance(x,mpl.quiver.Quiver)]
        if quadmesh_list:
            for quadmesh in quadmesh_list:
                quadmesh._coordinates[:,:,0] += val

        if quiver_list:
            for i,_ in enumerate(quiver_list):
                quiver_list[i].X += val
                quiver_list[i].XY = np.column_stack((quiver_list[i].X,quiver_list[i].Y))
                offsets = np.asanyarray(quiver_list[i].XY,float)
                if offsets.shape == (2,):
                    offsets = offsets[None,:]
                quiver_list[i]._offsets = offsets
                quiver_list[i]._transOffset = quiver_list[i].transform
                
        if quadmesh_list or quiver_list or lines:
            xlim = [x+val for x in self.get_xlim()]
            self.set_xlim(xlim)
        

    def shift_yaxis(self,val,shared=True,use_gcl=False):
        if shared and not use_gcl:
            lines = self.get_shared_lines()
        else:
            lines = self.get_lines()[-1:] if use_gcl else self.get_lines()

        if lines:
            for line in lines:
                y_data = np.array(line.get_ydata())
                y_data += val
                line.set_ydata(y_data)

            
        quadmesh_list = [x for x in self.get_children()\
                            if isinstance(x,mpl.collections.QuadMesh)]

        quiver_list = [x for x in self.get_children()\
                            if isinstance(x,mpl.quiver.Quiver)]

        if quadmesh_list:
            for quadmesh in quadmesh_list:
                quadmesh._coordinates[:,:,1] += val

        if quiver_list:
            for i,_ in enumerate(quiver_list):
                quiver_list[i].Y += val
                quiver_list[i].XY = np.column_stack((quiver_list[i].X,quiver_list[i].Y))
                offsets = np.asanyarray(quiver_list[i].XY,float)
                if offsets.shape == (2,):
                    offsets = offsets[None,:]
                quiver_list[i]._offsets = offsets
                quiver_list[i]._transOffset = quiver_list[i].transform

        if quadmesh_list or quiver_list or lines:
            ylim = [y+val for y in self.get_ylim()]
            self.set_ylim(ylim)


    def shift_legend_val(self,val,comp=None):
        leg = self.get_legend()
        leg_text = leg.get_texts()
        for text in leg_text:
            string = self._shift_text(text.get_text(),val,comp)
            text.set_text(string)
    def shift_title_val(self,val,loc='center',comp=None):
        title = self._shift_text(self.get_title(loc=loc),val,comp)
        self.set_title(title,loc=loc)

    def _shift_text(self, string,val,comp):
        list_string = string.split("=")
        for i in range(1,len(list_string)):
            try:
                old_val = float(list_string[i])
            except ValueError:
                for j in range(1,len(list_string[i])):
                    try:
                        old_val = float(list_string[i][:-j])
                        if comp is None:
                            list_string[i]=list_string[i].replace(list_string[i][:-j],
                                            "=%.3g"%(old_val+val))
                        elif comp in list_string[i-1]:
                            list_string[i]=list_string[i].replace(list_string[i][:-j],
                                            "=%.3g"%(old_val+val))
                        else:
                            list_string[i]=list_string[i].replace(list_string[i][:-j],
                                            "=%.3g"%old_val)

                        break
                    except ValueError:
                        if j == len(list_string[i])-1:
                            msg="Legend label not compatible with this function"
                            warnings.warn(msg+": %s"%string,stacklevel=2)
                            return string
                        continue
        string = "".join(list_string)
        return string

class Axes3DCHAPSim(Axes3D):
    name='Axes3DCHAPSim'
    def shift_xaxis(self,val):
        for collection in self.collections:
            collection._vec[0] +=val

        lims = [lim+val for lim in self.get_xlim()]
        self.set_xlim(lims)

    def shift_yaxis(self,val):
        for collection in self.collections:
            collection._vec[1] +=val

        lims = [lim+val for lim in self.get_ylim()]
        self.set_ylim(lims)

    def shift_zaxis(self,val):
        for collection in self.collections:
            collection._vec[2] +=val

        lims = [lim+val for lim in self.get_zlim()]
        self.set_zlim(lims)

    def plot_isosurface(self,X,Y,Z,data,level,scaling=None,**kwargs):
        if not _has_isosurface:
            msg = "This function cannot be used, see previous warning"
            raise RuntimeError(msg)
        
        had_data = self.has_data()

        if scaling is not None:
            if len(scaling) !=3:
                msg = "The length of the scaling array must be 3"
                raise ValueError(msg)

            X = X[::scaling[0],::scaling[1],::scaling[2]].copy()
            Y = Y[::scaling[0],::scaling[1],::scaling[2]].copy()
            Z = Z[::scaling[0],::scaling[1],::scaling[2]].copy()
            data = data[::scaling[0],::scaling[1],::scaling[2]].copy()

        old_verts, faces, normals, values = measure.marching_cubes(data, level=level)
        
        def get_var_index(array):
            avail_arr = []
            for dim in range(array.ndim):
                slice_list = [0]*array.ndim
                slice_list[dim] = slice(None)
                arr_single_dim = array[tuple(slice_list)]

                if all(arr_single_dim == arr_single_dim[0]):
                    continue
                elif not all(np.diff(arr_single_dim)>0):
                    msg = "array must be strictly increasing"
                    raise ValueError(msg)
                else:
                    #print(arr_single_dim)
                    avail_arr.append((tuple(slice_list),dim))

            if len(avail_arr) ==0 :
                msg = "array needs to be not constant in one direction"
                raise ValueError(msg)
            elif len(avail_arr) >1:
                msg = "array needs to be constant in ojust ne direction"
                raise ValueError(msg)
            else:
                return avail_arr[0]

        indexer1,dim1 = get_var_index(X)
        indexer2,dim2 = get_var_index(Y)
        indexer3,dim3 = get_var_index(Z)
        indexer_list = [indexer1,indexer2,indexer3]
        dim_list = [dim1,dim2,dim3]

        for indexer,array in zip(indexer_list,[X,Y,Z]):
            if isinstance(indexer[0],slice):
                xt = array[indexer]
            elif isinstance(indexer[1],slice):
                yt = array[indexer]
            elif isinstance(indexer[2],slice):
                zt = array[indexer]
            else:
                raise Exception

        xinterp = interp1d(np.arange(xt.size),xt)
        yinterp = interp1d(np.arange(yt.size),yt)
        zinterp = interp1d(np.arange(zt.size),zt)

        verts = np.zeros_like(old_verts)
        verts[:,dim1] = xinterp(old_verts[:,0])
        verts[:,dim2] = yinterp(old_verts[:,1])
        verts[:,dim3] = zinterp(old_verts[:,2])
        # print(scaling)

        Tri = mtri.Triangulation(verts[:,0],verts[:,1])

        # print(X.shape)
        # print(verts[faces])
        mesh = self.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],triangles=Tri.triangles, **kwargs)
        # self.add_collection(mesh)
        # self.auto_scale_xyz(X,Y, Z, had_data)

        return mesh

mpl.projections.register_projection(AxesCHAPSim)
mpl.projections.register_projection(Axes3DCHAPSim)

def flip_leg_col(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def figure(*args,**kwargs):
    if 'FigureClass' in kwargs.keys():
        warnings.warn("FigureClass keyword overriden with CHAPSimFigure\n")
    kwargs['FigureClass'] = CHAPSimFigure
    return plt.figure(*args,**kwargs)

def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None,*args, **fig_kw):
    fig = plt.figure(FigureClass=CHAPSimFigure,*args,**fig_kw)
    if subplot_kw is None:
        subplot_kw = {'projection':'AxesCHAPSim'}

    ax=fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, 
                    subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    return fig, ax

def _default_update_new(name,type_kw,**kwargs):
    if type_kw is None:
        contour_kw = {}
    elif not isinstance(type_kw,dict):
        raise TypeError(f"{name} needs to be a dictionary")

    for key, val in kwargs.items():
        if key not in type_kw.keys():
            type_kw[key] = val    

    return type_kw

def _default_update_replace(name,type_kw,**kwargs):
    if type_kw is None:
        type_kw = {}
    elif not isinstance(type_kw,dict):
        raise TypeError(f"{name} needs to be a dictionary")

    type_kw.update(kwargs)

    return type_kw

def update_pcolor_kw(pcolor_kw,**kwargs):
    
    pcolor_kw = _default_update_replace('pcolor_kw',pcolor_kw,
                                cmap = 'jet',
                                shading = 'gouraud',
                                **kwargs)
    return pcolor_kw

def update_contour_kw(contour_kw,**kwargs):
    return _default_update_replace('contour_kw',contour_kw,**kwargs)

def get_contour_func_params(ax, contour_kw, plot_func='pcolormesh',**kwargs):
    kw_copy = copy.deepcopy(contour_kw)
    
    if isinstance(kw_copy,dict):
        plot_func = kw_copy.pop('plot_func',plot_func)

    if hasattr(ax,plot_func):
        options = ['pcolormesh', 'contour', 'contourf']
        if plot_func not in options:
            msg = "An available function was not selected"
            raise AttributeError(msg)
    
        plot_func = getattr(ax,plot_func)
        
    else:
        msg = f"The default function must be an attribute of {type(ax)}"
        raise AttributeError(msg)

    if plot_func.__name__ == 'pcolormesh':
        kw_copy = update_pcolor_kw(kw_copy,**kwargs)
    else:
        kw_copy = update_contour_kw(kw_copy,**kwargs)
        
    return plot_func, kw_copy
    
def update_mesh_kw(mesh_kw,**kwargs):
    if mesh_kw is None:
        mesh_kw = {}

    for key, val in kwargs.items():
        if key not in mesh_kw.keys():
            mesh_kw[key] = val
    
    return mesh_kw

def update_quiver_kw(quiver_kw,**kwargs):
    return _default_update_new('quiver_kw',quiver_kw,**kwargs)

def update_line_kw(line_kw,**kwargs):
    return  _default_update_new('line_kw',line_kw,**kwargs)


def update_subplots_kw(subplots_kw,**kwargs):  
    return _default_update_new('subplots_kw',subplots_kw,**kwargs)

def _check__mpl_kwargs(artist,kwargs):
    kw_copy = copy.deepcopy(kwargs)
    kw_copy.pop('axes',None)
    for k in kw_copy:
        func = getattr(artist,f"set_{k}",None)
        if not callable(func):
            name = artist.__class__.__name__
            msg = (f"Artist of type {name}"
                    f" has no property {k}. You may"
                    " have passed an incorrect keyword")
            raise AttributeError(msg)
            
        
def create_fig_ax_with_squeeze(fig=None,ax=None,**kwargs):

    if fig is None:
        fig, ax = subplots(**kwargs)
    elif ax is None:
        ax=fig.add_subplot(1,1,1)
    else:
        fig = _upgrade_fig(fig)
        ax = _upgrade_ax(fig, ax)
    # _check__mpl_kwargs(fig,kwargs)
    
    return fig, ax

    
def create_fig_ax_without_squeeze(*args,fig=None,ax=None,**kwargs):
    kwargs['squeeze'] = False
    if fig is None:
        fig, ax = subplots(*args,**kwargs)
    elif ax is None:
        kwargs.pop('figsize',None)
        ax=fig.subplots(*args,**kwargs)
    
    fig = _upgrade_fig(fig)
    
    single_input = False
    if isinstance(ax,mpl.axes.Axes):
        ax = np.array([ax])
        single_input = True
    elif all([isinstance(a,mpl.axes.Axes) for a in ax.flatten()]):
        ax = np.array(ax)
    else:   
        msg = ("Axes provided to method must be of type "
                f"{mpl.axes.Axes.__name__}  or an iterable"
                f" of it not {type(ax)}")
        raise TypeError(msg)
    
    ax = ax.flatten()
    for i, a in enumerate(ax):
        ax[i] = _upgrade_ax(fig,a)
    
    # _check__mpl_kwargs(fig,kwargs)
    
    return fig, ax, single_input

def _upgrade_fig(fig):
    if not isinstance(fig, CHAPSimFigure):
        if isinstance(fig,mpl.figure.Figure):
            d = fig.__getstate__()
            fig = CHAPSimFigure()
            fig.__setstate__(d)
        else:
            msg = f"fig needs to be an instance of {CHAPSimFigure.__name__}"
            raise TypeError(msg)
    
    return fig

def _upgrade_ax(fig,ax):
    fig = _upgrade_fig(fig)
    if not isinstance(ax,(AxesCHAPSim, Axes3DCHAPSim)):
        if isinstance(ax,mpl.axes.Axes):
            d = ax.__getstate__().copy()
            #ax.remove()
            ax = fig.add_subplot(1,1,1)
            ax.__setstate__(d)
        elif isinstance(ax,Axes3D):
            d = ax.__getstate__().copy()
            #ax.remove()
            ax = fig.add_subplot(1,1,1,projection='3d')
            ax.__setstate__(d)
        else:
            msg = f"ax needs to be an instance of {AxesCHAPSim.__name__}"
            raise TypeError(msg)
    return ax
    
def get_legend_ncols(line_no):
    return 4 if line_no >3 else line_no

def close(*args,**kwargs):
    plt.close(*args,**kwargs)

def create_general_video(fig,path_to_folder,func,abs_path=True,func_args=None,func_kw=None,times=None,**kwargs):

    
    if times is None:
        times= misc_utils.time_extract(path_to_folder,abs_path)
    else:
        all_times = misc_utils.time_extract(path_to_folder,abs_path)
        if not all(time in all_times for time in times):
            msg = "Not all tims given are in the results folder"
            raise RuntimeError(msg)
        
    times.sort()

    if cp.rcParams["TEST"]:
        times = times[-10:]

    if func_args is None:
        func_args=()

    if func_kw is None:
        func_kw={}


    def animate(time):
        return func(fig,time,*func_args,**func_kw)

    return FuncAnimation(fig,animate,frames=times,**kwargs)

