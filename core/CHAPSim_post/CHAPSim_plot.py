'''
# CHAPSim_plot
This is a postprocessing module for CHAPSim_post library
'''
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolor
import numpy as np
from scipy import io
import os 
import subprocess
import warnings
import itertools
from shutil import which
from cycler import cycler
import itertools
import sys
try:
    import pyvista as pv
    import pyvistaqt as pvqt
except ImportError:
    warnings.warn("PyVista module unavailable")

import CHAPSim_post as cp
import CHAPSim_post.CHAPSim_Tools as CT
if which('lualatex') is not None:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['pgf.texsystem'] = 'lualatex'
    mpl.rcParams['text.latex.preamble'] =r'\usepackage{amsmath}'

def update_prop_cycle(**kwargs):
    avail_keys = [x[4:] for x in mpl.lines.Line2D.__dict__.keys() if x[0:3]=='get']

    if not all([key in avail_keys for key in kwargs.keys()]):
        msg = "The key is invalid for the matplotlib property cycler"
        raise ValueError(msg)
    
    cycler_dict = mpl.rcParams['axes.prop_cycle'].by_key()
    for key, item in kwargs.items():
        if not hasattr(item,"__iter__"):
            item = [item]
        cycler_dict[key] = item

    item_length = [ len(item) for _,item in cycler_dict.items()]
    cycle_length = np.lcm.reduce(item_length)

    for key, val in cycler_dict.items():
        cycler_dict[key] = list(val)*int(cycle_length/len(val))
    mpl.rcParams['axes.prop_cycle'] = cycler(**cycler_dict)

def reset_prop_cycler():
    update_prop_cycle(linestyle=['-','--','-.',':'],
                marker=['x','.','v','^','+'],
                color = 'bgrcmyk')

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
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


class CHAPSimFigure(mpl.figure.Figure):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.__clegend = None
    def clegend(self,*args, **kwargs):
        self.__clegend = super().legend(*args, **kwargs)
        return self.__clegend
    def add_subplot(self,*args, **kwargs):
        kwargs['projection']='AxesCHAPSim'
        return super().add_subplot(*args,**kwargs)
    def c_add_subplot(self,*args, **kwargs):
        kwargs['projection']='AxesCHAPSim'
        return super().add_subplot(*args,**kwargs)
    def get_legend(self):
        return self.__clegend
    def update_legend_fontsize(self, fontsize,tight_layout=True,**kwargs):
        if self.__clegend is not None:
            texts=self.__clegend.get_texts()
            for text in texts:
                text.set_fontsize(fontsize)
        if tight_layout:
            self.get_axes()[0].get_gridspec().tight_layout(self,**kwargs)
    
    def set_axes_title_fontsize(self, fontsize):
        axes = self.get_axes()
        for ax in axes:
            ax.set_title_fontsize(fontsize)

class AxesCHAPSim(mpl.axes.Axes):
    name='AxesCHAPSim'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def cplot(self,*args, **kwargs):

        counter = self.count_lines()

        plot_kw = {}
        for key,val in mpl.rcParams['axes.prop_cycle'].by_key().items():
            plot_kw[key] = val[counter%len(val)]
        for key,val in kwargs.items():
            plot_kw[key] = val
        if 'markevery' not in plot_kw:
            plot_kw['markevery'] = 10
        return super().plot(*args,**plot_kw)
    def count_lines(self):
        no_lines = 0
        twinned_ax = self._twinned_axes.get_siblings(self)
        for ax in twinned_ax:
            no_lines += len(ax.get_lines())
        return no_lines

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
            self.get_gridspec().tight_layout(self.get_figure(),**kwargs)

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

    def normalise(self,axis,val):
        i=0
        for line in self.get_lines():
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
            i+=1

        lim_val = max(val) if hasattr(val,"__iter__") else val
        if self.get_lines():
            if axis == 'x':
                xlims = [x/lim_val for x in self.get_xlim()]
                self.set_xlim(xlims)
            else:
                ylims = [y/lim_val for y in self.get_ylim()]
                self.set_ylim(ylims)

    def array_normalise(self,axis,val):
        i=0

        for line in self.get_lines():
            xdata, ydata = line.get_data()
            if xdata.size != len(val):
                raise ValueError("The size of val must be the same as the data")
            if axis=='x':
                xdata =  np.array(xdata)/val
            else:
                ydata =  np.array(ydata)/val
            line.set_data(xdata, ydata)
            i+=1
        self.relim()
        self.autoscale_view(True,True,True)

    def apply_func(self,axis,func):
        lines = self.get_lines()
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

    def shift_xaxis(self,val):
        lines = self.get_lines()
        lines = self.get_lines()
        if lines:
            for line in lines:
                x_data = line.get_xdata().copy()
                x_data += val
                line.set_xdata(x_data)
        # if lines:
        #     for line in lines:
        #         x_data = line.get_xdata().copy()
        #         x_data += val
        #         line.set_xdata(x_data)
        #     if (x_data>self.get_xlim()[0]).any() and (x_data<self.get_xlim()[1]).any(): 
        #         xlim = [x+val for x in self.get_xlim()]
        #         self.set_xlim(*xlim)
        #     else:
        #         self.relim()
        #         self.autoscale_view(True,True,True)
        # else:
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
        

    def shift_yaxis(self,val):
        lines = self.get_lines()
        if lines:
            for line in lines:
                y_data = line.get_ydata().copy()
                y_data += val
                line.set_ydata(y_data)
            # if (y_data>self.get_ylim()[0]).all() and (y_data<self.get_ylim()[1]).all(): 
            #     ylim = [x+val for x in self.get_ylim()]
            #     self.set_ylim(ylim)
            # else:
            #     self.relim()
            #     self.autoscale_view(True,True,True)

            
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


    def shift_legend_val(self,val):
        leg = self.get_legend()
        leg_text = leg.texts
        for text in leg_text:
            string = self._shift_text(text.get_text(),val)
            text.set_text(string)
    def shift_title_val(self,val,loc='center'):
        title = self._shift_text(self.get_title(loc=loc),val)
        self.set_title(title,loc=loc)
    def _shift_text(self, string,val):
        list_string = string.split("=")
        for i in range(1,len(list_string)):
            try:
                old_val = float(list_string[i])
            except ValueError:
                for j in range(1,len(list_string[i])):
                    try:
                        old_val = float(list_string[i][:-j])
                        list_string[i]=list_string[i].replace(list_string[i][:-j],
                                            "=%.3g"%(old_val+val))
                        break
                    except ValueError:
                        if j == len(list_string[i])-1:
                            msg="Legend label not compatible with this function"
                            warnings.warn(msg+": %s"%string,stacklevel=2)
                            return string
                        continue
        string = "".join(list_string)
        return string

def flip_leg_col(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

mpl.projections.register_projection(AxesCHAPSim)

def figure(*args,**kwargs):
    if 'FigureClass' in kwargs.keys():
        warnings.warn("FigureClass keyword overriden with CHAPSimFigure\n")
    kwargs['FigureClass'] = CHAPSimFigure
    return plt.figure(*args,**kwargs)

def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None,*args, **fig_kw):
    fig = plt.figure(FigureClass=CHAPSimFigure,*args,**fig_kw)
    if subplot_kw is None:
        subplot_kw = {'projection':'AxesCHAPSim'}
    else:
        subplot_kw['projection'] = 'AxesCHAPSim'
    ax=fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, 
                    subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    return fig, ax

def update_pcolor_kw(pcolor_kw):
    if pcolor_kw is None:
        pcolor_kw = {'cmap':'jet','shading':'gouraud'}
    else:
        if not isinstance(pcolor_kw,dict):
            msg = f"pcolor_kw must be None or a dict not a {type(pcolor_kw)}"
            raise TypeError(msg)
        if 'cmap' not in pcolor_kw.keys():
            pcolor_kw['cmap'] = 'jet'
        if 'shading' not in pcolor_kw.keys():
            pcolor_kw['shading'] = 'gouraud'
    return pcolor_kw

def update_quiver_kw(quiver_kw):
    if quiver_kw is not None:
        if 'angles' in quiver_kw.keys():
            del quiver_kw['angles']
        if 'scale_units' in quiver_kw.keys():
            del quiver_kw['scale_units']
        if 'scale' in quiver_kw.keys():
            del quiver_kw['scale']
    else:
        quiver_kw = {}

    return quiver_kw

def update_line_kw(line_kw,**kwargs):
    if line_kw is None:
        line_kw = {}
    elif not isinstance(line_kw,dict):
        raise TypeError("line_kw needs to be a dictionary")

    for key, val in kwargs.items():
        if key not in line_kw.keys():
            line_kw[key] = val    

    return line_kw

def update_contour_kw(contour_kw,**kwargs):
    if contour_kw is None:
        contour_kw = {}
    elif not isinstance(contour_kw,dict):
        raise TypeError("line_kw needs to be a dictionary")

    for key, val in kwargs.items():
        if key not in contour_kw.keys():
            contour_kw[key] = val    

    return contour_kw


def update_subplots_kw(subplots_kw,**kwargs):
    if subplots_kw is None:
        subplots_kw = {}

    for key, val in kwargs.items():
        if key not in subplots_kw.keys():
            subplots_kw[key] = val    

    return subplots_kw

def create_fig_ax_with_squeeze(fig=None,ax=None,**kwargs):
    
    if fig is None:
        fig, ax = subplots(**kwargs)
    elif ax is None:
        ax=fig.add_subplot(1,1,1)
    else:
        if not isinstance(fig, CHAPSimFigure):
            msg = f"fig needs to be an instance of {CHAPSimFigure.__name__}"
            raise TypeError(msg)
        if not isinstance(ax,AxesCHAPSim):
            msg = f"ax needs to be an instance of {AxesCHAPSim.__name__}"
            raise TypeError(msg)
    
    return fig, ax

def create_fig_ax_without_squeeze(*args,fig=None,ax=None,**kwargs):
    if fig is None:
        kwargs['squeeze'] = False
        fig, ax = subplots(*args,**kwargs)
    elif ax is None:
        if 'subplot_kw' in kwargs:
            kwargs['subplot_kw'].update({'squeeze':False})
        else:
            kwargs['subplot_kw'] = {'squeeze':False}

        kwargs.pop('figsize',None)
        ax=fig.subplots(*args,**kwargs)

    if isinstance(ax,mpl.axes.Axes):
        ax = np.array([ax])
    elif all([isinstance(a,mpl.axes.Axes) for a in ax.flatten()]):
        ax = np.array(ax)
    else:   
        msg = ("Axes provided to method must be of type "
                f"{mpl.axes.Axes.__name__}  or an iterable"
                f" of it not {type(ax)}")
        raise TypeError(msg)

    ax = ax.flatten()

    return fig, ax

def get_legend_ncols(line_no):
    return 4 if line_no >3 else line_no

def close(*args,**kwargs):
    plt.close(*args,**kwargs)

def create_general_video(fig,path_to_folder,abs_path,func,func_args=None,func_kw=None,time_range=None):

    times= CT.time_extract(path_to_folder,abs_path)
    if time_range is not None:
        times = list(filter(lambda x: x>time_range[0],times))
        times = list(filter(lambda x: x<time_range[1],times))
    times.sort()
    if cp.Params["TEST"]:
        times = times[-10:]

    if func_args is None:
        func_args=()

    if func_kw is None:
        func_kw={}


    def animate(time):
        return func(fig,time,*func_args,**func_kw)

    return mpl.animation.FuncAnimation(fig,animate,frames=times)


matlab_path =  which('matlab')
octave_path = which('octave')
if matlab_path is not None:
    try:
        try:
            import matlab.engine as me
            import matlab
        except ModuleNotFoundError:
            setup_path = os.path.join(os.path.dirname(matlab_path),
                                    '..','extern','engines','python',)
            cwd = os.path.dirname(sys.executable) 
            lib_dir = os.path.join(os.path.dirname(__file__),
                                                    '..','libs')               
            os.chdir(setup_path)
            conda_path = os.getenv('CONDA_PREFIX')
            subprocess.run([os.path.join(conda_path,'bin','python'),'setup.py','build',
                                        '--build-base='+lib_dir,'install',
                                        '--prefix='+conda_path])
            os.chdir(cwd)
        
            import matlab.engine as me
            import matlab
            print("## Successfully imported matlab engine\n")
    except Exception as e:
        warnings.warn("Error importing matlab: %s:\n CHAPSim_post cannot use matlab functionality\n"%e,ImportWarning)
if octave_path is not None:
    import oct2py

if matlab_path is None and octave_path is None:
    warnings.warn("Cannot process isosurfaces Matlab or GNU Octave are not installed ")

module = sys.modules[__name__]

module.useMATLAB =True

module.matlab_ref_count=0
module.oct_ref_count=0
module.eng = None

class mPlotEngine():

    @staticmethod
    def inc_ref_count():
        if module.useMATLAB:
            module.matlab_ref_count +=1
            if module.matlab_ref_count ==1:
                module.eng = me.start_matlab()
        else:
            module.oct_ref_count +=1
            if module.oct_ref_count ==1:
                module.eng = oct2py.Oct2Py()
    @staticmethod
    def dec_ref_count():
        if module.useMATLAB:
            module.matlab_ref_count -=1
            if module.matlab_ref_count ==0:
                mPlotEngine.quit()
        else:
            module.oct_ref_count -=1
            if module.oct_ref_count ==0:
                mPlotEngine.quit()
    @staticmethod
    def quit():
        try:
            if module.useMATLAB:
                if module.matlab_ref_count ==0:
                    eng.quit()       
            else:
                if module.oct_ref_count ==0:
                    eng.exit()
        except AttributeError:
            warnings.warn(".m engine changed during "+\
                "runtime cannot quit %s engine" %"octave"\
                            if  useMATLAB else "matlab")
                

mEngine = mPlotEngine()


class matlabFigure():
    def __init__(self, **kwargs):
        mEngine.inc_ref_count()

        args = []
        if 'visible' not in kwargs.keys():
            kwargs['visible'] = 'off'
        kwargs['Units']= 'inches' 
        if 'figsize' in kwargs.keys():
            kwargs['Position'] = matlab.double([0,0,*kwargs['figsize']])
            kwargs.pop('figsize')
            
        for key, val in kwargs.items():
            args.extend([key,val])
        self._figure = eng.figure(*args)
        list_pair = [ [args[i], args[i+1]] for i in range(0,len(args),2)]
        attr_dict = { x[0]:x[1] for x in list_pair}
        for key, val in attr_dict.items():
            setattr(self,key,val)

        self._matlab_objs=[]

    def Axes(self,*args):
        axes = matlabAxes(self,*args)
        self._matlab_objs.append(axes)
        return axes

    def savefig(self,filename,Resolution=mpl.rcParams['figure.dpi']):
        res='-r'+str(Resolution)
        eng.print(self._figure,res,'-dpng',filename,nargout=0)
    def __del__(self):
        mEngine.dec_ref_count()
    def subplots(self,cols=1,rows=1,squeeze=True,**kwargs):
        axes = []
        args=[]
        for key, val in kwargs.items():
            args.extend([key,val])
        
        for i in range(1,1+cols*rows):
            axes.append(self.Axes(*args))
            axes[i-1]._axes = eng.subplot(float(cols),float(rows),float(i),axes[i-1]._axes)
        if squeeze and len(axes)==1:
            return axes[0]
        else:
            return np.array(axes)
class matlabAxes():
    def __init__(self, fig,*args):
        assert type(fig) == matlabFigure, 'fig must be must be of type %s'%matlabFigure
        self._axes = eng.axes(fig._figure,*args)

        mEngine.inc_ref_count()

        list_pair = [ [args[i], args[i+1]] for i in range(0,len(args),2)]
        attr_dict = { x[0]:x[1] for x in list_pair}
        for key, val in attr_dict.items():
            setattr(self,key,val)
        self._matlab_objs=[]

    def plot(self,*args):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i],np.ndarray):
                args[i] = matlab.double(args[i].tolist())
        self._matlab_objs.append(eng.plot(self._axes,*args))
        eng.hold(self._axes,'on',nargout=0)
    
    def plot_isosurface(self,X,Y,Z,V,isovalue,color,*args):
        temp_mat = ".temp.mat"
        data_dict = {'x':X,'y':Y,'z':Z,'V':V}
        io.savemat(temp_mat,mdict=data_dict)
        colors = ['blue','green','red','yellow','magenta','cyan','black']
        if not color:
            color = colors[len(self._matlab_objs)%len(colors)]
        self.make_3d()
        bin_path = os.path.dirname(os.path.realpath(__file__))
        eng.addpath(bin_path)

        patch = matlabPatch.from_matlab(self,eng.plot_isosurface(self._axes,isovalue,color,*args))#matlabPatch(self,eng.isosurface(*args))
        # patch.set_color()
        eng.hold(self._axes,'on',nargout=0)
        self._matlab_objs.append(patch)
        os.remove(temp_mat)
        return patch
    def add_lighting(self):
        eng.camlight(self._axes,nargout=0)
        eng.lighting(self._axes,'gouraud',nargout=0)
    def get_objects(self):
        return self._matlab_objs
    def make_3d(self):
        eng.view(self._axes,3,nargout=0)
    def set_view(self,view,*args):
        view = matlab.double(view)
        view_ret = eng.view(self._axes,view,*args)
        setattr(self,'view',view_ret)
    def set_ratios(self,ratio):
        ratio = matlab.double(ratio)
        eng.pbaspect(self._axes,ratio,nargout=0)
    def __del__(self):
        mEngine.dec_ref_count()

class matlabPatch():
    def __init__(self,ax,*args,from_matlab=False):
        
        if from_matlab:
            self._patch = args[0]
        else:
            assert type(ax) == matlabAxes, 'ax must be must be of type %s'%matlabAxes
            self._patch =  eng.patch(ax._axes,*args)
        self._ax = ax._axes
        
        mEngine.inc_ref_count()
        
    
        # not_Args = itertools.takewhile(lambda x: type(x)!=string,args)
        # args = list(set(args).difference(not_Args))
        # list_pair = [ [args[i], args[i+1]] for i in range(0,len(args),2)]
        # attr_dict = { x[0]:x[1] for x in list_pair}
        # for key, val in attr_dict.items():
        #     setattr(self,key,val)


        self._matlab_objs=[]
    @classmethod
    def from_matlab(cls,*args):
        return cls(*args,from_matlab=True)
    def set_color(self,color):
        # eng.colormap(self._ax,'hot')
        eng.set(self._patch,'FaceColor',color.title(),'EdgeColor','none')

    def __del__(self):
        mEngine.dec_ref_count()

class octaveFigure():
    def __init__(self, *args):
        mEngine.inc_ref_count()
        
        self._figure = eng.figure(*args)
        list_pair = [ [args[i], args[i+1]] for i in range(0,len(args),2)]
        attr_dict = { x[0]:x[1] for x in list_pair}
        for key, val in attr_dict.items():
            setattr(self,key,val)

        self._mObjs=[]

    def Axes(self,*args):
        axes = octaveAxes(self,*args)
        self._mObjs.append(axes)
        return axes
    def savefig(self,filename):
        eng.saveas(self._figure,filename)
    def __del__(self):
        mEngine.dec_ref_count()


class octaveAxes():
    def __init__(self,fig,*args):
        mEngine.inc_ref_count()
        self._axes = eng.axes(*args)

        list_pair = [ [args[i], args[i+1]] for i in range(0,len(args),2)]
        attr_dict = { x[0]:x[1] for x in list_pair}
        for key, val in attr_dict.items():
            setattr(self,key,val)
        self._mObjs=[]

    def plot(self,*args):
        mAx = eng.plot(*args)
        self._mObjs.append(mAx)
        return mAx

    def plot_isosurface(self,*args):
        p=eng.patch(self._axes,eng.isosurface(*args))
        self._mObjs.append(p)
        eng.set(p,'FaceColor','interp','EdgeColor','none')
        cmap = eng.colormap('hot')
        return p
    def __del__(self):
        mEngine.dec_ref_count()
        
if 'pyvistaqt' in sys.modules:
    class vtkFigure(pv.Plotter):
        def __init__(self,*args,**kwargs):
            if 'notebook' not in kwargs.keys():
                kwargs['notebook'] = False
            super().__init__(*args, **kwargs)
            self.grid = None
            self.set_background('white')
            self.show_bounds(color='black')

        def plot_isosurface(self,X,Y,Z,V,isovalue,color=None,*args):

            self.grid = pv.StructuredGrid(X,Y,Z)
            if self.grid is not None:
                num = len(self.grid.cell_arrays.keys())
            else:
                num = 1
            self.grid.cell_arrays['iso_%d'%num] = V.flatten()
            pgrid = self.grid.cell_data_to_point_data()
            color_list = ['Greens_r','Blues_r','Reds_r' ]
            if color is not None:
                color = color.title() + "s_r"
            else:
                color = color_list[num%len(color_list)]

            contour = pgrid.contour(isosurfaces=1,scalars='iso_%d'%num,
                                    preference='point',rng=(isovalue,isovalue))

            self.add_mesh(contour,interpolate_before_map=True,cmap=color)
            # self.remove_scalarbar()

        
        def savefig(self,filename):
            self.show(screenshot=filename)

class mCHAPSimAxes(matlabAxes,octaveAxes):
    def __new__(cls,*args,**kwargs):
        if module.useMATLAB:
            if 'matlab.engine' not in sys.modules:
                raise ModuleNotFoundError("Matlab not found,"+\
                                " cannot use this functionality")
            return matlabAxes(*args,**kwargs)
        else:
            if 'oct2py' not in sys.modules:
                raise ModuleNotFoundError("Octave not found,"+\
                                " cannot use this functionality")
            return octaveAxes(*args,**kwargs)

class mCHAPSimFigure(matlabFigure,octaveFigure):
    def __new__(cls,*args,**kwargs):
        if module.useMATLAB:
            if 'matlab.engine' not in sys.modules:
                raise ModuleNotFoundError("Matlab not found,"+\
                                " cannot use this functionality")
            return matlabFigure(*args,**kwargs)
        else:
            if 'oct2py' not in sys.modules:
                raise ModuleNotFoundError("Octave not found,"+\
                                " cannot use this functionality")
            return octaveFigure(*args,**kwargs)


