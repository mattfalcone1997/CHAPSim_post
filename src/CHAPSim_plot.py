'''
# CHAPSim_plot
This is a postprocessing module for CHAPSim_post library
'''
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np
import os 
import subprocess
import warnings
from shutil import which
from cycler import cycler
import itertools
import sys
try:
    import pyvista as pv
    import pyvistaqt as pvqt
except ImportError:
    warnings.warn("PyVista module unavailable")

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.rcParams['lines.markerfacecolor'] = 'white'
# mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['legend.edgecolor'] = 'inherit'
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.grid'] = True

legend_fontsize=12

if which('lualatex') is not None:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['pgf.texsystem'] = 'lualatex'
    mpl.rcParams['text.latex.preamble'] =r'\usepackage{amsmath}'

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
        self.__line_counter_c=0
    def cplot(self,*args, **kwargs):
        if 'marker' in kwargs.keys() or 'linestyle' in kwargs.keys():
            return super().plot(*args,**kwargs)
        else:
            linestyle=['-','--','-.',':']
            marker=['x','.','v','^','+']
            if 'markevery' not in kwargs.keys():
                kwargs['markevery'] = 10
            kwargs['linestyle'] = linestyle[self.__line_counter_c%len(linestyle)]
            kwargs['marker'] = marker[self.__line_counter_c%len(linestyle)]
        
            self.__line_counter_c+=1

            return super().plot(*args,**kwargs)

    def clegend(self,*args, **kwargs):
        # if 'fontsize' not in kwargs.keys():
        #     kwargs['fontsize']=legend_fontsize
        if 'loc' not in kwargs.keys() and 'bbox_to_anchor' not in kwargs.keys():
            kwargs['loc'] = 'lower center'
            kwargs['bbox_to_anchor'] = (0.5,1.08)
        if not kwargs.pop('vertical',False):
            ncol = kwargs['ncol'] if 'ncol' in kwargs.keys() else 1
            if len(args)==2:
                args[0] = flip_leg_col(args[0],ncol)
                args[1] = flip_leg_col(args[1],ncol)
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

    def set_title_fontsize(self ,fontsize,tight_layout=True,**kwargs):
        title_str=self.get_title()
        self.set_title(title_str ,fontsize=fontsize)
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
        if hasattr(val,"__iter__"):
            if len(val) != len(self.get_lines()):
                raise RuntimeError("The length of vals must be the same as the"+\
                                    "number of lines in an axis") 
        i=0
        for line in self.get_lines():
            if hasattr(val,"__iter__"):
                norm_val = val[i]
            else:
                norm_val = val
            xdata=0; ydata=0
            xdata, ydata = line.get_data()
            if axis=='x':
                xdata =  np.array(xdata)/norm_val
            else:
                ydata =  np.array(ydata)/norm_val
            line.set_data(xdata, ydata)
            i+=1
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

def flip_leg_col(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

mpl.projections.register_projection(AxesCHAPSim)

def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None,*args, **fig_kw):
    fig = plt.figure(FigureClass=CHAPSimFigure,*args,**fig_kw)
    if subplot_kw is None:
        subplot_kw = {'projection':'AxesCHAPSim'}
    else:
        subplot_kw['projection'] = 'AxesCHAPSim'
    ax=fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, 
                    subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)
    return fig, ax



matlab_path =  which('matlab')
octave_path = which('octave')
if matlab_path is not None:
    try:
        import matlab.engine as me
        import matlab
    except ImportError:
        setup_file = os.path.join(os.path.dirname(matlab_path),
                                '..','extern','engines','python','setup.py')
        subprocess.run(['python',setup_file,'install'])
        import matlab.engine as me
        import matlab
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
    def __init__(self, *args):
        mEngine.inc_ref_count()
        
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

    def savefig(self,filename):
        eng.saveas(self._figure,filename,nargout=0)
    def __del__(self):
        mEngine.dec_ref_count()

class matlabAxes():
    def __init__(self, fig, *args):
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
    
    def plot_isosurface(self,*args):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i],np.ndarray):
                args[i] = matlab.double(args[i].tolist())

        self.make_3d()
        bin_path = os.path.dirname(os.path.realpath(__file__))
        eng.addpath(bin_path)

        patch = matlabPatch(self,eng.isosurface(*args))
        patch.set_color()
        eng.hold(self._axes,'on',nargout=0)
        self._matlab_objs.append(patch)
        return patch
    def get_objects(self):
        return self._matlab_objs
    def make_3d(self):
        eng.view(self._axes,3,nargout=0)
    def set_view(self,*args):
        view_ret = eng.view(self._axes,*args)
        setattr(self,'view',view_ret)
    def set_ratios(self,ratio):
        eng.pbaspect(self._axes,ratio,nargout=0)
    def __del__(self):
        mEngine.dec_ref_count()

class matlabPatch():
    def __init__(self,ax,*args,from_matlab=False):
        assert type(ax) == matlabAxes, 'ax must be must be of type %s'%matlabAxes
        if from_matlab:
            self._patch = args[0]
        else:
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
    def from_matlab(cls,p):
        return cls(p,from_matlab=True)
    def set_color(self):
        eng.colormap(self._ax,'hot')
        eng.set(self._patch,'FaceColor','interp','EdgeColor','none')

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
    class vtkFigure(pvqt.BackgroundPlotter):
        def Axes(self):
            pass
        def savefig(self,filename):
            pass
if 'pyvista' in sys.modules:
    class vtkAxes(pv.StructuredGrid):
        def plot_isosurface(self,*args):
            pass

class mCHAPSimAxes(matlabAxes,octaveAxes):
    def __new__(cls,*args):
        if module.useMATLAB:
            if 'matlab.engine' not in sys.modules:
                raise ModuleNotFoundError("Matlab not found,"+\
                                " cannot use this functionality")
            return matlabAxes(*args)
        else:
            if 'oct2py' not in sys.modules:
                raise ModuleNotFoundError("Octave not found,"+\
                                " cannot use this functionality")
            return octaveAxes(*args)

class mCHAPSimFigure(matlabFigure,octaveFigure):
    def __new__(cls,*args):
        if module.useMATLAB:
            if 'matlab.engine' not in sys.modules:
                raise ModuleNotFoundError("Matlab not found,"+\
                                " cannot use this functionality")
            return matlabFigure(*args)
        else:
            if 'oct2py' not in sys.modules:
                raise ModuleNotFoundError("Octave not found,"+\
                                " cannot use this functionality")
            return octaveFigure(*args)


