'''
# CHAPSim_plot
This is a postprocessing module for CHAPSim_post library
'''
import matplotlib as mpl 
import matplotlib.pyplot as plt
from shutil import which
from cycler import cycler
import itertools

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.rcParams['lines.markerfacecolor'] = 'white'
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['legend.edgecolor'] = 'inherit'
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'small'

legend_fontsize=12

if which('lualatex') is not None:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['pgf.texsystem'] = 'lualatex'

class Figure(mpl.figure.Figure):
    def clegend(self,*args, **kwargs):
        return super().legend(*args, **kwargs)
    def add_subplot(self,*args, **kwargs):
        kwargs['projection']='AxesCHAPSim'
        return super().add_subplot(*args,**kwargs)
    def c_add_subplot(self,*args, **kwargs):
        kwargs['projection']='AxesCHAPSim'
        return super().add_subplot(*args,**kwargs)
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
        if 'fontsize' not in kwargs.keys():
            kwargs['fontsize']=legend_fontsize
        if 'loc' not in kwargs.keys() and 'bbox_to_anchor' not in kwargs.keys():
            kwargs['loc'] = 'lower center'
            kwargs['bbox_to_anchor'] = (0.5,1.02)
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

    def set_label_fontsize(self,fontsize):
        xlabel_str=self.get_xlabel()
        ylabel_str=self.get_ylabel()

        self.set_xlabel(xlabel_str,fontsize=fontsize)
        self.set_ylabel(ylabel_str,fontsize=fontsize)
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

        self.clegend(*args,**kwargs)
    def ctwinx(self):
        return super().twinx()



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
        
        self.clegend(*args,**kwargs)

def flip_leg_col(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

mpl.projections.register_projection(AxesCHAPSim)

def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None,*args, **fig_kw):
    fig = plt.figure(FigureClass=Figure,*args,**fig_kw)
    if subplot_kw is None:
        subplot_kw = {'projection':'AxesCHAPSim'}
    else:
        subplot_kw['projection'] = 'AxesCHAPSim'
    ax=fig.subplots(nrows, ncols, sharex, sharey, squeeze, subplot_kw, gridspec_kw)
    return fig, ax