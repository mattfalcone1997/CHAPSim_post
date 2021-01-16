
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler
from shutil import which

from .mpl_class import CHAPSimFigure, AxesCHAPSim, subplots

import CHAPSim_post as cp
from CHAPSim_post.utils import misc_utils


if which('lualatex') is not None:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['pgf.texsystem'] = 'lualatex'
    mpl.rcParams['text.latex.preamble'] =r'\usepackage{amsmath}'

def update_prop_cycle(**kwargs):
    avail_keys = [x[4:] for x in mpl.lines.Line2D.__dict__.keys() if x[0:3]=='get']

    if not all([key in avail_keys for key in kwargs.keys()]):
        msg = "The key is invalid for the matplotlib property cycler"
        raise ValueError(msg)

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
    
    cycler_dict = mpl.rcParams['axes.prop_cycle'].by_key()
    for key, item in kwargs.items():
        if not hasattr(item,"__iter__"):
            item = [item]
        elif isinstance(item,str):
            if item == "" :
                item = [item]
        if key in alias_dict.keys():
            key = alias_dict[key]
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

    line_kw = line_kw.copy()

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
    kwargs['squeeze'] = False
    if fig is None:
        fig, ax = subplots(*args,**kwargs)
    elif ax is None:
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

    times= misc_utils.time_extract(path_to_folder,abs_path)
    if time_range is not None:
        times = list(filter(lambda x: x>time_range[0],times))
        times = list(filter(lambda x: x<time_range[1],times))
    times.sort()
    if cp.rcParams["TEST"]:
        times = times[-10:]

    if func_args is None:
        func_args=()

    if func_kw is None:
        func_kw={}


    def animate(time):
        return func(fig,time,*func_args,**func_kw)

    return mpl.animation.FuncAnimation(fig,animate,frames=times)