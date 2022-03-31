
import re
from .misc_utils import Params


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

class styleParameters:
    def __init__(self):
        self.ystyle_channel = y_styler(default_channel_ytransform,
                                        default_channel_ylims)

        self.ystyle_pipe = y_styler(default_pipe_ytransform,
                                    default_pipe_ylims)

        self.timeStyle = defaultTimeStyle
        self.CoordLabel_channel = defaultCoordLabel_channel
        self.CoordLabel_pipe = defaultCoordLabel_pipe
        self.locationStyle = defaultLocationStyle
        self.AVGStyle = defaultAVGStyle

        self.cart_to_polar = Params()
        self.polar_to_cart = Params()

        self.cart_to_polar.update(_cart_to_cylind_str)
        self.polar_to_cart.update(_cylind_to_cart)
        
    def format_location(self,text):
        floats = re.findall("\d+\.\d+|\d+",text)
        new_numbers = [float(x) for x in floats ]
        new_strs = [self.locationStyle(x) for x in new_numbers]
        print(text,floats,new_strs)
        for f, nf in zip(floats,new_strs):
            text.replace(f,nf)
        
        return text
class y_styler:
    def __init__(self,ydata_transform,ylims):
        self._ylims = ylims
        self._ydata_transform = ydata_transform

    def set_ystyle_lims(self,lims):
        self._ylims = lims

    def set_ydata_transform(self,func):
        self._ydata_transform = func

    def transformy(self,data):
        return self._ydata_transform(data)

    def set_ylims(self,ax):
        if self._ylims is not None:
            ax.set_ylim(self._ylims)

default_channel_ytransform = lambda ydata: ydata + 1
default_channel_ylims = [0,1]

default_pipe_ytransform = lambda ydata: -1. *(ydata - 1.)
default_pipe_ylims = None

def default_ystyle_channel(ax):
    ax.set_xlim([-1,0])
    ax.shift_xaxis(1.0)

def default_ystyle_pipe(ax):
    axis_func = lambda data: -1. *(data - 1.)
    ax.apply_func('x',axis_func)

defaultCoordLabel_channel = lambda label: r"%s/\delta"%label
defaultCoordLabel_pipe = lambda label: r"%s^*"%label

defaultAVGStyle = lambda label: r"\overline{%s}"%label
defaultLocationStyle = lambda x: r"%.2g"%x
defaultTimeStyle = r"t^*"