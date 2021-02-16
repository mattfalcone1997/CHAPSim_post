import subprocess
import os
import shutil

import matplotlib as mpl

_working_dir = ".pgfplot_wspace" 
_tex_file = "tex_file.tex"

Engine_no = 0

def _Increment_no():
    global Engine_no
    Engine_no += 1

def _Decrement_no():
    global Engine_no
    Engine_no -= 1
    if Engine_no == 0:
        path = os.path.join(os.getcwd(),_working_dir)
        if os.path.isdir(path):
            try:
                os.rmdir(path) 
            except OSError:
                pass



class TexEngine:
    def __init__(self):
        self._texExe = "pdflatex"
        self._preamble = [r"\usepackage{pgfplots}",
                            ]
    def generate_preamble(self):
        return preamble

    def generate_file(self,func_write):
        path = os.path.join(os.getcwd(),_working_dir)
        if not os.path.isdir(path):
            os.mkdir(path)

        tex_file = open(os.path.join(path,_tex_file),'w')

        resolution =mpl.rcParams['figure.dpi']
        
        tex_file.write(("\documentclass[tikz,convert={outext=.png,"
                            "density=%f}]{standalone}\n"%resolution))
        tex_file.write("\n".join(self._preamble))
        tex_file.write("\\begin{document}\n")

        func_write(tex_file)
        
        tex_file.write("\end{document}\n")

    def execute(self):
        pass



class pgfFigure:
    def __init__(self):
        
        self._renderer = TexEngine()
        self._artists = []

        _Increment_no()
    def add_axis(self,*args,**kwargs):
        self._artists.append(axesLayout(*args,**kwargs))

    def add_layout(self,*args,**kwargs):
        self._artists.append(axesLayout(*args,**kwargs))

    def write_to_Tex(self,file):
        file.write("\\begin{center}\n")
        for artist in self._artists:
            artist.write_to_Tex(file)
        file.write("\\end{center}\n")

    def savefig(self,file_name):
        self._renderer.generate_file(self.write_to_Tex)
    
    def __del__(self):
        _Decrement_no()



class axesLayout:
    def __init__(self,nrow,ncol,**kwargs):
        self._artists = []
        self._layout = (nrow,ncol)

        for i in range(nrow*ncol):
            self.add_axes(**kwargs)

    def _set_col_layout(self,ncols):

        if ncols == 1:
            layout = "{c}"
        elif ncols == 2:
            layout = "{lr}"
        else:
            c_list = "".join(["c" for _ in range(ncols-2)])
            layout = "{l" + c_list + "r}"
        return layout

    def add_axes(self,**kwargs):
        self._artists.append(Axis3D(**kwargs))
    def _get_trim_axes(self,layout,i):
        opt_args= ["baseline"]
        trim_str = "trim axis "

        if layout[1:-1][i] == 'c':
            opt_args.append(trim_str + "left")
            opt_args.append(trim_str + "right")
        elif layout[1:-1][i] == 'l':
            opt_args.append(trim_str + "left")
        elif layout[1:-1][i] == 'r':
            opt_args.append(trim_str + "right")
        else:
            raise Exception(layout[i])

        return opt_args
            


    def write_to_Tex(self,file):
        layout = self._set_col_layout(self._layout[1])
        file.write("\\begin{tabular}%s\n"%layout)

        for i, artist in enumerate(self._artists):
            opt_list = self._get_trim_axes(layout,i%self._layout[1])
            artist.write_to_Tex(file,*opt_list)
            if layout[1:-1][i%self._layout[1]] == 'r':
                file.write("\\\\ \n")
            else:
                file.write("&\n")

        file.write("\end{tabular}\n")

    def __getitem__(self,rowkey,colkey):
        return self._artists[rowkey*self._layout[1]+colkey]

class Axis3D:
    def __init__(self,**kwargs):
        self._params = kwargs
        self._artists = []

    def write_params(self,*param_list,**params_dict):
        param_list = list(param_list)
        for key,val in params_dict.items():
            param_list.append("%s=%s"%(key,val))

        return "[" +",".join(param_list) +"]" if param_list else ""

    def write_to_Tex(self,file,*args,**kwargs):

        file.write("\\begin{tikzpicture}%s\n"%self.write_params(*args,**kwargs))
        file.write("\\begin{axis}%s\n"%self.write_params(**self._params))

        for artist in self._artists:
            artist.write_to_Tex(file)

        file.write("\\end{axis}\n")
        file.write("\\end{tikzpicture}\n")
def pgfSubplots(nrow,ncol,**kwargs):

    fig = pgfFigure()

    ax = fig.add_layout(nrow,ncol,**kwargs)

    return fig, ax