import subprocess
import os
import shutil

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
            os.removedirs(path) 



class TexEngine:
    def __init__(self):
        self._texExe = "pdflatex"

    def generate_preamble(self):
        preamble = r"\usepackage{pgfplots}"
        return preamble

    def generate_file(self,func_write):
        path = os.path.join(os.getcwd(),_working_dir)
        if not os.path.isdir(path):
            os.mkdir(path)

        tex_file = open(os.path.join(path,_tex_file),'w')

        tex_file.write(r"\documentclass{standalone}")
        tex_file.write(self.generate_preamble())
        tex_file.write(r"\begin{document}")

        func_write(tex_file)
        
        tex_file.write(r"\end{document}")

    def execute(self):
        pass



class pgfFigure:
    def __init__(self):
        
        self._renderer = TexEngine()
        self._artists = []

    def add_axis(self,*args,**kwargs):
        self._artists.append(axesLayout(*args,**kwargs))

    def add_layout(self,*args,**kwargs):
        self._artists.append(axesLayout(*args,**kwargs))

    def write_to_Tex(self,file):
        file.write(r"\begin{center}")
        for artist in self._artists:
            artist.write_to_Tex(file)
        file.write(r"\end{center}")
    def savefig(self,file_name):
        self._renderer.generate_file(self.write_to_Tex)
    
    def __del__(self):
        _Decrement_no()



class axesLayout:
    def __init__(self,nrow,ncols):
        self._artists = []
        col_layout = []

    
    def write_to_Tex(self,file):

        file.write(r"\begin{tabular}")
        for artist in self._artists:
            artist.write_to_Tex(file)

class Axis3D:
    def __init__(self,*kwargs):
        self._artists = []
    
    def write_to_Tex(self,file):
        for artist in self._artists:
            artist.write_to_Tex(file)