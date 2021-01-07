from shutil import which
import os
import subprocess
import sys
import warnings


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


