
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import CHAPSim_post as cp

import itertools
import warnings

class CHAPSimFigure(mpl.figure.Figure):

    def clegend(self,*args, **kwargs):
        return super().legend(*args, **kwargs)
    def add_subplot(self,*args, **kwargs):
        kwargs['projection']='AxesCHAPSim'
        return super().add_subplot(*args,**kwargs)
    def c_add_subplot(self,*args, **kwargs):
        kwargs['projection']='AxesCHAPSim'
        return super().add_subplot(*args,**kwargs)
    def get_legend(self):
        if len(self.legends) == 1:
            return self.legends[0]
        else:
            return self.legends
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