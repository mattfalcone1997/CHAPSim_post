
from .core import *
import numpy as np
from CHAPSim_post.utils import indexing, gradient
from pyvista import StructuredGrid

class GeomHandler():
    def __init__(self,polar):
        if polar:
            self.coord_sys = 'polar'
        else:
            self.coord_sys = 'cart'

        self.Grad_calc = gradient.Grad_calc
        
    @property
    def is_polar(self):
        return self.coord_sys == 'polar'
    
    def __str__(self):
        if self.coord_sys == 'cart':
            coord = "cartesian"
        else:
            coord = "polar (cylindrical)"
            
        return f"{self.__class__.__name__} with %s coordinate system"%coord
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def check_polar_from_icase(iCase):
            
        if iCase in [1,4,5]:
            return False
        elif iCase in [2,3]:
            return True
        else:
            msg = "CHAPSim case type invalid"
            raise ValueError(msg)
        
    
class AxisData:
    _domain_handler_class = GeomHandler
    def __init__(self, *args,from_file=False, **kwargs):
        if from_file:
            self._hdf_extract(*args,**kwargs)
        else:
            self._coordstruct_extract(*args,**kwargs)
            

        self._check_integrity()
    
    def _check_integrity(self):
        if self.coord_staggered is None:
            return

        if self.coord_staggered.index != self.coord_centered.index:
            msg = "Indices of coordstructs must be the same"
            raise ValueError(msg)

        for x in self.coord_centered.index:
            size = self.coord_centered[x].size
            if self.coord_staggered[x].size != size + 1:
                msg = ("The shape of the staggered data if given must be"
                        " one greater than centered in each dimension")
                raise ValueError(msg)

            msg = "The staggered and centered coordinates must be interleaved"
            for i in range(size):
                if self.coord_centered[x][i] < self.coord_staggered[x][i]:
                    raise ValueError(msg)
                if self.coord_centered[x][i] > self.coord_staggered[x][i+1]:
                    raise ValueError(msg)
                
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,from_file=True,**kwargs)
    
    def _hdf_extract(self,filename,key=None):
        if key is None:
            key = self.__class__.__name__
        

        hdf_obj = hdfHandler(filename,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        iCase = False if hdf_obj.attrs['cart_mode'] else True
        self._domain_handler = self._domain_handler_class(iCase)

        self.coord_centered = coordstruct.from_hdf(filename,key=key+"/coord_centered")
        if 'coord_staggered' in hdf_obj.keys():
            self.coord_staggered = coordstruct.from_hdf(filename,key=key+"/coord_staggered")
        else:
            self.coord_staggered = None
                    
    def _coordstruct_extract(self,Domain,coord,coord_nd):
        self._domain_handler = Domain
        self.coord_staggered = coord_nd
        self.coord_centered = coord
        
    def to_hdf(self,filename,mode,key=None):
        if key is None:
            key = self.__class__.__name__

        self.coord_centered.to_hdf(filename,key=key+"/coord_centered",mode=mode)
        
        if self.contains_staggered:
            self.coord_staggered.to_hdf(filename,key=key+"/coord_staggered",mode=mode)

        hdf_obj = hdfHandler(filename,mode='r',key=key)
        cart_mode = False if self._domain_handler.is_polar else True
        hdf_obj.attrs['cart_mode'] = cart_mode
    
    def create_vtkStructuredGrid(self,staggered = True):
        if staggered:
            if not self.contains_staggered:
                msg = "The staggered data cannot be None if this options is set"
                raise ValueError(msg)

            x_coords = self.coord_staggered['x']
            y_coords = self.coord_staggered['y']
            z_coords = self.coord_staggered['z']
        else:
            x_coords = self.coord_centered['x']
            y_coords = self.coord_centered['y']
            z_coords = self.coord_centered['z']

        Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)

        grid = StructuredGrid(X,Z,Y)
        return grid
    
    @property
    def contains_staggered(self):
        return not self.coord_staggered is None
    
    @property
    def staggered(self):
        return self.coord_staggered

    @property
    def centered(self):
        return self.coord_centered
    
    def copy(self):
        return copy.deepcopy(self)
    
    
    def __eq__(self,other_obj):
        if not isinstance(other_obj,self.__class__):
            msg = "This operation can only be done on other objects of this type"
            raise TypeError(msg)

        if self._domain_handler.is_polar != other_obj._domain_handler.is_polar:
            return False

        if self.coord_centered != other_obj.coord_centered:
            return False

        if self.coord_staggered != other_obj.coord_staggered:
            return False

        return True

    def __ne__(self,other_obj):
        return not self.__eq__(other_obj)
    
class coordstruct(datastruct):
    
    def set_domain_handler(self,GeomHandler):
        self._domain_handler = GeomHandler

    @property
    def DomainHandler(self):
        if hasattr(self,"_domain_handler"):
            return self._domain_handler
        else: 
            return None

    def _get_subdomain_lims(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        if xmin is None:
            xmin = np.amin(self['x'])
        if xmax is None:
            xmax = np.amax(self['x'])
        if ymin is None:
            ymin = np.amin(self['y'])
        if ymax is None:
            ymax = np.amax(self['y'])
        if zmin is None:
            zmin = np.amin(self['z'])
        if zmax is None:
            zmax = np.amax(self['z'])
            
        xmin_index, xmax_index = (self.index_calc('x',xmin)[0],
                                    self.index_calc('x',xmax)[0])
        ymin_index, ymax_index = (self.index_calc('y',ymin)[0],
                                    self.index_calc('y',ymax)[0])
        zmin_index, zmax_index = (self.index_calc('z',zmin)[0],
                                    self.index_calc('z',zmax)[0])
        return xmin_index,xmax_index,ymin_index,ymax_index,zmin_index,zmax_index

    def create_subdomain(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        (xmin_index,xmax_index,
        ymin_index,ymax_index,
        zmin_index,zmax_index) = self._get_subdomain_lims(xmin,xmax,ymin,ymax,zmin,zmax)

        xcoords = self['x'][xmin_index:xmax_index]
        ycoords = self['y'][ymin_index:ymax_index]
        zcoords = self['z'][zmin_index:zmax_index]

        return self.__class__({'x':xcoords, 'y':ycoords,'z':zcoords})

    def index_calc(self,comp,vals):
        return indexing.coord_index_calc(self,comp,vals)
    
    def check_plane(self,plane):
        if plane not in ['xy','zy','xz']:
            plane = plane[::-1]
            if plane not in ['xy','zy','xz']:
                msg = "The contour slice must be either %s"%['xy','yz','xz']
                raise KeyError(msg)
        slice_set = set(plane)
        coord_set = set(list('xyz'))
        coord = "".join(coord_set.difference(slice_set))
        return plane, coord

    def check_line(self,line):
        if line not in self.index:
            msg = f"The line must be in {self.index}"
            raise KeyError(msg)

        return line