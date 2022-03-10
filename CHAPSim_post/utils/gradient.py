"""
## gradient.py
Module for calculating gradient in CHAPSim_post.
Currently uses numpy.gradient as its core. It may contain a
 variety of methodologies in the future.
"""
import warnings

import numpy as np
import sympy

import CHAPSim_post as cp

from CHAPSim_post._libs import gradient, integrate

from scipy import interpolate
import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    from scipy.integrate import simpson as simps
else:
    from scipy.integrate import simps
    from scipy.integrate import cumtrapz

__all__ = ["Grad_calc",'Scalar_grad_io',"Vector_div_io",
            "Scalar_laplacian_io","Scalar_laplacian_tg",
             "totIntegrate_y",'cumIntegrate_y']

_cupy_avail = False

try:
    import cupy as cnpy
    _cupy_avail = cnpy.is_available()
except:
    pass
class Gradient():
    def __init__(self):
        self.__type = None
    
    def setup(self,coordDF):
        if self.__type is None or self.__type != cp.rcParams['gradient_method']:
            self.__type = cp.rcParams['gradient_method']
            attr_name = "setup_" + cp.rcParams['gradient_method'] + "_method"
            try:
                func = getattr(self,attr_name)
            except AttributeError:
                msg = "The stated gradient calculation method is invalid"
                raise ValueError(msg) from None
            func = getattr(self,attr_name)
            func(coordDF)

    def setup_numpy_method(self,coordDF):
        if cp.rcParams['gradient_order'] != 2:
            msg = f"For this method only gradient order 2 can be used not {cp.rcParams['gradient_order']}"
            warnings.warn(msg)
        self.grad_calc_method = self.grad_calc_numpy

    def setup_cython_method(self,coordDF):
        self.grad_calc_method = self.grad_calc_cy

    def setup_cupy_method(self,coordDF):
        self.grad_calc_method = self.grad_calc_cupy
        
    def setup_symbolic_stored_method(self,coordDF):
        msg = "This methods have not been implemented yet"
        raise NotImplementedError(msg)

    def setup_symbolic_method(self,coordDF):
        msg = "This methods have not been implemented yet"
        raise NotImplementedError(msg)

    def setup_numeric_stored_method(self,coordDF):
        msg = "This methods have not been implemented yet"
        raise NotImplementedError(msg)

    def setup_numeric_method(self,coordDF):
        msg = "This methods have not been implemented yet"
        raise NotImplementedError(msg)

    def _get_grad_dim(self,flow_array,comp):
        if flow_array.ndim == 3:
            dim = ord('z') - ord(comp)
        elif flow_array.ndim == 2:
            dim = ord('y') - ord(comp)
            if comp == 'z':
                msg = "gradients in the z direction can only be calculated on 3-d arrays"
                raise ValueError(msg)
        elif flow_array.ndim == 1:
            dim = 0
            if comp != 'y':
                msg = "For 1D flow arrays only y can be used"
                raise ValueError(msg)
        else:
            msg = "This method can only be used on one, two and three dimensional arrays"
            raise TypeError(msg)
        
        return dim
            
    def grad_calc_numpy(self,CoordDF,flow_array,comp):

        dim = self._get_grad_dim(flow_array,comp)
        coord_array = CoordDF[comp]
        
        if coord_array.size != flow_array.shape[dim]:
            msg = (f"The coordinate array size ({coord_array.size})"
                    f" and flow array size in dimension ({flow_array.shape[dim]})"
                    " does not match")
            raise ValueError(msg)
            
        return np.gradient(flow_array,coord_array,edge_order=2,axis=dim)
    
    def grad_calc_cy(self,CoordDF,flow_array,comp):
        
        dim = self._get_grad_dim(flow_array,comp)

        coord_array = CoordDF[comp]
        if coord_array.size != flow_array.shape[dim]:
            msg = (f"The coordinate array size ({coord_array.size})"
                    f" and flow array size in dimension ({flow_array.shape[dim]})"
                    " does not match")
            raise ValueError(msg)
        
        return gradient.gradient_calc(flow_array.astype('f8'),
                                      coord_array.astype('f8'),dim)
    
    def grad_calc_cupy(self,CoordDF,flow_array,comp):
        
        
        msg = "This method cannot be called if cupy is not available"
        if not _cupy_avail: raise RuntimeError(msg)
                
        dim = self._get_grad_dim(flow_array,comp)

        coord_array = CoordDF[comp]
        if coord_array.size != flow_array.shape[dim]:
            msg = (f"The coordinate array size ({coord_array.size})"
                    f" and flow array size in dimension ({flow_array.shape[dim]})"
                    " does not match")
            raise ValueError(msg)
            
        return cnpy.gradient(flow_array,coord_array,edge_order=2,axis=dim).get()
    # def _grad_calc_cy_work(self,flow_array,coord_array,dim):


    #     dx_array = np.diff(coord_array)
    #     dx = None

    #     if np.allclose(dx_array,[dx_array[0]]):
    #         dx = dx_array[0]
    #         if flow_array.ndim ==2:
    #             return gradient_parallel.cy_gradient_calc2D_dx(flow_array.astype('f8'),dx,dim)
    #         else:
    #             return gradient_parallel.cy_gradient_calc3D_dx(flow_array.astype('f8'),dx,dim)
    #     else:
    #         dx = dx_array[0]
    #         if flow_array.ndim ==2:
    #             return gradient_parallel.cy_gradient_calc2D_var_x(flow_array.astype('f8'),dx_array,dim)
    #         else:
    #             return gradient_parallel.cy_gradient_calc3D_var_x(flow_array.astype('f8'),dx_array,dim)
        
        
    def grad_calc_sparse(self,CoordDF,flow_array,comp):
        pass
    
    def Grad_calc(self,coordDF,flow_array,comp):
        self.setup(coordDF)
        return self.grad_calc_method(coordDF,flow_array,comp)

Gradient_method = Gradient()

Grad_calc = Gradient_method.Grad_calc

def _getChannelParams(coords,flow_array,staggered):
    axis = _get_integrate_axis(flow_array)
    
    middle_index = (flow_array.shape[axis]+1) // 2
    
    base_slicer1 = slice(middle_index,None)
    base_slicer2 = slice(None,middle_index)
    reverser2 = slice(None,None,-1)
    
    if staggered:
        coords1 =  coords[(middle_index):].copy()
        coords2 =  coords[:(middle_index+1)][::-1].copy()
    else:
        coords1 =  coords[middle_index:].copy()
        coords2 =  coords[:middle_index][::-1].copy()
        
    if flow_array.ndim ==1:
        flow_slicer1 = base_slicer1
        flow_slicer2 = base_slicer2
        
    elif flow_array.ndim == 2:
        flow_slicer1 = base_slicer1
        flow_slicer2 = base_slicer2
        
    elif flow_array.ndim == 3:
        flow_slicer1 = (slice(None),base_slicer1)
        flow_slicer2 = (slice(None),base_slicer2)
        reverser2 = (slice(None),reverser2)
    return  coords1, coords2,\
            flow_array[flow_slicer1].copy(),\
            flow_array[flow_slicer2][ reverser2].copy(),\
            reverser2

def _getPipeCoords(coords,flow_array):
    if flow_array.ndim ==1 or flow_array.ndim == 2:
        coord_mul = coords
        coord_mul_rev = 1./coords[::-1]
        
    elif flow_array.ndim == 3:
        coord_mul = coords[np.new_axis,:,np.new_axis]
        coord_mul_rev = coords[np.new_axis,::-1,np.new_axis]
        
    return  coord_mul, coord_mul_rev
def _get_integrate_axis(flow_array):
    if flow_array.ndim == 1 or flow_array.ndim == 2:
        return 0
    elif flow_array.ndim ==3 :
        return 1
    else:
        raise Exception

    
def totIntegrate_y(CoordDF,flow_array,channel=True):

    coords = CoordDF['y']
    axis = _get_integrate_axis(flow_array)
    
    if coords.size == flow_array.shape[axis]:
        staggered = False
    elif coords.size+1 == flow_array.shape[axis]:
        staggered = True
    else:
        msg = "Coordinate array is the wrong size"
        raise ValueError(msg)
    
            
    if channel:        
        coord_sub1, coord_sub2, flow_sub1, flow_sub2,_ = _getChannelParams(coords,
                                                                         flow_array,
                                                                         staggered)
        
        flow_inty1 = integrate.IntegrateTrapz(flow_sub1,coord_sub1,axis=axis,staggered=staggered)

        flow_inty2 = integrate.IntegrateTrapz(flow_sub2,coord_sub2,axis=axis,staggered=staggered)
        return 0.5(flow_inty1 - flow_inty2)
    else:
        return (1./coords[-1])*integrate.IntegrateTrapz(coords[-1]*flow_array,coords,axis=axis,staggered=staggered)
    
    


def cumIntegrate_y(CoordDF,flow_array,channel=True):
    
    coords = CoordDF['y']
    axis = _get_integrate_axis(flow_array)

    if coords.size != flow_array.shape[axis]+1:
        msg = ("The cumulative integration"
               " method must use the staggered data."
               f" Sizes {coords.size} {flow_array.shape[axis]}")
        raise ValueError(msg)
        
    if channel:
        coord_sub1, coord_sub2,\
            flow_sub1, flow_sub2, reverser = _getChannelParams(coords,
                                                                flow_array,
                                                                True)
        
        flow_inty1 = integrate.CumulatIntegrateTrapz(flow_sub1,coord_sub1,axis=axis)
        flow_inty2 = integrate.CumulatIntegrateTrapz(flow_sub2,coord_sub2,axis=axis)

        return np.concatenate([flow_inty2[reverser],flow_inty1],axis=axis)
    else:
        coords_mul, coords_mul_inv = _getPipeCoords(coords, flow_array)

        return coords_mul_inv*integrate.CumulatIntegrateTrapz(flow_array*coords_mul,coords,axis=axis)

# def _cum_integrator(f_array,x_array):
#     # return cumtrapz(f_array,x_array)
#     int_array = np.zeros(x_array.size-1)
#     for i in range(int_array.size):
#         int_array[i] = _array_integrator(f_array[:i+2],x_array[:i+2])
    
#     return int_array

# def _array_integrator(f_array,x_array,point_density=1):
#     assert f_array.size == x_array.size
#     kind = 'linear' if len(x_array) < 4 else 'cubic'
        
#     f_interp = interpolate.interp1d(x_array,f_array,kind=kind,fill_value='extrapolate')

#     return scipy.integrate.quadrature(f_interp,min(x_array),max(x_array),rtol=0.0001,maxiter=10)[0]

def cumIntegrate_y_1D(CoordDF,flow_array,channel=True):
    coord_array = CoordDF['y']
    
    if flow_array.ndim != 1:
        msg = "This flunction can only be used on a 1D array"
        raise ValueError(msg)

    if channel:
        middle_index = (coord_array.size+1) // 2
        coord_sub1 = coord_array[middle_index:]
        flow_sub1 = flow_array[middle_index:]
        coord_sub2 = coord_array[:middle_index]
        flow_sub2 = flow_array[:middle_index]

        

        if coord_array.size % 2 !=0:
            initial = flow_array[middle_index]
        else:
            interp = interpolate.interp1d(coord_array,flow_array,kind='cubic')
            initial = interp([0.0])[0]

            coord1 = np.insert(coord_sub1,0,0.)
            coord2 = np.insert(coord_sub2,-1,0.)
            flow1 = np.insert(flow_sub1,0,initial)
            flow2 = np.insert(flow_sub2,-1,initial)            

            # def _int_creator(coord,flow):
            #     interp = interpolate.interp1d(coord,flow,fill_value='extrapolate',kind='cubic')
            #     coord_new= _refine_input(coord,2)
            #     flow_new = interp(coord_new)

            #     return coord_new, flow_new

            # flow_sub1,coord_sub1  = _int_creator(coord1,flow1)
            # flow_sub2,coord_sub2  = _int_creator(coord2,flow2)

        flow_inty1 = cumtrapz(flow1,coord1)
        flow_inty2 = cumtrapz(flow2[::-1],coord2[::-1])[::-1]

        flow_inty = np.concatenate([flow_inty2,flow_inty1])

    else:
        interp = interpolate.interp1d(coord_array,flow_array)
        initial = interp([0.0])[0]

        coord_sub = np.insert(coord_array,0,0.)
        flow_sub = np.insert(flow_array,0,initial)

        flow_inty = (1./coord_sub)*cumtrapz(flow_sub*coord_sub,coord_sub)

    return flow_inty

def Scalar_grad_io(coordDF,flow_array):

    if flow_array.ndim == 2:
        grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
    elif flow_array.ndim == 3:
        grad_vector = np.zeros((3,flow_array.shape[0],flow_array.shape[1],
                               flow_array.shape[2]))
    else:
        msg = "This function can only be used on 2 or 3 dimensional arrays"
        raise ValueError(msg)

    grad_vector[0] = Grad_calc(coordDF,flow_array,'x')
    grad_vector[1] = Grad_calc(coordDF,flow_array,'y')
    
    
    if flow_array.ndim == 3:
        grad_vector[2] = Gradient_method.Grad_calc(coordDF,flow_array,'z')
        
    return grad_vector


def Vector_div_io(coordDF,vector_array):
    if vector_array.ndim not in (3,4):
        msg = "The number of dimension of the vector array must be 3 or 4"
        raise ValueError(msg)

    grad_vector = np.zeros_like(vector_array)
    grad_vector[0] = Grad_calc(coordDF,vector_array[0],'x')
    grad_vector[1] = Grad_calc(coordDF,vector_array[1],'y')
    
    
    if vector_array.ndim == 4:
        grad_vector[2] = Grad_calc(coordDF,vector_array[2],'z')
        
    
    div_scalar = np.sum(grad_vector,axis=0)
    
    return div_scalar

def Scalar_laplacian_io(coordDF,flow_array):
    grad_vector = Scalar_grad_io(coordDF,flow_array)
    lap_scalar = Vector_div_io(coordDF,grad_vector)
    return lap_scalar

def Scalar_laplacian_tg(coordDF,flow_array):
    dflow_dy = Grad_calc(coordDF,flow_array,'y')
    return Grad_calc(coordDF,dflow_dy,'y')







def Stencil_calc(i_list,deriv):
    assert len(i_list)-deriv > 0 and deriv > 0
    assert 0 in i_list
    
    def sympy_add(x_list):
        cum_expr = x_list[0]
        for i in range(1,len(x_list)):
            cum_expr  =  cum_expr + x_list[i]
        return cum_expr

    h_list=[]
    for i in range(len(i_list)):
        if i_list[i]==0:
            zero_index=i
        if i != len(i_list)-1:
            h_list.append(sympy.symbols("h_"+str(i_list[i])))
    taylor_coeffs=[]

    
    for i in range(len(i_list)):
        if i < zero_index:
            taylor_coeffs.append(h_list[i:zero_index])
            
            taylor_coeffs[i] = sympy_add(taylor_coeffs[i])*-1
        elif i > zero_index:
            taylor_coeffs.append(h_list[zero_index:i])
            taylor_coeffs[i] = sympy_add(taylor_coeffs[i])
        else:
            taylor_coeffs.append(0)
    matrix_coeffs=[None]*len(i_list)
    for i in range(len(i_list)):
        matrix_coeffs[i] = [(x**i)/sympy.factorial(i) for x in taylor_coeffs]
    
    Matrix = sympy.Matrix(matrix_coeffs) 
     
    RHS =[0]*len(i_list)
    RHS[deriv]=1
    RHS_vec=sympy.Matrix(RHS)
    
    sol = sympy.simplify(Matrix.solve(RHS_vec))
    
    return sol, h_list

def Stencil_coeffs_eval(sol,h_list,delta_h_list):
    #assert sol.shape[0] == len(h_list)
    sub_list=dict(zip(h_list,delta_h_list))
    return list(sol.subs(sub_list))

# def numpy_grad_calc(flow_array,coord_array,axis=None):
#     return np.gradient(flow_array,coord_array,axis=axis)


# def Grad_calc(coordDF,flowArray,comp,two_D=True):
#     if two_D:
#         assert(flowArray.ndim == 2)
#     else:
#         assert(flowArray.ndim == 3)

#     coord_array = coordDF[comp]
#     grad = np.zeros_like(flowArray)
#     sol_f, h_list_f = Stencil_calc([0,1,2], 1)
#     a_f, b_f, c_f = Stencil_coeffs_eval(sol_f,h_list_f,[coord_array[1]-coord_array[0],coord_array[2]-coord_array[1]])
    
#     sol_b, h_list_b = Stencil_calc([-2,-1,0], 1)
#     a_b, b_b, c_b = Stencil_coeffs_eval(sol_b,h_list_b,[coord_array[-2]-coord_array[-3],coord_array[-1]-coord_array[-2]])
    
#     sol_c, h_list_c = Stencil_calc([-1,0,1],1)
    
    
#     if two_D:
#         if comp =='x':
#             dim_size = flowArray.shape[1]
            
#             grad[:,0] = a_f*flowArray[:,0] + b_f*flowArray[:,1] + c_f*flowArray[:,2]#(- flowArray[:,2] + 4*flowArray[:,1] - 3*flowArray[:,0])/(coord_array[2]-coord_array[0])
#             for i in range(1,dim_size-1):
#                 #a_c, b_c, c_c = CT.Stencil_coeffs_eval(sol_c,h_list_c,[coord_array[i]-coord_array[i-1],coord_array[i+1]-coord_array[i]])
#                 #grad[:,i] = a_c*flowArray[:,i-1] + b_c*flowArray[:,i] + c_c*flowArray[:,i+1]#
#                 grad[:,i] =(flowArray[:,i+1] - flowArray[:,i-1])/(coord_array[i+1]-coord_array[i-1])
            
#             grad[:,dim_size-1] = a_b*flowArray[:,-3] + b_b*flowArray[:,-2] + c_b*flowArray[:,-1]#(3*flowArray[:,dim_size-1] - 4*flowArray[:,dim_size-2] + flowArray[:,dim_size-3])\
#                                 #/(coord_array[dim_size-1]-coord_array[dim_size-3])
    
#         elif comp =='y':
#             dim_size = flowArray.shape[0]
#             grad[0,:] = a_f*flowArray[0,:] + b_f*flowArray[1,:] + c_f*flowArray[2,:]#(-flowArray[2,:] + 4*flowArray[1,:] - 3*flowArray[0,:])/(coord_array[2]-coord_array[0])
#             for i in range(1,dim_size-1):
#                 # a_c, b_c, c_c = CT.Stencil_coeffs_eval(sol_c,h_list_c,[coord_array[i]-coord_array[i-1],coord_array[i+1]-coord_array[i]])
#                 # grad[i] = a_c*flowArray[i-1] + b_c*flowArray[i] + c_c*flowArray[i+1]
#                 h1 = coord_array[i+1]-coord_array[i]
#                 h0 =  coord_array[i]-coord_array[i-1]
#                 grad[i] = -h1/(h0*(h0+h1))*flowArray[i-1] + (h1-h0)/(h0*h1)*flowArray[i] + h0/(h1*(h0+h1))*flowArray[i+1]
#                 #grad[i,:] = (flowArray[i+1,:] - flowArray[i-1,:])/(coord_array[i+1]-coord_array[i-1])
#             grad[-1,:] = a_b*flowArray[-3,:] + b_b*flowArray[-2,:] + c_b*flowArray[-1,:]#(3*flowArray[dim_size-1,:] - 4*flowArray[dim_size-2,:] + flowArray[-3,:])\
#                                 #/(coord_array[dim_size-1]-coord_array[dim_size-3])
#         else:    
#             raise Exception
#     else:
#         if comp =='x':
#             dim_size = flowArray.shape[2]
#             grad[:,:,0] = a_f*flowArray[:,:,0] + b_f*flowArray[:,:,1] + c_f*flowArray[:,:,2]# (flowArray[:,:,1] - flowArray[:,:,0])/(coord_array[1]-coord_array[0])
#             for i in range(1,dim_size-1):
#                 grad[:,:,i] = (flowArray[:,:,i+1] - flowArray[:,:,i-1])/(coord_array[i+1]-coord_array[i-1])
#             grad[:,:,-1] = a_b*flowArray[:,:,-3] + b_b*flowArray[:,:,-2] + c_b*flowArray[:,:,-1] #(flowArray[:,:,dim_size-1] - flowArray[:,:,dim_size-2])/(coord_array[dim_size-1]-coord_array[dim_size-2])
    
#         elif comp =='y':
#             dim_size = flowArray.shape[1]
#             grad[:,0,:] = a_f*flowArray[:,0,:] + b_f*flowArray[:,1,:] + c_f*flowArray[:,2,:] #(flowArray[:,1,:] - flowArray[:,0,:])/(coord_array[1]-coord_array[0])
#             for i in range(1,dim_size-1):
#                 h1 = coord_array[i+1]-coord_array[i]
#                 h0 =  coord_array[i]-coord_array[i-1]
#                 grad[:,i] = -h1/(h0*(h0+h1))*flowArray[:,i-1] + (h1-h0)/(h0*h1)*flowArray[:,i] + h0/(h1*(h0+h1))*flowArray[:,i+1]
#                 # a_c, b_c, c_c = CT.Stencil_coeffs_eval(sol_c,h_list_c,[coord_array[i]-coord_array[i-1],coord_array[i+1]-coord_array[i]])
#                 # grad[:,i] = a_c*flowArray[:,i-1] + b_c*flowArray[:,i] + c_c*flowArray[:,i+1]
#                 #grad[:,i,:] = (flowArray[:,i+1,:] - flowArray[:,i-1,:])/(coord_array[i+1]-coord_array[i-1])
#             grad[:,-1,:] = a_b*flowArray[:,-3,:] + b_b*flowArray[:,-2,:] + c_b*flowArray[:,-1,:]#(flowArray[:,dim_size-1,:] - flowArray[:,dim_size-2,:])/(coord_array[dim_size-1]-coord_array[dim_size-2])
#         elif comp=='z':
#             dim_size = flowArray.shape[0]
#             grad[0,:,:] = a_f*flowArray[0,:,:] + b_f*flowArray[1,:,:] + c_f*flowArray[2,:,:]#(flowArray[1,:,:] - flowArray[0,:,:])/(coord_array[1]-coord_array[0])
#             for i in range(1,dim_size-1):
#                 grad[i,:,:] = (flowArray[i+1,:,:] - flowArray[i-1,:,:])/(coord_array[i+1]-coord_array[i-1])
#             grad[-1,:,:] = a_b*flowArray[-3,:,:] + b_b*flowArray[-2,:,:] + c_b*flowArray[-1,:,:]#(flowArray[dim_size-1,:,:] - flowArray[dim_size-2,:,:])/(coord_array[dim_size-1]-coord_array[dim_size-2])

#         else:    
#             raise Exception
#     return grad

# def Grad_calc_tg(CoordDF,flowArray):
#     coord_array = CoordDF['y']
#     grad = np.zeros_like(flowArray)
#     dim_size = flowArray.shape[0]
#     dim_size = flowArray.shape[0]

#     sol_f, h_list_f = Stencil_calc([0,1,2], 1)
#     a_f, b_f, c_f = Stencil_coeffs_eval(sol_f,h_list_f,[coord_array[1]-coord_array[0],coord_array[2]-coord_array[1]])
    
#     sol_b, h_list_b = Stencil_calc([-2,-1,0], 1)
#     a_b, b_b, c_b = Stencil_coeffs_eval(sol_b,h_list_b,[coord_array[-2]-coord_array[-3],coord_array[-1]-coord_array[-2]])

#     grad[0] = a_f*flowArray[0] + b_f*flowArray[1] + c_f*flowArray[2]
#     grad[-1] = a_b*flowArray[-3] + b_b*flowArray[-2] + c_b*flowArray[-1]
#     for i in range(1,dim_size-1):
#         h1 = coord_array[i+1]-coord_array[i]
#         h0 =  coord_array[i]-coord_array[i-1]
#         grad[i] = -h1/(h0*(h0+h1))*flowArray[i-1] + (h1-h0)/(h0*h1)*flowArray[i] + h0/(h1*(h0+h1))*flowArray[i+1]

#     return grad

# def scalar_grad(coordDF,flow_array,two_D=True):
#     if two_D:
#         assert(flow_array.ndim == 2)
#         grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
#     else:
#         assert(flow_array.ndim == 3)
#         grad_vector = np.zeros((3,flow_array.shape[0],flow_array.shape[1],\
#                                flow_array.shape[2]))
    
#     grad_vector[0] = Grad_calc(coordDF,flow_array,'x')
#     grad_vector[1] = Grad_calc(coordDF,flow_array,'y')
    
    
#     if not two_D:
#         grad_vector[2] = Grad_calc(coordDF,flow_array,'z')
        
#     return grad_vector

# def Vector_div(coordDF,vector_array,two_D=True):
#     if two_D:
#         assert(vector_array.ndim == 3)
#     else:
#         assert(vector_array.ndim == 4)
    
#     grad_vector = np.zeros_like(vector_array)
#     grad_vector[0] = Grad_calc(coordDF,vector_array[0],'x')
#     grad_vector[1] = Grad_calc(coordDF,vector_array[1],'y')
    
    
#     if not two_D:
#         grad_vector[2] = Grad_calc(coordDF,vector_array[2],'z')
    
#     if two_D:    
#         div_scalar = np.zeros((vector_array.shape[1],vector_array.shape[2]))
#     else:
#         div_scalar = np.zeros((vector_array.shape[1],vector_array.shape[2],\
#                                vector_array.shape[3]))
    
#     div_scalar = np.sum(grad_vector,axis=0)
    
#     return div_scalar

# def Scalar_laplacian_io(coordDF,flow_array,two_D=True):
#     grad_vector = scalar_grad(coordDF,flow_array,two_D)
#     lap_scalar = Vector_div(coordDF,grad_vector,two_D)
#     return lap_scalar

# def Scalar_laplacian_tg(coordDF,flow_array):
#     return Grad_calc_tg(coordDF,Grad_calc_tg(coordDF,flow_array))