import numpy as np
import CHAPSim_parallel as cpar
import CHAPSim_post as cp
import warnings
from scipy.integrate import solve_bvp
import sympy
import itertools


def flip_leg_col(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

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

def Blasius_calc(no_points,x_array, y_array,U_array,REN):
    if isinstance(x_array, np.ndarray) or isinstance(y_array, np.ndarray) :
        assert x_array.shape == y_array.shape, 'x_array and y_array must be the same size'
        if type(U_array) ==np.ndarray:
            assert U_array.size == x_array.shape[0], 'u_array must be the same size as axis 1 of x_array'
        elif type(U_array) == float:
            assert x_array.ndim==1 
        else:
            raise TypeError('U_array must be type nd.array or float')
            
    RHS_fun = lambda x,y: np.array([y[1],y[2],-0.5*y[2]*y[0]])

    bc = lambda ya,yb: np.array([ya[0],ya[1],1-yb[1]])

    eta_eval = np.zeros_like((x_array))
    nu_star=1.0
    if isinstance(U_array,np.ndarray):
        for i in range(U_array.size):
            eta_eval[i] = y_array[i]*np.sqrt(U_array[i]*REN/(2*nu_star*x_array[i]))
    elif isinstance(U_array,float):
            eta_eval = y_array*np.sqrt(U_array*REN/(2*nu_star*x_array))
    print(eta_eval)
    eta = np.linspace(0, np.max(eta_eval),no_points)
    f1_init = eta
    f2_init = np.exp(-eta)
    f3_init = np.exp(-eta)
    finit = np.column_stack((f1_init,f2_init,f3_init))
    sol = solve_bvp(RHS_fun,bc,eta,finit.T)
    f = sol.sol(eta_eval)
    if isinstance(U_array,np.ndarray):
        u_array = np.zeros_like(x_array)
        for i in range(U_array.size):
            u_array[i] = f[1][i]*U_array[i]
    elif isinstance(U_array,float):
        u_array = f[1]*U_array
    
    return u_array
    
def Gen_Grad_calc(x_vals, array):
    assert(x_vals.size==array.size)
    d_array_d_x_vals = np.zeros_like(array)
    
    d_array_d_x_vals[0] = (array[1]-array[0])/(x_vals[1]-x_vals[0])#(-array[2]+4*array[1]-3*array[0])/(x_vals[2]-x_vals[0])
    for i in range(1,array.size-1):
        d_array_d_x_vals[i] = (array[i+1]-array[i-1])/(x_vals[i+1]-x_vals[i-1])
    d_array_d_x_vals[array.size-1] = (array[array.size-1]-array[array.size-2])/(x_vals[array.size-1]-x_vals[array.size-2])#(3*array[array.size-1] -4*array[array.size-2] +array[array.size-3])/\
                                    #(x_vals[array.size-1]-x_vals[array.size-3])
    return d_array_d_x_vals

def Y_plus_index_calc(AVG_DF,CoordDF,coord_list,x_vals=''):
    if x_vals:
        index_list = coord_index_calc(CoordDF,'x',x_vals)
        
    y_coords = CoordDF['y'].dropna().values
    if isinstance(coord_list,float) or isinstance(coord_list,int):
        coord_list = [coord_list]
    elif not isinstance(coord_list,list):
        raise TypeError("coord_list must be of type float, list or int")
    avg_time = AVG_DF.flow_AVGDF.index[0][0]
    if hasattr(AVG_DF,"par_dir"):
        par_dir = AVG_DF.par_dir
        u_tau_star, delta_v_star = cpar.wall_unit_calc(AVG_DF,avg_time,par_dir)
    else:
        u_tau_star, delta_v_star = cp.wall_unit_calc(AVG_DF,avg_time)
    
    if x_vals:
        Y_plus_list=[]
        if isinstance(index_list,list):
            for index in index_list:
                Y_plus_list.append((1-abs(y_coords[:int(y_coords.size/2)]))/delta_v_star[index])
        else:
            Y_plus_list.append((1-abs(y_coords[:int(y_coords.size/2)]))/delta_v_star[index_list])

        Y_plus_index_list = []
        for i in range(len(coord_list)):
            try:
                for j in range(Y_plus_list[0].size):
                    if Y_plus_list[i][j+1]>coord_list[i]:
                        if abs(Y_plus_list[i][j+1]-coord_list[i]) > abs(Y_plus_list[i][j]-coord_list[i]):
                            
                            Y_plus_index_list.append(j)
                            break
                        else:
                            Y_plus_index_list.append(j+1)
                            break
            except IndexError:
                warnings.warn("Value in coord_list out of bounds: "\
                                 + "Y_plus given: %g, max Y_plus: %g" % (coord_list[i],max(Y_plus_list[i])))
                    
    else:
        Y_plus = (1-abs(y_coords[:int(y_coords.size/2)]))/delta_v_star[0]
        
        Y_plus_index_list = []
        for coord in coord_list:
            try:
                for i in range(Y_plus.size):
                    if Y_plus[i+1]>coord:
                        if abs(Y_plus[i+1]-coord)>abs(Y_plus[i]-coord):
                            Y_plus_index_list.append(i)
                            break
                        else:
                            Y_plus_index_list.append(i+1)
                            break 
            except IndexError:
                warnings.warn("Value in coord_list out of bounds: "\
                                 + "Y_plus given: %g, max Y_plus: %g. Ignoring values beyond this" % (coord,max(Y_plus)))
                return Y_plus_index_list
    if len(coord_list)==1:
        return Y_plus_index_list[0]
    else:
        return Y_plus_index_list

def coord_index_calc(CoordDF,comp,coord_list):
    coords = CoordDF[comp].dropna().values
    if isinstance(coord_list,float) or isinstance(coord_list,int):
        coord_list = [coord_list]
    index_list=[]
    for coord in coord_list:
        try:
            for i in range(coords.size):
                if coords[i+1]>coord:
                    if abs(coords[i+1]-coord)>abs(coords[i]-coord):
                        index_list.append(i)
                        break
                    else:
                        index_list.append(i+1)
                        break 
        except IndexError:
            warnings.warn("Value in coord_list out of bounds: "\
                             + "%s coordinate given: %g, max %s coordinate:" % (comp,coord,comp)\
                             + " %g. Ignoring values beyond this" % max(coords))
            return index_list
    if len(coord_list)==1:
        return index_list[0]
    else:
        return index_list

def Grad_calc(coordDF,flowArray,comp,two_D=True):
    if two_D:
        assert(flowArray.ndim == 2)
    else:
        assert(flowArray.ndim == 3)
    coord_array = coordDF[comp].dropna().values
    grad = np.zeros_like(flowArray)
    if two_D:
        if comp =='x':
            dim_size = flowArray.shape[1]
            grad[:,0] = (- flowArray[:,2] + 4*flowArray[:,1] - 3*flowArray[:,0])/(coord_array[2]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[:,i] = (flowArray[:,i+1] - flowArray[:,i-1])/(coord_array[i+1]-coord_array[i-1])
            grad[:,dim_size-1] = (3*flowArray[:,dim_size-1] - 4*flowArray[:,dim_size-2] + flowArray[:,dim_size-3])\
                                /(coord_array[dim_size-1]-coord_array[dim_size-3])
    
        elif comp =='y':
            dim_size = flowArray.shape[0]
            grad[0,:] = (-flowArray[2,:] + 4*flowArray[1,:] - 3*flowArray[0,:])/(coord_array[2]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[i,:] = (flowArray[i+1,:] - flowArray[i-1,:])/(coord_array[i+1]-coord_array[i-1])
            grad[dim_size-1,:] = (3*flowArray[dim_size-1,:] - 4*flowArray[dim_size-2,:] + flowArray[dim_size-3,:])\
                                /(coord_array[dim_size-1]-coord_array[dim_size-3])
        else:    
            raise Exception
    else:
        if comp =='x':
            dim_size = flowArray.shape[2]
            grad[:,:,0] = (flowArray[:,:,1] - flowArray[:,:,0])/(coord_array[1]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[:,:,i] = (flowArray[:,:,i+1] - flowArray[:,:,i-1])/(coord_array[i+1]-coord_array[i-1])
            grad[:,:,dim_size-1] = (flowArray[:,:,dim_size-1] - flowArray[:,:,dim_size-2])/(coord_array[dim_size-1]-coord_array[dim_size-2])
    
        elif comp =='y':
            dim_size = flowArray.shape[1]
            grad[:,0,:] = (flowArray[:,1,:] - flowArray[:,0,:])/(coord_array[1]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[:,i,:] = (flowArray[:,i+1,:] - flowArray[:,i-1,:])/(coord_array[i+1]-coord_array[i-1])
            grad[:,dim_size-1,:] = (flowArray[:,dim_size-1,:] - flowArray[:,dim_size-2,:])/(coord_array[dim_size-1]-coord_array[dim_size-2])
        elif comp=='z':
            dim_size = flowArray.shape[0]
            grad[0,:,:] = (flowArray[1,:,:] - flowArray[0,:,:])/(coord_array[1]-coord_array[0])
            for i in range(1,dim_size-1):
                grad[i,:,:] = (flowArray[i+1,:,:] - flowArray[i-1,:,:])/(coord_array[i+1]-coord_array[i-1])
            grad[dim_size-1,:,:] = (flowArray[dim_size-1,:,:] - flowArray[dim_size-2,:,:])/(coord_array[dim_size-1]-coord_array[dim_size-2])

        else:    
            raise Exception
    return grad
def scalar_grad(coordDF,flow_array,two_D=True):
    if two_D:
        assert(flow_array.ndim == 2)
        grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
    else:
        assert(flow_array.ndim == 3)
        grad_vector = np.zeros(3,flow_array.shape[0],flow_array.shape[1],\
                               flow_array.shape[2])
    
    grad_vector[0] = Grad_calc(coordDF,flow_array,'x')
    grad_vector[1] = Grad_calc(coordDF,flow_array,'y')
    
    
    if not two_D:
        grad_vector[2] = Grad_calc(coordDF,flow_array,'z')
        
    return grad_vector
def Vector_div(coordDF,vector_array,two_D=True):
    if two_D:
        assert(vector_array.ndim == 3)
    else:
        assert(vector_array.ndim == 4)
    
    grad_vector = np.zeros_like(vector_array)
    grad_vector[0] = Grad_calc(coordDF,vector_array[0],'x')
    grad_vector[1] = Grad_calc(coordDF,vector_array[1],'y')
    
    
    if not two_D:
        grad_vector[2] = Grad_calc(coordDF,vector_array[2],'z')
    
    if two_D:    
        div_scalar = np.zeros((vector_array.shape[1],vector_array.shape[2]))
    else:
        div_scalar = np.zeros((vector_array.shape[1],vector_array.shape[2],\
                               vector_array.shape[3]))
    
    div_scalar = np.sum(grad_vector,axis=0)
    
    return div_scalar
def scalar_laplacian(coordDF,flow_array,two_D=True):
    grad_vector = scalar_grad(coordDF,flow_array,two_D)
    lap_scalar = Vector_div(coordDF,grad_vector,two_D)
    return lap_scalar