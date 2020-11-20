"""
# CHAPSim_Tools
Module of auxilliary functions to support but the internals and users 
of CHAPSim_post. 
"""

import numpy as np
import CHAPSim_parallel as cpar
import CHAPSim_post as cp
import CHAPSim_post_v2 as cp2

import warnings
import matplotlib as mpl
from scipy.integrate import solve_bvp
import sympy
import itertools
import os
import paramiko
from scp import SCPClient
import termios
import sys
import tracemalloc
import linecache


from functools import wraps
from multiprocessing import Process, Queue


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper

class debug_memory:
    def __init__(self):
        tracemalloc.start()

    def __del__(self):
        tracemalloc.stop()

    @staticmethod
    def take_snapshot():
        return tracemalloc.take_snapshot()

    @staticmethod
    def display_top(snapshot, key_type='lineno', limit=3):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        print("Top %s lines" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#%s: %s:%s: %.1f KiB"
                % (index, filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))

def from_HPC2Local(remote,remote_path,file,**kwargs):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    uname, host = remote.split('@')
    try:
        os.system("stty -echo")
        ssh.connect(host,username=uname)
        os.system("stty echo")
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except paramiko.PasswordRequiredException as e:
        import getpass
        
        password = getpass.getpass("%s, enter password/phrase: "%e)
        try:
            ssh.connect(host,username=uname,password=password)
        except paramiko.PasswordRequiredException as e:
            passphrase = getpass.getpass("%s, enter password/phrase: "%e)
            ssh.connect(host,username=uname,password=password,passphrase=passphrase)

    scp = SCPClient(ssh.get_transport(),**kwargs)

    file_path = os.path.join(remote_path, file)
    scp.get(file_path)
    scp.close()

def file_extract(path_to_folder,abs_path=True):
    if abs_path:
        mypath = os.path.join(path_to_folder,'1_instant_D')
    else:
        mypath = os.path.abspath(os.path.join(path_to_folder,'1_instant_D'))
    file_names = [f for f in os.listdir(mypath) if f[:8]=='DNS_peri']
    return file_names       

def time_extract(path_to_folder,abs_path=True):
    file_names = file_extract(path_to_folder,abs_path)
    time_list =[]
    for file in file_names:
        time_list.append(float(file[20:35]))
    times = list(dict.fromkeys(time_list))
    return list(set(times))

def max_time_calc(path_to_folder,abs_path):
    if isinstance(path_to_folder,list):
        max_time = np.float('inf')
        for path in path_to_folder:
            max_time=min(max_time,max_time_calc(path,abs_path))
    else:
        max_time=max(cp2.time_extract(path_to_folder,abs_path))
       
    return max_time

def moving_wall_similarity(Re_0,U_grad):
    def moving_wall_calc(y_vals,Eh):
        R=np.abs(Eh)*Re_0
        RHS_fun = lambda x,y: np.array([y[1],y[2],y[3],-R*(y[0]*y[3]-y[1]*y[2])])
        bc = lambda ya,yb: np.array([yb[0],yb[2],ya[0],1-ya[1]])
        # y_vals=np.linspace(0,1,100)
        f1_init=y_vals
        f2_init=np.exp(y_vals)
        f3_init=np.exp(y_vals)
        f4_init=np.exp(y_vals)
        finit = np.column_stack((f1_init,f2_init,f3_init,f4_init))
        
        
        sol = solve_bvp(RHS_fun,bc,y_vals,finit.T)
        f = sol.sol(y_vals)
        return f
    y_coords=np.linspace(-1,0,1000)
    f = moving_wall_calc(y_coords,U_grad)
    U=f[1]
    print(U[0],U[-1])
    return y_coords,(1-f[1])/(f[1,0]-f[1,-1])

def default_axlabel_kwargs():
    return {'fontsize':20}
def default_legend_kwargs():
    return {'loc':'upper center','bbox_to_anchor':(0.5,0.0)}

def filter_mpl_kwargs(mpl_class,kwargs):
    if not kwargs: return kwargs
    text_keys = mpl_class.properties().keys()
    output_dict={key: kwargs[key] for key in text_keys if key in kwargs.keys()}
    return output_dict

def filter_mpl_kwargs_text(kwargs):
    return filter_mpl_kwargs(mpl.text.Text(),kwargs)
def filter_mpl_kwargs_legend(kwargs):
    new_kwargs=filter_mpl_kwargs(mpl.legend(),kwargs)
    new_kwargs.pop('fontsize')
    return new_kwargs
def update_kwargs(orig_kwargs,new_kwargs):
    if not new_kwargs: return None
    for key,item in new_kwargs.items():
        orig_kwargs[key] = item
        
        
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
        
    y_coords = CoordDF['y']
    if isinstance(coord_list,float) or isinstance(coord_list,int):
        coord_list = [coord_list]
    elif not isinstance(coord_list,list):
        raise TypeError("\033[1;32 coord_list must be of type float, list or int")
    avg_time = AVG_DF.flow_AVGDF.index[0][0]
    if hasattr(AVG_DF,"par_dir"):
        par_dir = AVG_DF.par_dir
        u_tau_star, delta_v_star = cpar.wall_unit_calc(AVG_DF,avg_time,par_dir)
    else:
        u_tau_star, delta_v_star = cpAVG_DF.wall_unit_calc(avg_time)
    
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
                warnings.warn("\033[1;33Value in coord_list out of bounds: "\
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
                warnings.warn("\033[1;33Value in coord_list out of bounds: "\
                                 + "Y_plus given: %g, max Y_plus: %g. Ignoring values beyond this" % (coord,max(Y_plus)))
                return Y_plus_index_list
    if len(coord_list)==1:
        return Y_plus_index_list[0]
    else:
        return Y_plus_index_list

def y_coord_index_norm(AVG_DF,CoordDF,coord_list,x_vals='',mode='half_channel'):
    if mode=='half_channel':
        norm_distance=np.ones((AVG_DF.NCL[0]))
    elif mode == 'disp_thickness':
        norm_distance, *other_thickness = AVG_DF._int_thickness_calc(AVG_DF.flow_AVGDF.index[0][0])
    elif mode == 'mom_thickness':
        disp_thickness, norm_distance, shape_factor = AVG_DF._int_thickness_calc(AVG_DF.flow_AVGDF.index[0][0])
    elif mode == 'wall':
        u_tau_star, norm_distance = AVG_DF._wall_unit_calc(AVG_DF.flow_AVGDF.index[0][0])
    else:
        raise ValueError("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
                                " or 'wall. Value used was %s\n"%mode)
    #print(norm_distance)
    y_coords=CoordDF['y']
    if x_vals:
        # print(x_vals)
        x_index =[AVG_DF._return_index(x) for x in x_vals]
        if not hasattr(x_index,'__iter__'):
            x_index=[x_index]
    elif x_vals is None:
        x_index=list(range(AVG_DF.shape[-1]))
    else:
        x_index=[0]
    if not hasattr(coord_list,'__iter__'):
        coord_list=[coord_list]#raise TypeError("coord_list mist be an integer or iterable")
    #y_coords_thick=np.zeros((int(0.5*y_coords.size),len(x_index)))
    y_thick_index=[]
    for x in x_index:
        y_coords_thick=(1-abs(y_coords[:int(0.5*y_coords.size)]))/norm_distance[x]
        y_thick=[]
        for coord in coord_list:
            try:
                for i in range(y_coords_thick.size):
                    if y_coords_thick[i+1]> coord:
                        if abs(y_coords_thick[i+1]-coord)>abs(y_coords_thick[i]-coord):
                            y_thick.append(i)
                            break
                        else:
                            y_thick.append(i+1)
                            break 
            except IndexError:
                warnings.warn("\033[1;33Value in coord_list out of bounds: "\
                                 + "Y_plus given: %g, max Y_plus: %g. Ignoring values beyond this" % (coord,max(y_coords_thick)))
                break
        y_thick_index.append(y_thick)
    # print(y_thick_index)
    # if len(coord_list)==1:
    #     y_thick_index= list(itertools.chain(*y_thick_index))
    return y_thick_index
def coord_index_calc(CoordDF,comp,coord_list):
    coords = CoordDF[comp]
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
            
            coord_end_plus=2*coords[coords.size-1]-coords[coords.size-2]
            
            if coord_end_plus>coord:
                index_list.append(i)
            else:
                warnings.warn("\033[1;33Value in coord_list out of bounds: "\
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
    coord_array = coordDF[comp]
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

def Grad_calc_tg(CoordDF,flowArray):
    coord_array = CoordDF['y']
    grad = np.zeros_like(flowArray)
    dim_size = flowArray.shape[0]
    dim_size = flowArray.shape[0]

    sol_f, h_list_f = Stencil_calc([0,1,2], 1)
    a_f, b_f, c_f = Stencil_coeffs_eval(sol_f,h_list_f,[coord_array[1]-coord_array[0],coord_array[2]-coord_array[1]])
    
    sol_b, h_list_b = Stencil_calc([-2,-1,0], 1)
    a_b, b_b, c_b = Stencil_coeffs_eval(sol_b,h_list_b,[coord_array[-2]-coord_array[-3],coord_array[-1]-coord_array[-2]])

    grad[0] = a_f*flowArray[0] + b_f*flowArray[1] + c_f*flowArray[2]
    grad[-1] = a_b*flowArray[-3] + b_b*flowArray[-2] + c_b*flowArray[-1]
    for i in range(1,dim_size-1):
        h1 = coord_array[i+1]-coord_array[i]
        h0 =  coord_array[i]-coord_array[i-1]
        grad[i] = -h1/(h0*(h0+h1))*flowArray[i-1] + (h1-h0)/(h0*h1)*flowArray[i] + h0/(h1*(h0+h1))*flowArray[i+1]

    return grad

def scalar_grad(coordDF,flow_array,two_D=True):
    if two_D:
        assert(flow_array.ndim == 2)
        grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
    else:
        assert(flow_array.ndim == 3)
        grad_vector = np.zeros((3,flow_array.shape[0],flow_array.shape[1],
                               flow_array.shape[2]))
    
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

def Scalar_laplacian_tg(coordDF,flow_array):
    return Grad_calc_tg(coordDF,Grad_calc_tg(coordDF,flow_array))

def contour_plane(plane,axis_vals,avg_data,y_mode,PhyTime):
    if plane not in ['xy','zy','xz']:
        plane = plane[::-1]
        if plane not in ['xy','zy','xz']:
            msg = "The contour slice must be either %s"%['xy','yz','xz']
            raise KeyError(msg)
    slice_set = set(plane)
    coord_set = set(list('xyz'))
    coord = "".join(coord_set.difference(slice_set))

    

    if coord == 'y':
        tg_post = True if all([x == 'None' for x in avg_data.flow_AVGDF.times]) else False
        if not tg_post:
            norm_val = 0
        elif tg_post:
            norm_val = PhyTime
        else:
            raise ValueError("problems")
        norm_vals = [norm_val]*len(axis_vals)
        if avg_data is None:
            msg = f'For contour slice {slice}, avg_data must be provided'
            raise ValueError(msg)
        axis_index = y_coord_index_norm(avg_data,avg_data.CoordDF,axis_vals,norm_vals,y_mode)
    else:
        axis_index = coord_index_calc(avg_data.CoordDF,coord,axis_vals)
        if not hasattr(axis_index,'__iter__'):
            axis_index = [axis_index]
    # print(axis_index)
    return plane, coord, axis_index

def contour_indexer(array,axis_index,coord):
    if coord == 'x':
        indexed_array = array[:,:,axis_index].squeeze().T
    elif coord == 'y':
        indexed_array = array[:,axis_index].squeeze()
    else:
        indexed_array = array[axis_index].squeeze()
    return indexed_array

def vector_indexer(U,V,axis_index,coord,spacing_1,spacing_2):
    if isinstance(axis_index[0],list):
        ax_index = list(itertools.chain(*axis_index))
    else:
        ax_index = axis_index[:]
    if coord == 'x':
        U_space = U[::spacing_1,::spacing_2,ax_index]
        V_space = V[::spacing_1,::spacing_2,ax_index]
    elif coord == 'y':
        U_space = U[::spacing_2,ax_index,::spacing_1]
        V_space = V[::spacing_2,ax_index,::spacing_1]
        U_space = np.swapaxes(U_space,1,2)
        U_space = np.swapaxes(U_space,1,0)
        V_space = np.swapaxes(V_space,1,2)
        V_space = np.swapaxes(V_space,1,0)

    else:
        U_space = U[ax_index,::spacing_2,::spacing_1]
        V_space = V[ax_index,::spacing_2,::spacing_1]
        U_space = np.swapaxes(U_space,2,0)
        V_space = np.swapaxes(V_space,0,2)
        
    return U_space, V_space

def get_title_symbol(coord_dir,y_mode,local=True):
    title_symbol = ''
    if coord_dir =='y':
        if y_mode == 'half_channel':
            title_symbol = coord_dir
        elif y_mode == 'wall' and local:
            title_symbol = r"%s^+"%coord_dir
        elif y_mode == 'wall' and not local:
            title_symbol = r"%s^{+0}"%coord_dir
        elif y_mode == 'disp_thickness':
            title_symbol = r"%s/\delta^*"%coord_dir
        elif y_mode == 'mom_thickness':
            title_symbol = r"%s/theta^*"%coord_dir
    else:
        title_symbol = coord_dir

    return title_symbol