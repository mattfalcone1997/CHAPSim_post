import os
from pathlib import PurePath
import numpy as np
import sys

from CHAPSim_post import rcParams

__all__ = ["max_time_calc","time_extract"]

def check_paths(path_to_folder,*folder_options):

    check_path_exists(path_to_folder,False)

    for folder in folder_options:
        full_path = os.path.join(path_to_folder,folder)
        if os.path.isdir(full_path):
            return full_path

    msg = "Directory(ies) %s cannot be found in path: %s"%(folder_options,path_to_folder)
    raise FileNotFoundError(msg)
    
def check_path_exists(file_folder_path,check_file=None):
    
    if check_file is None:
        check_func, handle_name = (os.path.exists, "Path")
    elif check_file:
        check_func, handle_name = (os.path.isfile, "File")
    else:
        check_func, handle_name = (os.path.isdir, "Directory")
    
    if check_func(file_folder_path):
        return True
    else:
        comps = PurePath(file_folder_path).parts
        for i in range(1,len(comps)):
            partial_path = os.path.join(*comps[:i])
            if not os.path.exists(partial_path):
                if i == len(comps) -1:
                    msg = "%s provided \"%s\" does not exist\n"%(handle_name,file_folder_path) +\
                            "Sub-path \"%s\" not found"%partial_path

                    raise FileNotFoundError(msg)
                else:
                    continue
            else:
                break

def file_extract(path_to_folder,abs_path=True):
    full_path = check_paths(path_to_folder,'2_averaged_rawdata',
                                                            '2_averagd_D')
    # if abs_path:
    #     mypath = os.path.join(path_to_folder,'2_averagd_D')
    # else:
    #     mypath = os.path.abspath(os.path.join(path_to_folder,'2_averagd_D'))
    file_names = [f for f in os.listdir(full_path) if f[:8]=='DNS_peri']
    return file_names       

def time_extract(path_to_folder,abs_path=True):
    file_names = file_extract(path_to_folder,abs_path)
    time_list =[]
    for file in file_names:
        time_list.append(float(file[20:35]))

    times = sorted(set(time_list))
    if not times:
        msg = "No averaged results to give the times list"
        raise RuntimeError(msg)
    return times

def max_time_calc(path_to_folder,abs_path=True):
    if isinstance(path_to_folder,list):
        max_time = np.float('inf')
        for path in path_to_folder:
            max_time=min(max_time,max_time_calc(path,abs_path))
    else:
        max_time=max(time_extract(path_to_folder,abs_path))
    return max_time

def check_list_vals(x_list):
    msg =  f"x_list must be of type float, int, or an iterable of them not %s"%type(x_list)
    if isinstance(x_list,(float,int,np.floating)):
        x_list=[x_list]
    elif hasattr(x_list,'__iter__'):
        if not all([isinstance(x,(float,int,np.floating)) for x in x_list]):
            raise TypeError(msg)
    else: # x_list is a tuple or list
        raise TypeError(msg)
    
    return x_list


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

def meshgrid(*args):
    if not all([isinstance(arg,np.ndarray) for arg in args]):
        msg = "The inputs must be numpy arrays"
        raise TypeError(msg)
    if not all([arg.ndim == 1 for arg in args]):
        msg = "Each numpy array must have a single dimension"
        raise ValueError(msg)

    out_arr_size = tuple(arg.size for arg in args)
    array_tuple = tuple(np.zeros(out_arr_size) for arg in args)
    
    for i, arg in enumerate(args):
        for j in range(len(args[i])):
            slice_list = [slice(None)]*len(args)
            slice_list[i] = j
            array_tuple[i][tuple(slice_list)] = args[i][j]

    return array_tuple

def AVG_Style_overline(label):
    return "\overline{%s}"%label

def AVG_Style_langle(label):
    return "\langle %s\rangle"%label

def GetStyle_func():
    module = sys.modules[__name__]

    if not hasattr(module,"AVG_Style_%s"%rcParams['AVG_Style']):
        msg = "No style named %s found"%rcParams['AVG_Style']
        raise AttributeError(msg)
    else:
        return getattr(module,"AVG_Style_%s"%rcParams['AVG_Style'])

