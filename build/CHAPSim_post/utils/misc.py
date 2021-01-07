import os

__all__ = ["max_time_calc"]

def file_extract(path_to_folder,abs_path=True):
    if abs_path:
        mypath = os.path.join(path_to_folder,'2_averagd_D')
    else:
        mypath = os.path.abspath(os.path.join(path_to_folder,'2_averagd_D'))
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
        max_time=max(time_extract(path_to_folder,abs_path))

def check_list_vals(x_list):

    if isinstance(x_list,(float,int)):
        x_list=[x_list]
    elif not isinstance(x_list,(tuple,list)):
        msg =  f"x_list must be of type float, int, tuple or list"
        raise TypeError(msg)
    else: # x_list is a tuple or list
        if not all([isinstance(x,(float,int)) for x in x_list]):
            msg = "If tuple or list provided, all items must be of instance float or int"
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