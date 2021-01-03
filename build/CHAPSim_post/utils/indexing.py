"""
## indexing.py
This is a module to provide indexing routines for CHAPSim_post
"""

import warnings

def Y_plus_index_calc(AVG_DF,CoordDF,coord_list,x_vals=None):
    if x_vals is not None:
        index_list = coord_index_calc(CoordDF,'x',x_vals)
        
    y_coords = CoordDF['y']
    if isinstance(coord_list,float) or isinstance(coord_list,int):
        coord_list = [coord_list]
    elif not isinstance(coord_list,list):
        raise TypeError("\033[1;32 coord_list must be of type float, list or int")
    avg_time = AVG_DF.flow_AVGDF.index[0][0]
    # if hasattr(AVG_DF,"par_dir"):
    #     par_dir = AVG_DF.par_dir
    #     u_tau_star, delta_v_star = cpar.wall_unit_calc(AVG_DF,avg_time,par_dir)
    # else:
    u_tau_star, delta_v_star = AVG_DF.wall_unit_calc(avg_time)
    
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