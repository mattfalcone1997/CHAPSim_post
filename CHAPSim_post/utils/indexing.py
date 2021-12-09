"""
## indexing.py
This is a submodule of utils to provide indexing routines
and some utilities for contour and vector plots.
"""
import numpy as np

import warnings
import itertools

from CHAPSim_post.utils import misc_utils
__all__ = ["coord_index_calc"]

# def Y_plus_index_calc(AVG_DF,avg_time,CoordDF,coord_list,x_vals=None):
#     if x_vals is not None:
#         index_list = coord_index_calc(CoordDF,'x',x_vals)
        
#     y_coords = CoordDF['y']
#     if isinstance(coord_list,float) or isinstance(coord_list,int):
#         coord_list = [coord_list]
#     elif not isinstance(coord_list,list):
#         raise TypeError("\033[1;32 coord_list must be of type float, list or int")

#     _, delta_v_star = AVG_DF.wall_unit_calc(avg_time)
    
#     if x_vals:
#         Y_plus_list=[]
#         if isinstance(index_list,list):
#             for index in index_list:
#                 Y_plus_list.append((1-abs(y_coords[:int(y_coords.size/2)]))/delta_v_star[index])
#         else:
#             Y_plus_list.append((1-abs(y_coords[:int(y_coords.size/2)]))/delta_v_star[index_list])

#         Y_plus_index_list = []
#         for i in range(len(coord_list)):
#             for j in range(Y_plus_list[0].size):
#                 if Y_plus_list[i][j+1]>coord_list[i]:
#                     if abs(Y_plus_list[i][j+1]-coord_list[i]) > abs(Y_plus_list[i][j]-coord_list[i]):
                        
#                         Y_plus_index_list.append(j)
#                         break
#                     else:
#                         Y_plus_index_list.append(j+1)
#                         break
                    
#     else:
#         Y_plus = (1-abs(y_coords[:int(y_coords.size/2)]))/delta_v_star[0]
        
#         Y_plus_index_list = []
#         for coord in coord_list:
#             try:
#                 for i in range(Y_plus.size):
#                     if Y_plus[i+1]>coord:
#                         if abs(Y_plus[i+1]-coord)>abs(Y_plus[i]-coord):
#                             Y_plus_index_list.append(i)
#                             break
#                         else:
#                             Y_plus_index_list.append(i+1)
#                             break 
#             except IndexError:
#                 warnings.warn("\033[1;33Value in coord_list out of bounds: "\
#                                  + "Y_plus given: %g, max Y_plus: %g. Ignoring values beyond this" % (coord,max(Y_plus)))
#                 return Y_plus_index_list
#     if len(coord_list)==1:
#         return Y_plus_index_list[0]
#     else:
#         return Y_plus_index_list

# def y_coord_index_norm(avg_data,avg_time,coord_list,x_vals=None,mode='half_channel'):
#     if mode=='half_channel':
#         norm_distance=np.ones((avg_data.NCL[0]))
#     elif mode == 'disp_thickness':
#         norm_distance, _,_ = avg_data._int_thickness_calc(avg_time)
#     elif mode == 'mom_thickness':
#         _, norm_distance, _ = avg_data._int_thickness_calc(avg_time)
#     elif mode == 'wall':
#         _, norm_distance = avg_data._wall_unit_calc(avg_time)
#     else:
#         raise ValueError("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
#                                 " or 'wall. Value used was %s\n"%mode)
    
    
#     if x_vals is None:
#         x_index=list(range(avg_data.shape[-1]))
#     else:
#         x_vals = misc_utils.check_list_vals(x_vals)
#         x_index =[avg_data._return_index(x) for x in x_vals]

#     y_indices = []
#     for i,x in enumerate(x_index):
#         norm_coordDF = (1-abs(avg_data.CoordDF))/norm_distance[x]
#         y_index = coord_index_calc(norm_coordDF,'y', coord_list)
#         y_indices.append(y_index)

#     return y_indices
    
def coord_index_calc(CoordDF,comp,coord_list):
    coords = CoordDF[comp]
    coord_list = misc_utils.check_list_vals(coord_list)
    
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
                msg = "Value in coord_list out of bounds: "\
                    + "%s coordinate given: %g, max %s coordinate:" % (comp,coord,comp)\
                    + " %g. Ignoring values beyond this" % max(coords)
                raise IndexError(msg) from None

    return index_list

def true_coords_from_coords(CoordDF,comp,coord_list):
    indices = coord_index_calc(CoordDF, comp, coord_list)
    return CoordDF[comp][indices]

# def ycoords_from_coords(avg_data,coord_list,x_vals=None,mode='half_channel'):
#     if mode=='half_channel':
#         norm_distance=np.ones((avg_data.NCL[0]))
#     elif mode == 'disp_thickness':
#         norm_distance, _,_ = avg_data._int_thickness_calc(avg_data.flow_AVGDF.index[0][0])
#     elif mode == 'mom_thickness':
#         _, norm_distance, _ = avg_data._int_thickness_calc(avg_data.flow_AVGDF.index[0][0])
#     elif mode == 'wall':
#         _, norm_distance = avg_data._wall_unit_calc(avg_data.flow_AVGDF.index[0][0])
#     else:
#         raise ValueError("The mode of normalisation must be 'half_channel', 'disp_thickness','mom_thickness',"+\
#                                 " or 'wall. Value used was %s\n"%mode)

#     indices = y_coord_index_norm(avg_data,coord_list,x_vals,mode)

#     if x_vals is None:
#         x_index=list(range(avg_data.shape[-1]))
#     else:
#         x_vals = misc_utils.check_list_vals(x_vals)
#         x_index =[avg_data._return_index(x) for x in x_vals]

#     true_ycoords = []
#     for x,index in zip(x_index,indices):
#         norm_coordDF = (1-abs(avg_data.CoordDF))/norm_distance[x]
#         true_ycoords.append(norm_coordDF['y'][index])

#     return true_ycoords

# def ycoords_from_norm_coords(avg_data,coord_list,x_vals=None,mode='half_channel'):
#     indices = y_coord_index_norm(avg_data,coord_list,x_vals,mode)
#     true_ycoords = []
#     for index in zip(indices):
#         true_ycoords.append(avg_data.CoordDF['y'][index])
    
#     return true_ycoords

def coords_from_norm_coords(CoordDF,comp,coord_list):
    indices = coord_index_calc(CoordDF,comp,coord_list)
    return CoordDF[comp][indices]




def contour_plane(plane,axis_vals,avg_data,y_mode,PhyTime):
    if plane not in ['xy','zy','xz']:
        plane = plane[::-1]
        if plane not in ['xy','zy','xz']:
            msg = "The contour slice must be either %s"%['xy','yz','xz']
            raise KeyError(msg)
    slice_set = set(plane)
    coord_set = set(list('xyz'))
    coord = "".join(coord_set.difference(slice_set))

    axis_vals = misc_utils.check_list_vals(axis_vals)

    if coord == 'y':
        tg_post = True if all([x == 'None' for x in avg_data.flow_AVGDF.outer_index]) else False
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
        axis_index = y_coord_index_norm(avg_data,axis_vals,norm_vals,y_mode)
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