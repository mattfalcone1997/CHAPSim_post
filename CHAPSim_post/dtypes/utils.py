
import itertools
import numpy as np

from CHAPSim_post.utils import misc_utils,indexing

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
        axis_index = indexing.y_coord_index_norm(avg_data,axis_vals,norm_vals,y_mode)
    else:
        axis_index = indexing.coord_index_calc(avg_data.CoordDF,coord,axis_vals)
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

def line_indexer(array,axis_index,axis_dim):
    if axis_dim == 0:
        return array[axis_index]
    elif axis_dim == 1:
        return array[:,axis_index].T
