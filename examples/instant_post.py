import CHAPSim_post.post as cp
import CHAPSim_post.plot as cplt
from CHAPSim_post import utils

import os
import matplotlib as mpl

# path to remote HPC directory
path = "/home/username/remote_HPC_dir"

# images directory
output_dir = "instant_pictures"


# update matplotlib rcParams 

mpl.rcParams['figure.dpi']=600
mpl.rcParams['axes.titlesize']=8
mpl.rcParams['font.size'] = 9
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['patch.linewidth'] = 0.75

# If output directory has not been created, create it
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# extract from results directory if not already saved
if not os.path.isfile('avg_data.h5'):
    max_time = utils.max_time_calc(path)
    avg_data = cp.CHAPSim_AVG_io(max_time, path_to_folder=path,time0=80)
    avg_data.save_hdf('avg_data.h5','w')

else: # extract from file
    avg_data = cp.CHAPSim_AVG_io.from_hdf("avg_data.h5")

time = 112

# extract from results directory if not already saved
if not os.path.isfile('inst_data.h5'):
    max_time = utils.max_time_calc(path)
    inst_data = cp.CHAPSim_Inst(time, path_to_folder=path)
    inst_data.save_hdf('inst_data.h5','w')
else: # extract from file
    inst_data = cp.CHAPSim_Inst.from_hdf("inst_data.h5")

# Create fluctuating data structure
fluct_data = cp.CHAPSim_fluct_io(inst_data,avg_data)

# create subplots
fig,ax = cplt.subplots(2,figsize=[6,3])

# plot streamwise and wall-normal fluctuation
fig, ax[0] = fluct_data.plot_contour('u',5,fig=fig, ax=ax[0])

fig, ax[1] = fluct_data.plot_contour('v',5,fig=fig, ax=ax[1])

# set colorbar limits
ax[0].set_clim([-0.45,0.7])
ax[1].set_clim([-0.1,0.1])

# remove x label from top plot
ax[0].axes.set_xlabel(None)

# better to use tight fgure before setting aspect
fig.tight_layout()
ax[0].axes.set_aspect('equal')
ax[1].axes.set_aspect('equal')

# save plot
fig.savefig(os.path.join(output_dir,'fluct_contour.png'))