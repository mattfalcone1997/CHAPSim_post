import CHAPSim_post.post as cp
from CHAPSim_post import utils

path = "/home/username/remote_HPC_dir"


max_time = utils.max_time_calc(path)
avg_data = cp.CHAPSim_AVG_io(max_time, path_to_folder=path,time0=80)


time = 112

# extract from results directory if not already saved
inst_data = cp.CHAPSim_Inst(time, path_to_folder=path)


fluct_data = cp.CHAPSim_fluct_io(inst_data,avg_data)

lambda2DF = inst_data.lambda2_calc()


index = (time,'u')
dataDF = fluct_data.fluctDF.create_slice(index)

dataDF.concat(lambda2DF)

dataDF.to_vtk("fluct_lambda2.vts")
