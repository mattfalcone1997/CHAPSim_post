#!/usr/bin/env python3

import CHAPSim_post as cp
import os
abs_path='/home/mfalcone/Desktop/iceberg_fastdata/Linear_test/case8_tanh_20200504-1759'

meta_data = cp.CHAPSim_meta(abs_path)
avg_data = cp.CHAPSim_AVG(275.0,time0=80,path_to_folder=abs_path)
inst_data = cp.CHAPSim_Inst(275.0,path_to_folder=abs_path)
autocov_data = cp.CHAPSim_autocov2('u','u',max_x_sep=0,path_to_folder=abs_path,time0=80)

file_name_avg='avg_data_test.h5'
file_name_inst='inst_data_test.h5'
file_name_uu='autocov_test.h5'

avg_data.save_hdf(file_name_avg,'w')
avg_data2 = cp.CHAPSim_AVG.from_hdf(file_name_avg)

inst_data.save_hdf(file_name_inst,'w')
inst_data2 = cp.CHAPSim_Inst.from_hdf(file_name_inst)

autocov_data.save_hdf(file_name_uu,'w')
autocov_data2 = cp.CHAPSim_autocov2.from_hdf(file_name_uu)

print(avg_data)
print(avg_data2)
print(inst_data)
print(inst_data2)
print(autocov_data.autocorrDF)
print(autocov_data2.autocorrDF)

