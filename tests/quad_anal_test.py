#!/usr/bin/env python3

import CHAPSim_post as Im
import numpy as np
import os
abs_path='/home/mfalcone/Desktop/iceberg_fastdata/Linear_test/case8_20200402-1459/'
h_list=[0,2,4]
quad_anal = Im.CHAPSim_Quad_Anal(h_list,path_to_folder=abs_path,time0=80)
print(quad_anal.QuadAnalDF)
quad_anal.save_hdf('quad_anal.h5','w')
quad_anal1= Im.CHAPSim_Quad_Anal.from_hdf('quad_anal.h5')
print(quad_anal1.QuadAnalDF)

fig, ax = quad_anal1.line_plot(h_list,[5,10],'x',x_vals=0,y_mode='wall',norm=True)

dir_name='quad_test1'
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

fig.savefig(os.path.join(dir_name,"quad_anal_test.png"))