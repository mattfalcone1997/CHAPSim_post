#!/usr/bin/env python3

import CHAPSim_post as Im
import numpy as np

abs_path='/home/mfalcone/Desktop/iceberg_fastdata/Linear_test/case8_20200402-1459/'
h_list=[0,2,4]
quad_anal = Im.CHAPSim_Quad_Anal(h_list,path_to_folder=abs_path,time0=80)
print(quad_anal.QuadAnalDF)
fig, ax = quad_anal.line_plot(h_list,[5,10],'x',x_vals=0,y_mode='wall',norm=True)
fig.savefig("quad_anal_test.png")