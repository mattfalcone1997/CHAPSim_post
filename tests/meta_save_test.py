#!/usr/bin/env python3

import CHAPSim_post as cp
import os
abs_path='/home/mfalcone/Desktop/iceberg_fastdata/Linear_test/case8_tanh_20200504-1759'
meta_data = cp.CHAPSim_meta(abs_path)
file_name = 'meta_test.h5'
meta_data.save_hdf(file_name,'w')
# print(os.path.isfile(file_name))

meta_data2 = cp.CHAPSim_meta.from_hdf(file_name)
print(meta_data2.metaDF)
print(meta_data2.CoordDF)
print(meta_data2.Coord_ND_DF)
