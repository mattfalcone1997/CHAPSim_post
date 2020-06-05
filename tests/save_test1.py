#!/usr/bin/env python3

import CHAPSim_post as cp
import CHAPSim_Tools as CT
import pandas as pd
import h5py

abs_path='/home/mfalcone/Desktop/iceberg_fastdata/Linear_test/case8_tanh_20200504-1759'
meta_data = cp.CHAPSim_meta(abs_path)
avg_data = cp.CHAPSim_AVG(275.0,time0=80,path_to_folder=abs_path)

file = h5py.File("meta_test_store.h5",'w')
group_name="meta_data"
meta = file.create_group(group_name)
meta.attrs["path_to_folder"] = meta_data.path_to_folder.encode('utf-8')
file.close()
#meta_hdf = pd.HDFStore("meta_test_store.h5",'a')
#meta_hdf.put("metaDF",meta_data.metaDF,format='fixed',data_columns=True)
meta_data.metaDF.to_hdf('meta_test_store.h5',key=group_name+'/metaDF',mode='a',format='fixed',data_columns=True)
meta_data.metaDF.to_hdf('meta_test_store.h5',group_name+'/CoordDF',mode='a',format='fixed',data_columns=True)
meta_data.metaDF.to_hdf('meta_test_store.h5',group_name+'/Coord_ND_DF',mode='a',format='fixed',data_columns=True)

file = h5py.File("meta_test_store.h5",'r')
print(file[group_name].keys())
# meta.create_dataset("path_to_folder",data=meta_data.path_to_folder,shape=)
metaDF_read = pd.read_hdf("meta_test_store.h5",key=group_name+'/metaDF')
print(metaDF_read)
print(file[group_name].attrs['path_to_folder'].decode('utf-8'))
