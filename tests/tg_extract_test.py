import CHAPSim_post as cp

abs_path='/home/matthew/Desktop/iceberg_fastdata/Linear_test/case8_tanh_20200504-1759'

avg_data = cp.CHAPSim_AVG_tg_base([274.0,275.0],path_to_folder=abs_path)
print(avg_data.flow_AVGDF)
print(avg_data.get_times())
avg_data.save_hdf("avg_data_tg.h5",'w')
avg_data2 = cp.CHAPSim_AVG_tg_base.from_hdf("avg_data_tg.h5")
print(avg_data2.flow_AVGDF)
print(avg_data2.get_times())

fig,ax = avg_data2.avg_line_plot(avg_data2.get_times(),'u')
fig.savefig("avg_tg_test.png")

print(avg_data2.bulk_velo_calc(avg_data2.get_times(),'u'))

avg_data_io = cp.CHAPSim_AVG([275.0],path_to_folder=abs_path)
avg_data_io.save_hdf("avg_io_new_test.h5",'w')
avg_data_io2 = cp.CHAPSim_AVG.from_hdf("avg_io_new_test.h5")