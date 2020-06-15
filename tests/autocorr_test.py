#!/usr/bin/env  python3
import CHAPSim_post as cp 
import matplotlib as mpl
import os
comp1='u' ; comp2 = 'u'
abs_path = '/home/mfalcone/Desktop/archer_work/CHAPSim_projects/Linear_Sims/case4_20200323-1202/'
autocorr_data = cp.CHAPSim_autocov2(comp1,comp2,path_to_folder=abs_path,
                                    time0=80,
                                    max_x_sep=0)
print(autocorr_data.autocorrDF)
autocorr_data.save_hdf('autocorr_data.h5','w')
autocorr_data1=cp.CHAPSim_autocov2.from_hdf('autocorr_data.h5')
print(autocorr_data1.autocorrDF)

fig, ax = autocorr_data.autocorr_contour_x('z',[20,40,100],axis_mode='wall')
fig1, ax = autocorr_data.autocorr_contour_y([10,13,17,24],'z',Y_plus=True,
                        Y_plus_max=50)

fig2, axes = autocorr_data.plot_spectra('z',[10,13,17,24],[0.6,0.8],y_mode='disp_thickness')
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
fig3, axes = autocorr_data.spectrum_contour('z',[0.6,0.8],axis_mode='disp_thickness')
for ax in axes:
    ax.axes.set_yscale('log')
fig4, axes = autocorr_data.plot_autocorr_line('z',[10,13,17,24],[0.6,0.8],y_mode='disp_thickness')
if not os.path.isdir('autocorr1'):
    os.mkdir('autocorr1')

fig.savefig(os.path.join('autocorr1',"autocor1.png"))
fig1.savefig(os.path.join('autocorr1',"autocor_y_plus1.png"))
fig2.savefig(os.path.join('autocorr1',"spectra1.png"))
fig3.savefig(os.path.join('autocorr1',"spectra_contour1.png"))
fig4.savefig(os.path.join('autocorr1',"autocor_line1.png"))