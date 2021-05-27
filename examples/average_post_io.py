# This is an example for script for producing average images
# 
# This script produces the following images
# * params.png - plots C_f and H
# * Reynolds_stresses_y.png - Reynolds stress vs y
# * uu_budget.png - uu budget
# * Reynolds_x.png - max Reynolds stress against x
#
#
# This particular example was plotted using the moving wall module
# Although the functions would be unchanged if the post module
# was used. The data used here was from a moving wall case and for
# reasons specified in the Developer's Guide, the post module 
# would result in errors
#===================================================================

import CHAPSim_post.post  as cp

import CHAPSim_post.plot as cplt

from CHAPSim_post import utils


import matplotlib as mpl
import os


# Set up processing

mpl.rcParams['figure.dpi']=600
mpl.rcParams['axes.titlesize']=8
mpl.rcParams['font.size'] = 9
mpl.rcParams['lines.linewidth'] = 0.75
mpl.rcParams['lines.markeredgewidth'] =  0.5
mpl.rcParams['lines.markersize'] =  3.0
mpl.rcParams['lines.marker'] =  "None"
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.labelspacing'] = 0.3
mpl.rcParams['legend.edgecolor'] = "w"
mpl.rcParams['legend.framealpha'] = 0.8
mpl.rcParams['legend.columnspacing'] = 0.3
mpl.rcParams['legend.handletextpad'] = 0.3
mpl.rcParams['legend.handlelength'] = 1.5
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['patch.linewidth'] = 0.75
mpl.rcParams['axes.grid'] = False

# path to remote HPC directory
path = "/home/username/remote_HPC_dir"

# images directory
output_dir = "pictures"

# If output directory has not been created, create it
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)



# If the data has been saved to file, extract from file otherwise 
# extract from Results folder

if not os.path.isfile('avg_data.h5'):
    max_time = utils.max_time_calc(path)
    avg_data = cp.CHAPSim_AVG_io(max_time, path_to_folder=path,time0=80)
    avg_data.save_hdf('avg_data.h5','w')
else: 
    avg_data = cp.CHAPSim_AVG_io.from_hdf("avg_data.h5")

## Printing the case setup

print(avg_data._metaDF)

# Setting streamwise processing visualisations
x_loc_mut1=[6,7,8,10,11,12]
x_loc_mut2=[12,14,16,18,21,27]
x_loc_budget = [6,8,10,12,16,24]

#=============================================================

## Plotting skin friction and shape factor using twinned axes

# altering the plotting kwargs so the marker is switched off
line_dict = {"marker":""}

fig, ax = avg_data.plot_skin_friction(figsize=[5,3.5],line_kw=line_dict)

# Setting the tick format to scientific notation
ax.ticklabel_format(axis='y',style='sci',scilimits=[0,0])

#adjusting y limits
ax.set_ylim([0.005,0.009])

# Creating a twinned axis which will use the right y axis
ax1 = ax.twinx()

fig, ax1 = avg_data.plot_shape_factor(fig=fig,ax=ax1,line_kw=line_dict)

# Changing the y limit
ax1.set_ylim(top=1.7)

# Set x limits to the limits of the domain
ax.set_xlim([0,30])

# Setting up the legend labels so that they will be plotted on 1
# figure legend as opposed to axis legend

handles, labels = ax.get_legend_handles_labels()
handles1, labels1 = ax1.get_legend_handles_labels()
handles.extend(handles1); labels.extend(labels1)

# Add legend
fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(0.8,0.8),fontsize=12)

#Save plot
fig.savefig(os.path.join(output_dir,"params"))

#=============================================================

## Plotting the Reynolds stresses

# Setting parameters for the property cycle
prop_cycle = {'color' : 'k',
            'linewidth' : [0.75,0.75,0.75,0.75,0.5,0.5,0.5,0.5],
            "marker":""}

# update property cycler with dict
cplt.update_prop_cycle(**prop_cycle)

# Create plots with the subplots method
fig, ax = cplt.subplots(3,2,figsize=(6,5))


# Plot all the normal Reynolds stresses
fig,ax[0,0] = avg_data.plot_Reynolds('uu',x_loc_mut1,norm=None,Y_plus=False,fig=fig,ax=ax[0,0])

fig,ax[0,1] = avg_data.plot_Reynolds('uu',x_loc_mut2,norm=None,Y_plus=False,fig=fig,ax=ax[0,1])

fig,ax[1,0] = avg_data.plot_Reynolds('vv',x_loc_mut1,norm=None,Y_plus=False,fig=fig,ax=ax[1,0])

fig,ax[1,1] = avg_data.plot_Reynolds('vv',x_loc_mut2,norm=None,Y_plus=False,fig=fig,ax=ax[1,1])

fig,ax[2,0] = avg_data.plot_Reynolds('ww',x_loc_mut1,norm=None,Y_plus=False,fig=fig,ax=ax[2,0])

fig,ax[2,1] = avg_data.plot_Reynolds('ww',x_loc_mut2,norm=None,Y_plus=False,fig=fig,ax=ax[2,1])


for i,a in enumerate(ax.flatten()):

    # Shift so wall starts at 0 for channel flow
    a.shift_xaxis(1)

    # remove all legends
    a.get_legend().remove()
    
    # Ensure bottom of each axis starts at 0
    a.set_ylim(bottom=0)
    
# remove y axis label if not on the left
for a in ax[:,1]:
    a.set_ylabel("")

# remove labels from all rows except bottom
for a in ax[:-1,:].flatten():
    a.set_xlabel("")

# Add legend to first plot on each column
ax[0,0].legend()
ax[0,1].legend()

# Add tight layout
fig.tighter_layout()

#Save plot
fig.savefig(os.path.join(output_dir,"Reynolds_stresses_y"))

#=============================================================

## Plotting u'u' budgets
# Plotting budgets in initial wall units

# Calculating budgets
u2_budget = cp.CHAPSim_budget_io('u','u',avg_data=avg_data)

# Creating subplots
fig, ax = cplt.subplots(3,2,figsize=[6,3*2.2],dpi=600)

cplt.reset_prop_cycler()

# Plot budgets against y
fig, ax = u2_budget.budget_plot(x_loc_budget,wall_units=False,fig=fig,ax=ax.flatten())

# Get viscous scales
u_tau_star ,delta_v_star = avg_data.wall_unit_calc()

# Set x scale to log and normalise axes
# budget scale = UU/T
# viscous scale = u_tau^3/delta_v

for a in ax:
    a.set_xscale('log')
    a.normalise('y',u_tau_star[0]**3/delta_v_star[0])
    a.normalise('x',delta_v_star[0])

    # remove all x labels
    a.set_xlabel("")

# tight layout but leave space for legend
fig.tight_layout(rect=(0,0.12,1,1))

# remove y labels on right column
ax=ax.reshape((3,2))
for a in ax[:,1]:
    a.set_ylabel("")

# set label to bottom row of axes
for a in ax[-1,:].flatten():
    a.set_xlabel(r"$y^{+0}$")

# Save plot
fig.savefig(os.path.join(output_dir,"uu_budget"))

#=============================================================

## Plot wall-normal maximum of normal Reynolds stress

# Create subplots
fig, ax = cplt.subplots(figsize=[6,2.7])

# remove markers from this plot
cplt.update_prop_cycle(marker='')

# Streamwise Reynolds stress
fig,ax = avg_data.plot_Reynolds_x('uu','max',fig=fig,ax=ax)

# Create twinned axes
ax2 = ax.twinx()

# Wall-normal Reynolds stress
fig, ax2 = avg_data.plot_Reynolds_x('vv','max',fig=fig,ax=ax2)

# Spanwise Reynolds stress
fig, ax2 = avg_data.plot_Reynolds_x('ww','max',fig=fig,ax=ax2)

# Set y axis labels
ax.set_ylabel(r"$\overline{u'u'}$")
ax2.set_ylabel(r"$\overline{v'v'}\ \ \ \ \ \overline{w'w'}$")

# calculate maximum values
max_y=max([max(a.get_ydata()) for a in ax2.get_lines()])

# set limits
ax.set_ylim(bottom=0.02)
ax2.set_ylim([0,2*max_y])

# Set text label on plots
ax.text(26,0.09,r"$\overline{u'u'}_{max}$")
ax2.text(26,0.021,r"$\overline{w'w'}_{max}$")
ax2.text(26,0.009,r"$\overline{v'v'}_{max}$")

# Set x limits to the limits of the domain
ax.set_xlim([0,30])

# tight layout
fig.tight_layout()

#Save plot
fig.savefig(os.path.join(output_dir,"Reynolds_x"))
