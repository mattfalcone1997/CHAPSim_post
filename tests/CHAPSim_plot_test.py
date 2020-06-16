#!/usr/bin/env python3

import CHAPSim_plot as cplt
import numpy as np
fig, ax = cplt.subplots()


ax.cplot(np.linspace(0,1,100),np.linspace(0,1,100))#,label='plot1')
ax.cplot(np.linspace(0,1,100),2*np.linspace(0,1,100))#,label='plot2')
ax.cplot(np.linspace(0,1,100),3*np.linspace(0,1,100))#,label='plot3')
ax.cplot(np.linspace(0,1,100),4*np.linspace(0,1,100))#,label='plot4')
ax.cplot(np.linspace(0,1,100),5*np.linspace(0,1,100))#,label='plot5')
ax.cplot(np.linspace(0,1,100),6*np.linspace(0,1,100))#,label='plot6')
ax.set_xlabel("test $x$ ")
ax.set_ylabel("test $y$ ")
labels=['plot1','plot2','plot3','plot4','plot5','plot6',]
ax.clegend(labels,vertical=False,ncol=3)
fig.savefig("plot_test.png")
ax.set_label_fontsize(18)
ax.toggle_default_linestyle(labels,vertical=False,ncol=3)
print(ax.get_legend_handles_labels())

fig.savefig("plot_test1.png")