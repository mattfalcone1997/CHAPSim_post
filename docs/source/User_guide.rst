.. _user_guide:

==========
User Guide
==========

Introduction
------------

This introduction serves to give an overview of the functionality provided by the CHAPSim_post, a script-based Python 3 library for the CHAPSim Direct Numerical Simulation code. This library does not represent a finished article but is under continuous development as functionality is added and extended. CHAPSim_post can provide a high-level framework on which your own functionality can be built. If you wish to collaborate on extending functionality you can e-mail the main developer at mfalcone1@sheffield.ac.uk. More information on the internal structure of the code can be found in :ref:`develop_guide`.

A non-exhaustive list of functionality:

* Extract rawdata from CHAPSim results directories
* Well developed for isothermal channel flow with extensions in development for pipe and thermal flows.
* Process and store the data in an easy-to-access framework
* Provide flexible visualisation tools based on the matplotlib library
* Functionality to store and access the process data
* Works on both personal computers and HPCs
* Output data into a format readable by Paraview.


Installation Guide
------------------

It is recommended that CHAPSim_post is controlled using ``conda``. This requires and installation of either Anaconda or Miniconda. Note Anaconda install is around 1.5GB where as Miniconda is only around 70MB although you will have to manually install many of the modules that you typically use. This has currently been tested on python 3.7, it is not known whether it works correctly on later versions. The latest version of anaconda or miniconda can be downloaded on Linux using the commands:

.. code-block: bash

   wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86.sh

Instructions for the install can be readily found online. 

To install the code and its dependent modules the script ``setup_CHAPSim_post`` can be executed. This does the following:

#. Creates a Conda environment called CHAPSim_post and installs all the necessary modules
#. Executes ``conda develop`` to add CHAPSim_post's build directory to the environment's search path.  
#. Modifies the Conda environment's environment activate script such that the path of CHAPSim_post's build directory is added the `PYTHONPATH` while the environment is active.
#. Executes ``python setup.py build --build-lib build/`` to build the python and src files in the build directory.
#. Performs an import test, if this fails the environment is removed

Note that ``python setup.py install`` could have also been called. The module can also be installed directly by using ``pip`` but given that often usage of this module involves code development this is not generally recommended. Now idiosyncracies of the install on a personal computer versus and HPCs will be discussed

On your personal computer
^^^^^^^^^^^^^^^^^^^^^^^^^

The guide above can generally be used on your personal computer 

On an HPC
^^^^^^^^^

On HPCs, the installation is sometimes a little more complex. It may be possible to piggy-back the installation of Anaconda (if present) on the HPC and go straight to the execution of the script. However, sometimes the version of Anaconda is outdated and you will typically lack the admin privileges required to update it. It is therefore recommended that you install a version of *Miniconda* on your personal e.g. home directory which you will then have control of. Anaconda is not recommended due its large install size whichcan take an age on HPCs' login nodes.

It may also be necessary to ``module load`` an HPCs modules for the install. Some of the code has been written in ``Cython`` and ``Fortran`` hence these are compiled when the ``setup.py`` script is run.

Running Scripts
---------------

It is generally recommended that this code is run from python scripts although the ``iPython`` terminal can also be used and is installed in the Conda environment. The code has not been test on Jupiter notebooks although dues to the often large memory requirements on processing data, it is not advised.

Before the iPython terminal or a script is run, the Conda environment needs to be activated. This is done by executing ``conda activate CHAPSim_post`` in the terminal. The script can then be run using ``python script_name.py``. After you are finished you can call ``conda deactivate``

If you are running on HPCs, note that the code is designed to run on the shared memory nodes and can therefore use a maximum of one node which should be sufficient for any postprocessing tasks. Also note that if you are calculating any quantities that use compiled code, the libraries used during compilation must be loaded to give python access to the runtime libraries or the code will fail. Note that it shouldn't matter if you are not using this code in your script as these compiled modules are not imported until they are needed.


Extracting Data and Visualisation
---------------------------------

There is a discussion on ways of accessing data from remote filesystems in the Best Practice Guide: :ref:`access_data`. If you are using an HPC this should not be an issue and you should be able to access the data directly. An example of data visualisation for the Averaged and instantaneous data will now be shown. The code snippets should make it clear that the code is designed to be high-level: the details of implementation are not needed at the point of use. It should also be noted that domain information is handled implicitly from the readdata.ini file. As a result, CHAPSim_post will work out whether your flow is a pipe or channel flow.

Averaged Data
^^^^^^^^^^^^^

In the various modules of CHAPSim_post, the class ``CHAPSim_AVG_io`` is invoked to extract the statistics of the non-periodic results. For the periodic results, the ``CHAPSim_AVG_tg`` classes are used which are currently optimised for temporally developing flows although this will be improved going forward. The ``CHAPSim_AVG`` classes generally are the most important classes in the code. Not only do they extract and visualise data but also have methods for calculating useful quantites such as the wall shear stress and integral thicknesses which are used in the normalisation of plots throughout the code. An example of some of this functionality is shown later in :ref:`plot_derived`. Some more details on their implementation is given in the :ref:`develop_guide`.

.. code-block:: python

   import CHAPSim_post.post as cp
   from CHAPSim_post import utils
   import os
   
   # path to data on remote HPC mount using the sshfs command
   path = "home/username/mounted_dirs/HPC1/DNS_data1/"

   # Output directory for pictures
   pictures_dir = "Example_pictures"
   
   # Calculates the final time in the results directory, one
   # usually want to final value as the data is asymptotically Averaged
   max_time = utils.max_time_calc(path)

   # Instantiating a CHAPSim_AVG class which extracts and processes
   # the data required for mean, second-order statistics, and budgets
   avg_data = cp.CHAPSim_AVG_io(max_time,path_to_folder=path,time0=80.0)

   # Plotting the skin friction coefficient
   fig, ax = avg_data.plot_skin_friction()

   #Saving the plot
   fig.savefig(os.path.join(pictures_dir,'C_f.png'))

This code performs the following

#. Extracts the data from the averaged rawdata directory at time ``max_time``.
   
   * The averaged data at 80.0 is also extracted (keyword argument ``time0``) so that an earlier average can be subtracted to eliminate effect of initial transient

#. Conducts preliminary processing of data. For example finding the Reynolds stress.

   * For example the rawdata gives the average :math:`\overline{uu}`. To calculate :math:`\overline{u'u'}=\overline{uu}-\bar{u}\bar{u}`.

#. Stores data in a ``datastruct``
#. Plots the skin friction coefficient
#. Saves the plot to file


.. _inst_data:

Instantaneous Data
^^^^^^^^^^^^^^^^^^

The processing of the instantaneous data is carried out in a similar format to the averaged data. In this case, an instance of ``CHAPSim_Inst`` is created. Then using both this and the above CHAPSim_AVG_io instance, the ``CHAPSim_fluct_io`` class will be created and then a near-wall contour of the fluctuating streamwise velocity :math:`u'` will be created.

.. code-block:: python
   
   # Defining a time to be processed
   time = 122.0

   # Instantiating CHAPSim_Inst
   inst_data = cp.CHAPSim_Inst(time,path_to_folder=path)
   

   #Instantiating CHAPSim_fluct_io
   # Note the prototype can also use:
   #    CHAPSim_fluct_io(time,avg_data,path_to_folder=path)
   # or
   #    CHAPSIm_fluct_io(time,path_to_folder=path,time0=80)

   fluct_data = cp.CHAPSim_fluct_io(inst_data,avg_data)
   
   # Setting the y location, note that the default plane for
   # this method is the x-z plane and for this plane the default
   # normalisation is initial wall units.
   # 5 therefore indicates y^+ =5. The other plane just use the 
   # half channel height

   y_location = 5

   # plotting the streamwise component
   fig, ax = fluct_data.plot_contour('u',y_location)

   # Saving the plot
   fig.savefig(os.path.join(pictures_dir,'fluct_contour.png'))

The comments contain much useful information. Firstly, the ``CHAPSim_fluct_io`` class can take a variety of different prototypes. Also note that the contour plot takes the plane location in useful quantities. For the :math:`x-y` and :math:`z-y` plane this is the solver's default normalisation. For the :math:`x-z` plane plotted above there is also wall units (indicated with keyword argument ``y_mode='wall'``); displacement thickness (``y_mode=disp_thickness``); momentum thickness (``y_mode=mom_thickness``). 

The instantiation of ``CHAPSim_Inst`` extracts the instant rawdata at the time specified, then interpolates to 'centre' the data. The call to ``CHAPSim_fluct_io`` subtracts the average from the instantaneous data to get the fluctuating data.

Storing Data Locally
^^^^^^^^^^^^^^^^^^^^

As discussed in the :ref:`best_practice`, the ability to store data locally is very important particularly for high quality visualisations where lots of tinkering with figures is necessary. All the main classes can be saved and extracted from the HDF5 format with methods called ``save_hdf`` and ``from_hdf``. Additionally, classes that contain 3D data such as ``CHAPSim_Inst`` and ``CHAPSim_fluct_io`` contain attributes that can be saved as ``.vtk`` and ``.vts`` file which can subsequently be viewed in paraview.

An example of saving data as HDF5 files:

.. code-block:: python

   # saving the CHAPSim_AVG_io class from earlier
   avg_data.save_hdf("avg_data.h5","w")

   # saving the CHAPSim_Inst class from earlier
   inst_data.save_hdf("inst_data.h5","w")

Extracting the data:

.. code-block:: python
   
   # saving CHAPSim_AVG_io class
   avg_data_new = cp.CHAPSim_AVG_io.from_hdf("avg_data.h5")

   # saving CHAPSim_Inst class
   inst_data_new = cp.CHAPSim_AVG_io.from_hdf("inst_data.h5")

These files can also be saved to the same file by taking advantage of HDF5's POSIX structure. The default HDF5 key under which a class is stored in a file is its name. In the low-level code this is taken as ``self.__class__.__name__``. For the ``avg_data`` it would be ``CHAPSim_AVG_io``. This key can be modified by providing a keyword argument. This allows many instances of the same or different classes to be stored in the same file if necessary.

.. code-block:: python
   
   # Saving CHAPSim_AVG_io instance in the key avg_data
   avg_data.save_hdf("many_data.h5","w",key="avg_data")

   # Saving CHAPSim_Inst instance in the key inst_data
   inst_data.save_hdf("many_data.h5","a",key="inst_data")

To extract this data:

.. code-block:: python

   # saving CHAPSim_AVG_io class
   avg_data_new = cp.CHAPSim_AVG_io.from_hdf("many_data.h5",key="avg_data")

   # saving CHAPSim_Inst class
   inst_data_new = cp.CHAPSim_AVG_io.from_hdf("many_data.h5",key="inst_data")
   
Finer control can be acheived through knowledge of the ``h5py`` library which was used to implement this feature. The library documentation can be found `here <https://docs.h5py.org/en/stable/>`_


CHAPSim_post's ``rcParams``
---------------------------

This is not to be confused with Matplotlib's rcParams. Global parameters can be set in CHAPSim_post to aid the processing of results. The parameters which can be set like dictionary are:

* **TEST** - Limits the nmber of timesteps used in certain calculations. This is often used when testing code particularly when creating videos to ensure that the layout is correct before running the script on an HPC. Default ``False``.
* **autocorr_mode** - Method used to calculate the autocovariance. This code is loop-heavy therefore not ideal for native python. There are three modes: 0: numba accelerator; 1: Fortran with OpenMP; 2: Cython with OpenMP. Default ``2``. 
* **ForceMode** - If there is an error using either mode 1 or 2, should the code fallback to mode 0. Default False (fallback allowed).
* **Spacing** - Not implemented yet. This will enable code stored in datastructs to be reduced by 'skipping' values. So 2 would store every two values. Default 1.
* **dtype** - The float type which data is stored at. Sometimes using double precision may require substantial memory resources. This allows data to be downcast to single precision. Default double precision.
* **SymmetryAVG** - Whether to automatically perform symmetry averaging on channel flow data. Default True. 
* **gradient_method** - The method used to calculate gradients. Default numpy. Potentially more will be implemented which can use an arbitrary stencil and hence arbitrary order.
* **gradient_order** - Only relevant if more gradient methods are implemented.
* **AVG_Style** - The averaging style on labels. Some may prefer to use angled brackets although default is to use an overline.

Here is an exmaple of some commonly used rcParams

.. code-block:: python

   from CHAPSim_post import rcParams
   
   # setting storage type to single precision
   rcParams['dtype'] = 'f4'

   # activating test mode
   rcParams['TEST'] = True



The ``plot`` module and Figure cutomisation
-------------------------------------------

As this library is designed to be high-level, it is necessary to implement features to enable simple customisation of figures produced by the code. The code primarily uses the python module ``matplotlib`` to produce images. This is a well used python library, and it is recommended to be proficient at this module when visualising results. All plotting routines return ``fig`` and ``ax`` which allow a user to heavily modify the figures output from the code. 

Use Matplotlib's ``rcParams``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A very useful tool for modifying plots is Matplotlib's ``rcParams`` dictionary which allows heavy customisation. For example to globally change the label size:

.. code-block:: python

   import matplotlib as mpl

   mpl.rcParams['axes.labelsize'] = 12

CHAPSim_post also adds functionality to aid image customisation. This is mostly controlled through the ``CHAPSim_post.plot`` module. Some of the most useful tools are now discussed

The function ``update_prop_cycle`` takes keyword arguments and uses them to update Matplotlib's property cycler (can also be done though ``mpl.rcParams['axes.prop_cycle]`` although this is simpler). For example to change the line plot color order to blue, black, red, green:

.. code-block:: python

   import CHAPSim_post.plot as cplt

   cplt.update_prop_cycle(color='bkrg')

This can also be used to update the line style, width, marker, marker size etc. 

CHAPSim_post's Keyword Argument Passing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The plotting methods in CHAPSim_post are designed to provide inputs which allow customisation. Firstly, for all plotting routines additional keyword arguments are passed to the figure and axis creation routine which is based on Matplotlib's ``subplots`` function (It has the same keyword arguments as this routine). This is commonly used to give a figure size.

.. code-block:: python

   fig, ax = avg_data.plot_skin_friction(figsize=[10,5])

This is a repeat of the example earlier although with the figure size (in inches) specified. Plotting routines also usually contain argument designed to take a dictionary which is passed directly into the plotting routines. For line plots this is typically called ``line_kw``, for contour and surface plots this is typically named ``surf_kw``. In another example the line plot marker, color and line width are altered:

.. code-block:: python
   
   # creating dictionary with line properties
   line_dict = {"marker": "", "color": "k", "lw" : 1.5}

   #passing line properties to plotting method
   fig, ax = avg_data.plot_skin_friction(line_kw=line_dict)

The available keyword arguments depends on the the type of plot and the Matplotlib method the dictionary is passed to. For line plots, this `link <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot>`_ will give more information. For contour plots, this `link <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html#matplotlib.axes.Axes.pcolormesh>`_ will give more information.

.. _plot_derived:

``plot`` module's functions and derived classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CHAPSim_post does not use Matplotlib's classes directly. It uses derived classes that give additional functionality to the base classes which help users to postprocess turbulence data. Some of this functionality is given below. A more comprehensive list can be seen in the source code. There three derived classes:

#. **CHAPSimFigure:** Derived from class ``matplotlib.figure.Figure``

   * Ensures that an instance of ``AxesCHAPSim`` is created by default in its ``add_subplot`` method.

#. **AxesCHAPSim:** Derived from class ``matplotlib.axes.Axes``

   * New line plot method ``cplot``: Improved handling of the property cycler when using twinned twinned axes .
   * Methods for retrospective modification for example custom normalising data on axes.
   * Axes and label shifting: For example if you wish to change the point where :math:`x/\delta=0` in legends, titles and in plotted data.

#. **Axes3DCHAPSim:** Derived from class ``mpl_toolkits.mplot3d.Axes3D``

   * Adds axis shifting to the 3D plots
   * Adds method ``plot_isosurface``: This allows isosurfaces to be plotted in matplotlib. This is down using ``scikit-image`` modules matching cubes algorithm. 

.. code-block:: python

   # Example plotting the streamwise Reynolds stress
   # against y with modifications


   # plotting at these streamwise locations
   x_loc = [2,4,6,8]   

   # plotting the Reynolds stress
   fig, ax = avg_data.plot_Reynolds('uu',x_loc,wall_units=False,figsize=[8,6])

   # Shifting y coordinates on the x axis by 1
   # channel flow y coordinates run from -1 to 1
   # this way the wall starts at 0

   ax.shift_xaxis(1.0)

   # extracting the inner scalings from CHAPSim_AVG_io class
   u_tau_star, delta_v_star = avg_data.wall_unit_calc()

   # normalising the data on the x and y axis with initial wall units
   ax.normalise('y',u_tau_star[0]**2)
   ax.normalise('x',delta_v_star[0])
   
   # Changing the default x and y axis labels
   ax.set_ylabel(r"$\overline{u'u'}^{+0}$")
   ax.set_xlabel(r"$y^{+0}$")

This is a full example showing the power of Matplotlib's object-oriented library alonside the extensions added in CHAPSim_post. In this example, the Reynolds stress :math`\overline{u'u'}` is plotted against :math:`y`, then modified. Firstly ``shift_xaxis`` is used so the coordinate of the wall is :math:`0`. The inner scalings are then extracted from the CHAPSim_AVG_io class and the first value in those arrays corresponding to the inlet is used with the ``normalise`` method to scale the data in the plots to inlet wall units. The labels of the :math:`x` and :math:`y` axis are then modified.  

A range of other functions exist. The ``subplots`` function is a recreation of matplotlib's function of the same name which ensures CHAPSim_post's classes in instantiated. There is also helper function for creating videos ``create_general_video`` alongside other functions primarily but not exclusively designed to be used within the code

The ``utils`` module
--------------------

This is a very useful module containing additional *helper* functions which are used both at high and low level. The dual purpose of this module results in this module being separated into several smaller modules. The functions designed to be used high-level can all be accessed from the utils ``module`` directly.

.. code-block:: python

   from CHAPSim_post import utils

   #calculates the maximum time in the
   # 2_averaged_rawdata directory

   max_time = utils.max_time_calc(path)

A list of some other useful high-level functions:

* ``utils.coord_index_calc`` - converts coordinates into indices
* ``utils.y_coord_index_norm`` - converts :math:`y` coordinates with a range of normalisation into indices.
* ``utils.grad_calc`` - calculates gradients of arrays. There are also vector calculus operators such as ``utils.Scalar_laplacian_io`` although this has been superceeded by the functions in the ``DomainHandler`` class which accounts for coordinate system. 

Note that there is another module called ``CHAPSim_Tools`` however, this has been superseeded by ``utils`` and will be removed soon.

The ``dtypes`` module
---------------------

A key component of CHAPSim_post are the classes used to store data. These are found in the module ``CHAPSim_post.dtypes``. The classes are:

* **datastruct**
* **metastruct**
* **coordstruct** - dervied from datastruct
* **flowstruct3D** - dervied from datastruct

``datastruct``
^^^^^^^^^^^^^^

This is the core data storage class, used to store all sizeable data with the exception of full 3D data where the flowstruct3D class is used. A list of functionality:

* Allows data to be indexed at high-level
  
  * The physical time and the component can be used to return the a ``numpy`` array of the data. For example:

.. code-block:: python
   
   # extracts mean streamwise velocity
   u_mean = avg_data.flow_AVGDF[time,'u']
   
   # extracts streamwise Reynolds stress
   uu = avg_data.UU_tensorDF[time,'uu']

This means that data can be processed easily at high-level even if there isn't a method already implemented and makes extending the code straightforward. The indexing can also recover in some cases where the input time is wrong, an appriate warning message is displayed.

* Saving and extract data

  * The classes including CHAPSim_AVG_io use this class to do the heavy lifting when saving data to and from the HDF5 files with the methods ``to_hdf`` and ``from_hdf``.

.. code-block:: python
   
   #importing datastruct class from dtypes module
   from CHAPSim_post.dtypes import datastruct
   
   # saving mean flow data to file
   avg_data.flow_AVGDF.to_hdf('avg_data.h5','w',key='avg_data_data/flow_AVGDF')

   # reading mean flow data from file
   mean_data_new = datastruct.from_hdf('avg_data.h5',key='avg_data_data/flow_AVGDF')

* Binary, unary, and logical operations

  * Most operations typically carried out on arrays can be carried out directly at high level on datastructs both for scalars and other datastructs.

.. code-block:: python

   # multiply datastruct by 2
   multiplied_data = avg_data.flow_AVGDF * 2

``metastruct``
^^^^^^^^^^^^^^

This class is designed to store metadata which is extracted from a Results readdata.ini file. This is desined to work like a dictionary with the exception that it can be saved to the HDF5 format which allows this class, which is found as an attribute of the ``CHAPSim_meta`` class (which in turn is found in all main classes), to be stored as part of other classes.

.. code-block:: python
   
   # creating instance of CHAPSim_meta
   meta_data = cp.CHAPSim_meta(path)

   # extracting streamwise length from netastruct
   x_length = meta_data.metaDF['HX']

``coordstruct`` and ``flowstruct3D``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These classes are relatively recent additions to the code introduced to exploit the similarity of the visualisations from 3D classes. For example it was found instantaneous data and fluctuating data tended to be visualised in similar ways for example with contour plots, surface plots and isosurfaces. It was also found that other methods for example using the :math:`\lambda_2` criterion to identify streaks (part of ``CHAPSim_Inst``) would also use similar methods. Other modules such as ``POD``  for proper orthogonal decomposition (POD) and flow reconstruction used similar visualisation but with subtle differences in implementation which made creating new data structure useful. These methods contain additional functionality compared to the base datastruct class:

* **coordstruct** - Mainly facilitates functionality from the ``flowstruct3D`` class

  * Contains a ``DomainHandler`` attribute to ensure class' knowledge of the the geometry whether pipe of channel flow.
  * Coordinates can be transferred from cell-centred to staggered (to transferring data to vtkStructuredGrid)
  * Contains in-built ways of determining array indices from coordinates and chekcing contour planes.

* **flowstruct3D** - Visualisation functionality embedded into the class.

  * Contains in-built functionality for contour plots, surface plots, isosurface, and vector plots.
  * Can be exported to the ``.vtk`` or ``.vts`` format which can be viewed in Paraview.
  * Can dynamically work out more appropriate figure sizes unless figsize is passed to the method.

While this functionality is most typically used under-the-hood, it can also be useful high-level. Here is an example of exporting to VTK:

.. code-block:: python
   
   fluct_data.fluctDF.to_vtk("fluct_vtk.vts")

The ``.vts`` file extension is recommended as it uses the ``vtk.vtkXMLStructuredWriter`` class which results in files which take up much less space than the standard ``vtk.vtkStructuredWriter`` class.

