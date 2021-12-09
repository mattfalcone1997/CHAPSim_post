.. _dtypes:

===========================
The ``dtypes`` Module Guide
===========================

At the heart of CHAPSim_post is the ``dtypes`` module this contains the main data storage, access and visualisation tools for the package. This guide discusses the Key features of these classes and how to use them. This guide contains the following sections:

1. Core ``datastruct`` functionality
2. Coordinate and Domain Handling
3. ``FlowStructND`` and its derived classes
4. ``dtypes`` and the Visualisation Toolkit (VTK)

Core ``datastruct`` functionality
---------------------------------

The ``datastruct`` class contains some core features that are utilised by dervied classes such as the FlowStructND classes and coordstruct classes:

* Storing data with various components and times.
* Accessing data using handy string based keys.
* Saving data to and reading data from the HDF5 file format.
* Basic arithmetic operations with scalars, numpy arrays and other ``datastruct`` instances.

The examples presented here are somewhat contrived the examples in the examples folder will show this functionality used on real data.
  
A basic introduction to the ``datastruct``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This introduction will showcase the basics of how the datastruct operates. The following code snippets will give an indication of the behaviour of the class with more detail given below.

There are two main ways to initialise the ``datastruct`` 

* Numpy array and an index
  
.. code-block:: python
    import CHAPSim_post.dtypes as cd
    import numpy as np

    # creates a 4 x 10 x 20 x 30 array
    data = np.ones((4,10,20,30))
    index = ['a','b','c','d']

    dstruct = cd.datastruct(data,index=index)

* A dictionary

.. code-block:: python
    import CHAPSim_post.dtypes as cd
    import numpy as np

    # creates a 4 x 10 x 20 x 30 array
    data = np.ones((10,20,30))
    index = ['a','b','c','d']
    data_dict = {'a': data,
                 'b': data,
                 'c': data,
                 'd': data}

    dstruct = cd.datastruct(data)

To access the data stored under these keys, indexing can be used. The following code can be used to print the numpy array at key ``'a'``.

.. code-block:: python
    print(dstruct['a'])

The data from this datastruct can be saved to and read from the HDF5 file format using

.. code-block:: python
    # Saving data
    file_name = 'test_file.h5'
    dstruct.save_hdf(file_name,'w')

    #reading data
    dstruct2 = cd.datastruct.from_hdf(file_name)

Arithmetic operations can be performed with: 

* scalars:

.. code-block::python
    print(dstruct*2.0) # Would print the dstruct with all elements multiplied by two

* numpy arrays obeying normal Numpy broadcasting rules. This would only work if the data under all keys can perform a valid broadcast.
  
.. code-block::python
    # Multiplies along the last array which also has size 30
    array = np.arange(30)
    print(dstruct*array) 

* other ``datastruct`` instances. Note that both datastructs must have the same keys with the same shapes 

.. code-block::python
    data = 2.0 * np.ones((4,10,20,30))
    index = ['a','b','c','d']

    dstruct_3 = cd.datastruct(data,index=index)

    print(dstruct*dstruct_3)

Coordinate and Domain Handling
------------------------------

A key component of the FlowStructND-based classes that is required is knowledge of the geometry of the data. For example whether it is a pipe, a channel, or a Boundary layer. It also needs to have the coordinate data. There are three main classes for this

* ``coordstruct``: A class derived from the ``datastruct`` which contains the coordinate arrays
* ``GeomHandler``: Contains information regarding the domain. E.g. what the geometry is. It is common replaced with a ``DomainHandler`` which also includes gradient calculation methods.
* ``AxisData``: A class which contains a ``coordstruct`` for the centered coordinate data, a ``coordstruct`` for the staggered coordinate data, and a ``GeomHandler`` class. This class is often replaced with the ``coorddata`` class which also has the ability to extract coordinate information from CHAPSim's results folder.


``FlowStructND`` and its derived classes
----------------------------------------

The key functionality added by this class is combining the geometry and coordinate information with the ``datastruct`` features to enable the classes to plot the data contained within them. While the ``FlowStructND`` contains most of the functionality, the dervied classes will be encountered more often and typically more useful:

* FlowStruct3D
* FlowStruct2D
* FlowStruct1D
* FlowStruct1D_time

Some examples will now be shown using the FlowStruct3D and FlowStruct2D plotting data.

* Using a FlowStruct3D to plot a contour plot.
* Using a FlowStruct2D to plot some lines.

``dtypes`` and the Visualisation Toolkit (VTK)
----------------------------------------------

The FlowStruct2D and FlowStruct3D classes can be easily output to Pyvista ``StructuredGrid``, which are derived from the ``vtk.vtkStructuredGrid`` class. This is achieved via the ``VTKStruct3D`` and ``VTKStruct2D`` classes. Normally, the staggered data is used for the points with the data and the FlowStruct's arrays passed as cell_data. If there is no staggered data, the centered data is used as point data with the arrays passed as point data.


.. code-block::python
    # fstruct is a FlowStruct3D instance:
    vtk3D = fstruct.VTK # vtk3D is a VTKStruct3D instance

If the VTKStruct is indexed it will output to a ``StructuredGrid``. If ``fstruct`` has a key ``'a'``:

.. code-block::python
    structuregrid = vtk3D['a']

If valid functions from the Structured grid can be passed to the VTKStruct (which uses the ``__getattr__`` special method).

.. code-block::python
    point_data = vtk3D.cell_data_to_point_data()