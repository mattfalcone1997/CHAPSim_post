.. _best_practice:

===================
Best Practice guide
===================

CHAPSim_post is designed to be used on both personal computers as  well as High Performance Computers (HPCs). Due to CHAPSim being a DNS code, most data to be processed will be found on HPCs which can bring some challenges to the processing of data.

.. _access_data:

Accessing data
^^^^^^^^^^^^^^

It is envisaged that most processing will be done of workstations and personal computers therefore it is necessary to be able to access the data remotely. The most advised may of doing this is using the SSHFS command to mount an HPCs remote filesystem to your local computer which can be done as follows:

.. code-block:: bash

  sshfs user-name@HPC_host.ac.uk:/remote/file/location/ /local/file/location/

Storing data
^^^^^^^^^^^^

Often constantly accessing data over an SSH connection can be time consuming and hence it is recommended to store processed data on the local machine. Information on how to do this is in the User Guide.

Processing data on HPCs
^^^^^^^^^^^^^^^^^^^^^^^

Certain visualisations and statistics can require substantial quantities of instantaneous data this can include quadrant analysis, autocorrelations and the processing of videos. It is recommended that these are conducted on HPCs to reduce file access times which is often a substantial proportion of the overall processing time. Information on installing the code on HPCs is present in the user guides