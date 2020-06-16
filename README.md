# CHAPSim_post_v0.1
Project providing postprocessing facilities to the CHAPSim fluids solver

## Setup
The simplest way of using this library is to use conda (miniconda or anaconda). To setup this module run the setup_CHAPSIm_post in either this directory or in the scripts directory. This script creates a new conda environment with the required packages called CHAPSim_post. It also adds the bin directory to both the PYTHONPATH environment variable and to the path of the CHAPSim_post environment.

## Capabilities
This should be can be run on local linux machines and HPCs on shared memory parallel environments. The CHAPSim_parallel module is developed for distributed memory using mpi4py but is deprecated and hasn't been developed or maintained for a while so it is not advised to used.
