#!/bin/bash

VTK_INSTALL_ROOT=$HOME/data/SOFTWARE/VTK
VTK_CONDA_PATH=$HOME/SOFTWARE/anaconda3
VTK_CONDA_ENV=CHAPSim_post


###############################################################################
# test setup
source ./helper_funcs.sh

if [ -z ${VTK_INSTALL_ROOT+x} ]; then 
    echo -e "environment variable VTK_INSTALL_ROOT must be set"
    exit 1
else 
    echo -e "VTK installation root directory is: $VTK_INSTALL_ROOT"
fi


chk_dir $VTK_CONDA_PATH

echo -e "Path to conda is $VTK_CONDA_PATH"
echo -e "Using conda environment: $VTK_CONDA_ENV"

source $VTK_CONDA_PATH/etc/profile.d/conda.sh 2>/dev/null

test_return "Error in conda activation\n"

test_cmd conda "Check the anaconda root path, conda not found"
conda activate $VTK_CONDA_ENV
test_return "Error in conda activation\n"

echo -e "Using conda python with conda environment $VTK_CONDA_ENV" 
sleep 5



PYBIN=$(which python3)

test_cmd $PYBIN


mkdir -p $VTK_INSTALL_ROOT
rm -rf $VTK_INSTALL_ROOT/* 
#==========================================================

# Installing vtk dependencies
conda install -y -c conda-forge -c conda-forge conda-build libgcrypt mesalib ffmpeg matplotlib numpy 

test_return "Issue downloading vtk"

LIB_PATH=$VTK_CONDA_PATH/envs/$VTK_CONDA_ENV
# installing vtk

VERSION_MAIN=9.1
MINOR_VERSION=0
FULL_VERSION=$VERSION_MAIN.$MINOR_VERSION

VTK_MAIN_DIR=$VTK_INSTALL_ROOT/VTK-$FULL_VERSION
VTK_BUILD_PATH=$VTK_MAIN_DIR/build

cd $VTK_INSTALL_ROOT

#downloading vtk
wget https://www.vtk.org/files/release/$VERSION_MAIN/VTK-$FULL_VERSION.tar.gz

test_return "Issue downloading vtk"


tar xvf VTK-$FULL_VERSION.tar.gz

NSLOTS=$(grep -c ^processor /proc/cpuinfo)
CPU= $(( $NSLOTS / 2 ))

cd VTK-$FULL_VERSION
mkdir -p $VTK_BUILD_PATH && cd $VTK_BUILD_PATH

#configuring and building vtk
cmake CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    -G"Unix Makefiles"\
    -DFFMPEG_INCLUDE_DIRS=$LIB_PATH/include \
    -DFFMPEG_LIBRARIES=$LIB_PATH/lib \
    -DVTK_BUILD_TESTING=OFF \
    -DVTK_WHEEL_BUILD=ON \
    -DVTK_PYTHON_VERSION=3 \
    -DVTK_WRAP_PYTHON=ON \
    -DVTK_OPENGL_HAS_OSMESA=ON \
    -DVTK_OPENGL_HAS_EGL=OFF \
    -DOSMESA_INCLUDE_DIR=$LIB_PATH/include \
    -DOSMESA_LIBRARY=$LIB_PATH/lib/libOSMesa.so \
    -DVTK_USE_X=OFF \
    -DPython3_EXECUTABLE=$PYBIN \
    -DCMAKE_INSTALL_LIBDIR=${PWD} \
    -DCMAKE_LIBRARY_PATH=$LIB_PATH/lib \
    -S $VTK_MAIN_DIR -B $VTK_BUILD_PATH
    
test_return "Issue configuring install of VTK"

sleep 5

make

test_return "Issue building VTK"

#setting up python wheels
$PYBIN setup.py bdist_wheel

test_return "Issue setting up python setup.py"

#Installing, if using conda it will install it as a local conda package


mkdir -p recipes
cd recipes
echo -e "#!/bin/bash\n\n$PYBIN -m pip install $VTK_BUILD_PATH/dist/vtk-*.whl" > build.sh
echo -e "package:\n  name: vtk\n  version: $VERSION_MAIN" > meta.yaml

conda build .

test_return "Issue in conda build step"

conda install --use-local -y vtk

test_return "Issue with local install of vtk conda package"

conda deactivate



