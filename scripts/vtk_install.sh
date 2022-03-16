#!/bin/bash

# VTK_INSTALL_ROOT=${PWD}/test
# VTK_CONDA_PATH=$HOME/SOFTWARE/anaconda
# VTK_CONDA_ENV=CHAPSim_post

###############################################################################
# test setup
source ./helper_funcs.sh

if [ -z ${VTK_INSTALL_ROOT+x} ]; then 
    echo -e "environment variable VTK_INSTALL_ROOT must be set"
    exit 1
else 
    echo -e "VTK installation root directory is: $VTK_INSTALL_ROOT"
fi

if [ ! -z ${VTK_CONDA_ENV+x} ]; then

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

else
    echo -e "Using system python. Press ctrl - C"\
        ", if you want to use conda and "\
        "set the required environment variables" 
    sleep 5
fi

PYBIN=$(which python3)

test_cmd $PYBIN

mkdir -p $VTK_INSTALL_ROOT

#==========================================================

# checking cmake version
CMAKE_VERSION=$(cmake --version | gawk 'NR==1{print $3}')

CMAKE_MAJOR_VERSION=$(echo $CMAKE_VERSION | tr "." " " | gawk '{print $1}')
CMAKE_MINOR_VERSION=$(echo $CMAKE_VERSION | tr "." " " | gawk '{print $2}')

if [ $CMAKE_MAJOR_VERSION -lt 3 ]; then
    echo "cmake version required is 3.18 or higher"
fi

if [ $CMAKE_MINOR_VERSION -lt 12 ]; then
    echo "cmake version required is 3.18 or higher"
    exit 1
fi

# Installing vtk dependencies

VTK_DEPS=$VTK_INSTALL_ROOT/deps

if [ -d $VTK_DEPS ]; then
    rm -rf $VTK_DEPS
fi

mkdir -p $VTK_DEPS


cd $VTK_DEPS
LIB_PATH=$VTK_DEPS/build/install

git clone --recursive -b v5.10.0-RC1 https://gitlab.kitware.com/paraview/paraview-superbuild.git
mkdir -p build && cd build
cmake -GNinja -DENABLE_osmesa=ON \
                -DENABLE_mpi=OFF \
                -DENABLE_hdf5=ON \
                -DENABLE_ffmpeg=ON \
                -DENABLE_paraview=OFF \
                ../paraview-superbuild

test_return "Issue configuring install of VTK dependencies"

sleep 5

ninja -v

test_return "Issue building VTK dependencies"

sleep 5
# installing vtk

VTK_BUILD_PATH=$VTK_INSTALL_ROOT/build

cd $VTK_INSTALL_ROOT

#downloading vtk
wget https://www.vtk.org/files/release/9.1/VTK-9.1.0.tar.gz

test_return "Issue downloading vtk"


tar xvf VTK-*.tar.gz


cd VTK-*
mkdir $VTK_BUILD_PATH && cd $VTK_BUILD_PATH

#configuring and building vtk
cmake -GNinja\
    -DFFMPEG_ROOT=$LIB_PATH \
    -DVTK_BUILD_TESTING=OFF \
    -DVTK_WHEEL_BUILD=ON \
    -DVTK_PYTHON_VERSION=3 \
    -DVTK_WRAP_PYTHON=ON \
    -DVTK_OPENGL_HAS_OSMESA=ON \
    -DVTK_OPENGL_HAS_EGL=OFF \
    -DOSMESA_ROOT=$LIB_PATH \
    -DVTK_USE_X=OFF \
    -DPython3_EXECUTABLE=$PYBIN \
    -DCMAKE_INSTALL_LIBDIR=${PWD} \
    -S ../ -B .
    
test_return "Issue configuring install of VTK"

sleep 5

ninja -v

test_return "Issue building VTK"

#setting up python wheels
$PYBIN setup.py bdist_wheel

test_return

#Installing, if using conda it will install it as a local conda package

if [ -z ${USE_CONDA+x} ]; then
    $PYBIN -m pip install dist/vtk-*.whl
    test_return "Issue creating vtk python wheel"
else
    mkdir -p recipes
    cd recipes
    echo -e "\#!/bin/bash\n\n$PYBIN -m pip install $VTK_BUILD_PATH/dist/vtk-*.whl" > build.sh
    echo -e "package:\n\tname: vtk\n\tversion: 9.0" > meta.yaml

    conda build .

    test_return "Issue in conda build step"

    conda install --use-local vtk

    test_return "Issue with local install of vtk conda package"

    conda deactivate

fi


