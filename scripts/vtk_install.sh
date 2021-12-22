#!/bin/bash

VTK_INSTALL_ROOT=$HOME/SOFTWARE/VTK_install
VTK_DEPS=$VTK_INSTALL_ROOT/deps

CMAKE_VERSION=$(cmake --version | gawk 'NR==1{print $3}')

CMAKE_MAJOR_VERSION=$(echo $CMAKE_VERSION | tr "." " " | gawk '{print $1}')
CMAKE_MINOR_VERSION=$(echo $CMAKE_VERSION | tr "." " " | gawk '{print $2}')

if [ $CMAKE_MAJOR_VERSION -lt 3 ]; then
    echo "cmake version required is 3.12 or higher"
    exit 1
fi

if [ $CMAKE_MINOR_VERSION -lt 12 ]; then
    echo "cmake version required is 3.12 or higher"
    exit 1
fi

