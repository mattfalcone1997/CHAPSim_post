#!/bin/bash

VTK_INSTALL_ROOT=$HOME/SOFTWARE/VTK_install
VTK_DEPS=$VTK_INSTALL_ROOT/deps

CMAKE_VERSION=${cmake --version | gawk 'NR==1{print $3}'}



