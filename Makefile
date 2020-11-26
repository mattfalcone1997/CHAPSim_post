
FC=gfortran
SRC_F2PY=src/autocorr_parallel.f90
BINS:=bin/$(SRC_F2PY:%.f90=%)
LOC:=$(shell pwd)

.PHONY : clean  scripts
clean : 
	rm -r bin/*
scripts : 
	cp -r src/*.py bin

source : 
	
	cd bin; python3 -m numpy.f2py -c  --opt="-O3 -ftree-vectorize -fopenmp" --fcompiler=$(FC) -m f_autocorr_parallel ${LOC}/src/autocorr_parallel.f90


all : clean source scripts
