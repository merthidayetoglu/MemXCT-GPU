#!/bin/bash

#DOMAIN INFORMATION
export NUMTHE=1500
export NUMRHO=1024
export PIXSIZE=1
#SOLVER DATA
export NUMITER=24
#TILE SIZE (MUST BE POWER OF TWO)
export SPATSIZE=128
export SPECSIZE=128
#BLOCK SIZE
export PROJBLOCK=128
export BACKBLOCK=128
#BUFFER SIZE
export PROJBUFF=8 #KB
export BACKBUFF=8 #KB
#I/O FILES
export THEFILE=~/MemXCT_datasets/ADS3_theta.bin
export SINFILE=~/MemXCT_datasets/ADS3_sinogram.bin
export OUTFILE=~/MemXCT_datasets/recon_ADS3.bin

export OMP_NUM_THREADS=1

mpirun -np 4 ./memxct
