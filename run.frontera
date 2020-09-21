#!/bin/bash

#DOMAIN INFORMATION
export NUMTHE=360
export NUMRHO=256
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
export THEFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/ADS2_theta.bin
export SINFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/ADS2_sinogram.bin
export OUTFILE=/gpfs/alpine/scratch/merth/csc362/recon_ADS2.bin

jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./memxct
