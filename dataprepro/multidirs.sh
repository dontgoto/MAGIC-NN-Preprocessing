#!/bin/bash

#datadir="/fhgfs/users/phoffman/masterthesis/data/dnndata/root/"
datadir="/net/big-tank/POOL/users/phoffman/masterthesis/realdata/crab/"
glob="2016_11_*"
declare -a directories=($(ls $datadir))
normfile="/home/phoffman/masterthesis/magic_nn_preprocessing/data/norm.txt"
processPY="/home/phoffman/masterthesis/magic_nn_preprocessing/data/process.py"
echo $directories

#for directory in "${directories[@]}"
#do
   #echo $directory
   #python3.6 $processPY --rootdir=$datadir$directory --normfile=$normfile --csv2hdf5="False"
#done
