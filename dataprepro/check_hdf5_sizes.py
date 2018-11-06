import h5py as h5
from glob import glob
from csv2hdf5 import glob_and_check
from os import path
rootdir = "/net/big-tank/POOL/users/phoffmann/masterthesis/realdata/crab/good_st0307/mc/test/"
rootdir = "/net/big-tank/POOL/users/phoffmann/masterthesis/realdata/crab/good_st0307/root/"
f = "*737990*.hdf5"
f = "*57148*.hdf5"
f = "*05057441*"

def get_file_shape(subglob):
    fname = glob_and_check(path.join(rootdir, subglob))[0]
    h5file = h5.File(fname)
    shape = h5file["data"].shape
    return shape

def get_txt_shape(subglob):
    pass

normalized = get_file_shape("csv/"+f)
print(normalized)
normalized = get_file_shape("telmerged/"+f)
print(normalized)
normalized = get_file_shape("telmerged/"+f)
print(normalized)
normalized = get_file_shape("normalized/"+f)
print(normalized)
