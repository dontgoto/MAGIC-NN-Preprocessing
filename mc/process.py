import os
from joblib import Parallel, delayed
from tqdm import tqdm
from mergetelescopes import append_and_merge
from csv2hdf5 import create_hdf5
from dataprepro import csv2hdf5 as cth
from normalisation import pixelwise_normalisation
from normalisation import camerawise_normalisation

dir1 = "test/"
dir2 = "train/"

rootDir = "/net/big-tank/POOL/users/phoffmann/masterthesis/realdata/crab/good_st0307/mc/"
# rootDir = "/net/big-tank/POOL/users/phoffmann/masterthesis/realdata/crab/35to50/mc/"
directories = ["csv/filtered/", "hdf5/filtered/", "hdf5_32/filtered/", "hdfmerged/filtered/",
               "telmerged/filtered/", "normalized/filtered/"]
testdirs = [dir1+directory for directory in directories]
traindirs = [dir2+directory for directory in directories]
for directory in testdirs:
    os.makedirs(rootDir+directory, exist_ok=True)
for directory in traindirs:
    os.makedirs(rootDir+directory, exist_ok=True)


csvFilenames11 = cth.glob_and_check(rootDir+testdirs[0]+'*GA*M1*.csv')
csvFilenames12 = cth.glob_and_check(rootDir+traindirs[0]+'*GA*M1*.csv')
csvFilenames21 = cth.glob_and_check(rootDir+testdirs[0]+'*GA*M2*.csv')
csvFilenames22 = cth.glob_and_check(rootDir+traindirs[0]+'*GA*M2*.csv')
cth.check_for_equality(csvFilenames11, csvFilenames21)
cth.check_for_equality(csvFilenames12, csvFilenames22)
all_csvFilenames = [csvFilenames11, csvFilenames12, csvFilenames21, csvFilenames22]

# for csvFilenames in all_csvFilenames:
    # Parallel(n_jobs=28)(delayed(create_hdf5)(infileName, infileNumber)
                        # for infileNumber, infileName in enumerate(csvFilenames))
print("csv2hdf5 done.")

hdfFilenames11 = cth.glob_and_check(rootDir+testdirs[1]+'*_M1_*.hdf5')
hdfFilenames12 = cth.glob_and_check(rootDir+traindirs[1]+'*_M1_*.hdf5')
hdfFilenames21 = cth.glob_and_check(rootDir+testdirs[1]+'*_M2_*.hdf5')
hdfFilenames22 = cth.glob_and_check(rootDir+traindirs[1]+'*_M2_*.hdf5')

# no merging b/c it potentially fucks up the energy imputation, everything else is parallelized so whatever
# print("merging hdf5 files.")
# cth.merge_all_hdf5_files(hdfFilenames11, rootDir+testdirs[3])
# cth.merge_all_hdf5_files(hdfFilenames12, rootDir+traindirs[3])
# cth.merge_all_hdf5_files(hdfFilenames21, rootDir+testdirs[3])
# cth.merge_all_hdf5_files(hdfFilenames22, rootDir+traindirs[3])
# print("HDF5 files merged.")

files11 = cth.glob_and_check(rootDir+testdirs[1]+'*_M1_*.hdf5')
files12 = cth.glob_and_check(rootDir+traindirs[1]+'*_M1_*.hdf5')
files21 = cth.glob_and_check(rootDir+testdirs[1]+'*_M2_*.hdf5')
files22 = cth.glob_and_check(rootDir+traindirs[1]+'*_M2_*.hdf5')
files = [[files11, files21], [files12, files22]]
m1files = files11 + files12
m2files = files21 + files22

print("merging telescope files.")
# Parallel(n_jobs=30)(delayed(append_and_merge)(fs[0], fs[1], outpath=rootDir+odir) for fs, odir in zip(files, [testdirs[-2], traindirs[-2]]))
Parallel(n_jobs=30)(delayed(append_and_merge)([m1], [m2], outpath=rootDir+testdirs[-2]) for m1, m2 in zip(m1files, m2files))
Parallel(n_jobs=30)(delayed(append_and_merge)([m1], [m2], outpath=rootDir+traindirs[-2]) for m1, m2 in zip(m1files, m2files))
print("Telescope files merged.")


infiles1 = cth.glob_and_check(rootDir+'test/telmerged/*.hdf5')
infiles2 = cth.glob_and_check(rootDir+'train/telmerged/*.hdf5')
for infile1, infile2 in zip(infiles1, infiles2):
    print("normalizing \n {infile1} \n {infile2}")
    pixelwise_normalisation(infile1)
    camerawise_normalisation(infile1)
    pixelwise_normalisation(infile2)
    camerawise_normalisation(infile2)
print("Files normalized.")
