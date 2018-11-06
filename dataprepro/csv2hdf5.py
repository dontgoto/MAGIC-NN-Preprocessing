### convert .csv to .hdf5. For documentation, see equivalent MC file
from functools import partial
from threading import Thread
from glob import glob
import sys
import os
from os import path
from time import sleep
import random as rnd
import logging

import h5py
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
tqdm = partial(tqdm, mininterval=2.0)

### function to read .csv files and save them as .hdf5 dataset files

def _try_closing(hdf5_file):
    sleep(3)
    try:
        hdf5_file.close() #close the file
    except UnboundLocalError:
        logger.warning("File didn't even get created")
    sleep(1)

def create_hdf5_from_csv(infile, outfile, dtype=float, retries=0):
    input_dataset = genfromtxt(infile, delimiter=',').astype(dtype) #read file
    if path.isfile(outfile):
        os.remove(outfile)
    try:
        hdf5_file = h5py.File(outfile, "w") #create .hdf5-file
    except OSError:
        del input_dataset
        logger.warning("creating hdf5 failed, doing recursion")
        if retries > 6:
            logger.error("\n \n***** Maximum number of retries reached. *****")
            logger.error("infile", infile)
            logger.error("outfile", outfile, "\n\n")
            # if this gets reraised root2csv or remove_events could be called again to reprocesses the failed .csv file, but honestly that is way to much work
            raise
        else:
            _try_closing(hdf5_file)
            retries += 1
            create_hdf5_from_csv(infile, outfile, dtype=dtype, retries=retries)
    else:
        hdf5_file.create_dataset("data", data=input_dataset) #write dataset to .hdf5-file
        del input_dataset
        hdf5_file.close() #close the file
    return outfile

def create_hdf5_from_dataset(input_dataset, outfile, retries=0):
    if path.isfile(outfile):
        os.remove(outfile)
    try:
        hdf5_file = h5py.File(outfile, "w") #create .hdf5-file
    except OSError:
        if retries > 6:
            logger.error("\n \n***** Maximum number of retries reached. *****")
            logger.error("outfile", outfile, "\n\n")
            # if this gets reraised root2csv or remove_events could be called again to reprocesses the failed .csv file, but honestly that is way to much work
            raise
        else:
            _try_closing(hdf5_file)
            retries += 1
            create_hdf5_from_dataset(input_dataset, outfile, retries=retries)
    else:
        hdf5_file.create_dataset("data", data=input_dataset) #write dataset to .hdf5-file
        del input_dataset
        hdf5_file.close() #close the file
    return outfile

def create_hdf5_files(csvFilenames, dtype=float):
    '''takes the csvfiles and converts them to .hdf5 files'''
    # outfiles = [csvFilename.split('.')[0]+'.hdf5' for csvFilename in csvFilenames]
    if isinstance(csvFilenames, str):
        csvFilenames = list(csvFilenames)
    outfiles = []
    infileiter = tqdm(csvFilenames)
    # for i, infile in enumerate(tqdm(csvFilenames)):
    for i, infile in enumerate(infileiter):
        infileiter.set_description(f"Converting {path.basename(infile)} to hdf5")
        outfile = infile.replace(".csv", ".hdf5")
        create_hdf5_from_csv(infile, outfile, dtype=dtype)
        outfiles.append(outfile)
    return outfiles

def create_hdf5(infileName):
    cth.create_hdf5_from_csv(infileName, newOutfileName, dtype="float16") #load .csv, save as .hdf5

### function to read .hdf5-files

def read_hdf5_dataset(filename):
    hdf5_file = h5py.File(filename, 'r') #open file
    data = hdf5_file['data'] #extract data
    hdf5_file.close()
    del hdf5_file
    return data #return data

### function to create .hdf5-file from dataset

def check_for_equality(*args):
    """Checks if both telescopes have the same number of files, takes 1 list of lists or 2 lists as arguments"""
    if len(args) == 1:
        csv1 = args[0][0]
        csv2 = args[0][1]
    elif len(args) == 2:
        csv1 = args[0]
        csv2 = args[1]
    else:
        raise NotImplementedError("idk how to handle more than two args for this func")
    assert len(csv1) == len(csv2), "Number of files for different telescopes not equal, aborting."
    if '_M1_' in csv1[0]:
        for string in csv1:
            string.replace('_M1_', '_M2_')
    else:
        for string in csv1:
            csv1.replace('_M2_', '_M1_')
    if not  int(np.sum(not np.equal(csv1, csv2))) == 0:
        raise NameError("Input telescope files are not the same, exiting.")
    logger.info('Telescope files equal, continuing.')

def glob_and_check(globPath):
    globbedList = glob(globPath)
    if not globbedList:
        errormsg = f"\n \n Glob: {globPath} \n No infiles, wrong glob? Aborting."
        if not path.exists(path.dirname(globPath)):
            errormsg += f"\n The globdir also doesnt exist."
        else:
            errormsg += f"\n But the dir {path.dirname(globPath)} exists.\n"
        raise FileNotFoundError(errormsg)
    return sorted(globbedList)

def _get_time_str(filename):
    """Returns the timestamp in the filename"""
    basename = path.basename(filename)
    splitters = np.array(["_Y_", "_I_"])
    isin = [spl in basename for spl in splitters]
    splitter = splitters[np.argwhere(isin)][0][0]
    head, _, _ = basename.partition(splitter)
    # filename example:
    # 20160929_M1_05057144.004_Y_CrabNebula-W0.40+215.root
    # simulated data example:
    # GA_M1_za05to35_8_1737993_Y_wr.root
    time = head.split("_")[-1]
    return time

def merge_time_strs(filename1, filename2):
    """Takes the timestamp in from the first file and merges it with that of the second."""
    print(filename1)
    print(filename2)
    starttime = _get_time_str(path.basename(filename1))
    endtime = _get_time_str(path.basename(filename2))
    mergedtime = starttime+"_TO_"+endtime
    basename = path.basename(filename1)
    assert ("_Y_" in basename) or ("_I_" in basename), ("_Y_ or _I_ not in basename, merging timestrs undefined, aborting. "
                                                        f"basename: {basename} filename1: {filename1} filename2: {filename2}")
    # filename example:
    # 20160929_M1_05057144.004_Y_CrabNebula-W0.40+215.root
    # simulated data example:
    # GA_M1_za05to35_8_1737993_Y_wr.root
    head, ysep, tail = basename.partition("_Y_")
    if "GA_" in head:
        newhead = head.split("_")[0] + "_"
        msep = "_".join(head.split("_")[1:3]) + "_"
    else:
        if "M1" in head:
            newhead, msep, _ = head.partition("_M1_")
        else:
            newhead, msep, _ = head.partition("_M2_")
    newfilename = newhead+msep+mergedtime+ysep+tail
    return newfilename

def get_shape(hdfgroup):
    if isinstance(hdfgroup, str):
        hdfgroup = h5py.File(hdfgroup, "r")
    keys = list(hdfgroup.keys())
    # from IPython import embed; embed()
    rowlen, collen = hdfgroup[keys[0]].shape
    return [rowlen, collen]

### function to merge multiple smaller hdf5 files
def merge_hdfbatch(i, hdfBatch, outpath):
    temp = h5py.File(hdfBatch[0], 'r')["data"]
    shapes = np.array([get_shape(f) for f in hdfBatch])
    dat1 = np.ones((shapes[::,0].sum(), shapes[0,1]), dtype="float32")
    startrow = 0
    for i, fname in enumerate(tqdm(hdfBatch)):
        dat1[startrow:startrow+shapes[i][0],::] = h5py.File(fname, "r")["data"]
        startrow += shapes[i][0]

    oldsize = np.sum([s[0]*s[1] for s in shapes])
    newsize = dat1.shape[0]*dat1.shape[1]
    assert oldsize == newsize, "Sizes pre and post merging dont match up anymore, aborting."

    if ("_Y_" in hdfBatch[0]) or ("_I_" in hdfBatch[0]):
        outfile = path.join(outpath, merge_time_strs(hdfBatch[0], hdfBatch[-1]))
    else:
        outfile = path.join(outpath, f"data{i}.hdf5")
    logger.info("outfile", outfile)
    create_hdf5_from_dataset(input_dataset=dat1,
                             outfile=outfile)
    del dat1
    return outfile

def merge_all_hdf5_files(hdfFilenames, outpath, outFilesize=2.5*10**9):
    '''outFilesize : float
           maximum size of the resulting outfiles in bytes'''

    inFilesSize = np.sum([os.path.getsize(filename) for filename in hdfFilenames])
    noOfOutfiles = inFilesSize // outFilesize
    outfiles = []
    if noOfOutfiles:
        hdfFilenames = np.array_split(hdfFilenames, noOfOutfiles)
    if noOfOutfiles == 0:
        outfile = merge_hdfbatch(1, hdfFilenames, outpath)
        outfiles.append(outfile)
    else:
        # for i, hdfBatch in enumerate(tqdm(hdfFilenames)):
        for i, hdfBatch in enumerate(hdfFilenames):
            outfile = merge_hdfbatch(i, hdfBatch, outpath)
            outfiles.append(outfile)

        # outfiles = Parallel(n_jobs=2)\
            # (delayed(merge_hdfbatch)(i, hdfBatch, outpath) for i, hdfBatch in enumerate(hdfFilenames))
    return outfiles
