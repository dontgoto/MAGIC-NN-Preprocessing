from functools import partial
from subprocess import call
from os import path
import os
import logging

from csv2hdf5 import glob_and_check
import click
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from csv2hdf5 import glob_and_check

tqdm = partial(tqdm, mininterval=300.0)
# CERNROOTVER = "/net/nfshome/home/phoffmann/.local/root/root-6.08.00/bin/root"
CERNROOTVER = "/net/big-tank/POOL/users/phoffmann/software/root-6.08.00/bin/root"
MERGECOLDIR = "mergecols/"
MERGECOLENDING = "_mergecols.csv"
ROOTCONVERTER_NORMAL = "globRoot2csv.C"
ROOTCONVERTER_STAR = "star2csv.C"
MERGEDCSVPREFIX = "merged_"
SCRIPTDIR = path.join(path.dirname(__file__), "")
EGREP = " | egrep -v"
EGREPSTR = '"no dictionary for class"'

logger = logging.getLogger(__name__)


def setup_dirs(rootdir):
    dirs = ["csv/", MERGECOLDIR]
    for directory in dirs:
        os.makedirs(path.join(rootdir, directory), exist_ok=True)
    logger.info("Directory setup done.")

def get_new_globstr(globstr):
    if "_mergecols" in globstr:
        newstr = globstr.replace(".root", ".csv")
    else:
        newstr = globstr.replace(".root", MERGECOLENDING)
    return newstr

def check_telescope_files(rootdir=None, globstr1=None, globstr2=None, replacer=("_M1_", "_M2_"), force=False):
    """Checks whether the files under globstr1 and globstr2 are equal (the filenames should all
	   be identical once the replacer is swapped out)."""
    # needs implementation If force is set to False it aborts if the files are not equal.
	# If force is set to True it simply returns the union of filenames
    if rootdir is not None:
        globstr1 = path.join(rootdir, globstr1)
        globstr2 = path.join(rootdir, globstr2)
    if isinstance(globstr1, str):
        files1 = glob_and_check(globstr1)
    else:
        files1 = globstr1
    if isinstance(globstr2, str):
        files2 = glob_and_check(globstr2)
    else:
        files2 = globstr2

    bnames1 = set([path.basename(fname.replace(replacer[0], replacer[1])) for fname in files1])
    bnames2 = set([path.basename(fname.replace(replacer[0], replacer[1])) for fname in files2])
    assert bnames1 == bnames2, \
           (f"Telescope files not equal, set difference: len files1 {len(files1)} len files2 {len(files2)}\n\n"
            f"dirs: {path.dirname(files1[0])}, {bnames1.symmetric_difference(bnames2)}")

def convert_to_csv(rootdir, globstr1, globstr2, mergecolsonly, parallelprocessing, star, njobs=1):
    """Converts the rootfiles that are found in the globstrings to csv and puts them in MERGECOLDIR.
	   If there are multiple csvfiles it merges them with prefix MERGEDCSVPREFIX.
       Returns the name of the merged csv file."""
    setup_dirs(rootdir)
    #maybe implement parellel processing here so the filenames in globstring get split into subarrays
	# and each processed seperately
    # the arguments can be found in globRoot2.csv the third one should be whether to glob or not
    # the 1 is for doGlob
    if (star is False) and (mergecolsonly is False):
        check_telescope_files(rootdir, globstr1, globstr2)
    elif (star is True) and (mergecolsonly is False):
        check_telescope_files(rootdir, globstr1, globstr2, replacer=("_Y_", "_I_"))

    call_c_converter(rootdir, globstr1, globstr2, mergecolsonly, parallelprocessing, star, njobs)

    if mergecolsonly is False:
        outFilenames = path.dirname(globstr1)
    # used to be else, idk yet if that works with star mode
    elif star is False:
        newdir = path.join(rootdir, MERGECOLDIR)
        newglobstr1 = get_new_globstr(globstr1)
        newglobstr2 = get_new_globstr(globstr2)
        check_telescope_files(newdir, newglobstr1, newglobstr2)
        mergecolFilenames1 = glob_and_check(path.join(newdir, newglobstr1))
        mergecolFilenames2 = glob_and_check(path.join(newdir, newglobstr2))
        mergecolFilenames1.extend(mergecolFilenames2)
        outFilenames = mergecolFilenames1
    return list(set(outFilenames))

def fcall(rootdir, str1, str2, mergecolsonly, parallelprocessing, star):
    if parallelprocessing:
        # deactivate globbing in globRoot2csv.C if perallelProcessing is set to True
        functionargs = f'(\"{rootdir}\", \"{str1}\", \"{str2}\", {int(0)}, {int(mergecolsonly)})'
    else:
        functionargs = f'(\"{rootdir}\", \"{str1}\", \"{str2}\", {int(1)}, {int(mergecolsonly)})'
    if star is False:
        rootconverter = ROOTCONVERTER_NORMAL
    else:
        rootconverter = ROOTCONVERTER_STAR
        assert "_I_" in str1, "The first file needs to be a starfile in starmode"
    functioncall = SCRIPTDIR + rootconverter + functionargs
    fcalllist = [CERNROOTVER, "-b", "-q", "-l", functioncall]
    call(fcalllist)

def call_c_converter(rootdir, globstr1, globstr2, mergecolsonly, parallelprocessing, star, njobs=1):
    """Function that encapsulates the call to the root macro that converts the root files to csv."""
    if isinstance(globstr1, str):
        filenames1 = glob_and_check(path.join(rootdir, globstr1))
        filenames2 = glob_and_check(path.join(rootdir, globstr2))
    elif isinstance(globstr1, list):
        filenames1 = globstr1
        filenames2 = globstr2

    if parallelprocessing is False:
        fcall(rootdir, globstr1, globstr2, mergecolsonly, parallelprocessing, star)
    else:
        Parallel(n_jobs=njobs)(delayed(fcall)(rootdir, file1, file2, mergecolsonly, parallelprocessing, star)
                          for file1, file2 in zip(filenames1, filenames2))

def get_outfilename(oldfilename, outpath=None):
    oldDirname = path.dirname(oldfilename)
    oldBasename = path.basename(oldfilename)
    newBasename = MERGEDCSVPREFIX + oldBasename
    # outFilename = get_OutFilename(oldfilename, outpath)
    if outpath is None:
        outFilename = path.join(oldDirname, newBasename)
    else:
        outFilename = path.join(outpath, newBasename)

    return outFilename

def get_outfilenames(oldfilenames, outpath=None):
    outFilenames = [get_outfilename(oldfname) for oldfname in oldfilenames]
    return outFilenames

def remove_old_files(filenames):
    """Removes the separate csv files after merging them into a single file."""
    for filename in filenames:
        os.remove(filename)

def merge_csv_files(csvFilenames, outpath=None):
    """Merges multiple csv files to a single file."""
    dataframe = pd.concat([pd.read_csv(fname, header=None) for fname in csvFilenames])
    outFilename = get_outfilename(csvFilenames[0])
    dataframe.to_csv(outFilename)
    # remove_old_files(csvFlenames)

    return outFilename

def convert_multiple_dirs(rootdir, mergecolsonly, parallelprocessing):
    """Descends into multiple dirs and converts their root files to csv.
       Dirnames need to be formatted like: YYYY_MM_DD or YYYY_DD_MM"""
       # only works for data as of now, not for mc
    subdirs = glob_and_check(path.join(rootdir, r"20\d\d_\d\d_\d\d/"))
	# not used, setting globstr1 and 2 to None and subsequent none handling in call_c_converter needs to be implemented first
    Parallel(n_jobs=njobs)(delayed(convert_to_csv)(rootdirectory, mergecolsonly, parallelprocessing)
                           for rootdirectory in subdirs)
    subdirGlobs = sorted([path.join(subdir, "/*"+ MERGECOLENDING) for subdir in subdirs])
    filenames = [glob_and_check(subdirGlob) for subdirGlob in subdirGlobs]
    outFilename = merge_csv_files(filenames)
    return outFilename

def converter(rootdir, globstr1, globstr2, multidir=False, njobs=8, mergecolsonly=False, parallelprocessing=True, star=False):
    """See convert_to_csv documentation. Converts from root to csv for either one
	   direcotry or multiple dirs."""
    if mergecolsonly is True:
        assert path.dirname(globstr1) == "", f"Globstring is not supposed to contain a directory, aborting. globstr1: {globstr1}"
        assert path.dirname(globstr2) == "", f"Globstring is not supposed to contain a directory, aborting. globstr2: {globstr2}"
    if multidir is False:
        outFilenames = convert_to_csv(rootdir, globstr1, globstr2, mergecolsonly, parallelprocessing, star, njobs)
    else:
        outFilenames = convert_multiple_dirs(rootdir, mergecolsonly, parallelprocessing)
    logger.info("Converting root to csv all done.")
    return outFilenames

@click.command()
@click.option('--rootdir', default="./", type=click.Path(resolve_path=True, dir_okay=True, file_okay=False, writable=True),
              help='RootDir from which to read the files')
@click.option('--globstr1', default="./root/*_M1_*.root", type=str,
              help='globstring from which to read the M1 files')
@click.option('--globstr2', default="./root/*_M2_*.root", type=str,
              help='globstring from which to read the M2 files')
@click.option('--multidir', default=False, type=bool,
              help='Whether to process multiple directories or not. If `True`, discovers all directories in the `rootdir`, and descends into each dir iteratively.')
@click.option('--njobs', default=16, type=int,
              help='Number of jobs to use for multiprocessing. Defaults to 8. Script only needs around 500mb RAM')
@click.option('--mergecolsonly', '-mo', default=False, type=bool,
              help='Whether to only extract the merge cols from the root file.')
@click.option('--star', default=False, type=bool,
              help='Whether to do star level proccessing. In this case the globstr1 becomes that for the star files and globstr2 for the calibrated files')
@click.option('--parallelprocessing', '-pp', default=True, type=bool,
              help='Whether to parallellize extraction of the csv.')
def call_converter(**args):
    converter(**args)


if __name__ == '__main__':
    call_converter()
