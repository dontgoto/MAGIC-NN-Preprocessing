from os import path
import os
import logging

import pandas as pd
import click
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from dataprepro.csv2hdf5 import glob_and_check

logger = logging.getLogger(__name__)

DATAM1SSTARCOLS = ["nanotime_1", "time2_1", "id_1"]
DATAM2SSTARCOLS = ["nanotime_2", "time2_2", "id_2"]
MCM1SSTARCOLS = ["mcE_1", "id_1", "zd_1", "az_1"]
MCM2SSTARCOLS = ["mcE_2", "id_2", "zd_2", "az_2"]
CALIBCOLS = ["nanotime", "time2", "id"]
DATACALIBCOLS = ["nanotime", "time2", "id"]
MERGECOLSIDX = [0, 1, 2]
DATAMERGECOLSIDX = [0, 1, 2]
MCMERGECOLSIDX = [0, 1, -2, -1]
FILTERPREFIX = "eventfiltered_"
FILTERDIR = "filtered"
PROCESSEDDIR = "processed/"
BUGDIR = "weirdbug/"
MISSINGDIR = "missing/"


def within_pm1(len1, len2):
    if (len1 == len2) or (len1+1 == len2) or (len1-1 == len2):
        return True
    else:
        return False

def asssert_anyx_in_y(x, y):
    assert isinstance(x, str)
    if isinstance(y, (list, np.array)):
        assert np.any([x in f for f in y]), f"No {x} in {y[0:2]}..., aborting."
    elif isinstance(y, str):
        assert x in y, f"No {x} in {y}..., aborting."
    else:
        raise NotImplementedError(f"checking if {x}, type {type(x)} is in {y}, type {type(y)} is not implemented")

def split_into_telescopes(fnames):
    """Takes a list of fnames and returns two lists,
       each containing only fnames from M1 and M2"""
    if isinstance(fnames, str):
        fnames = glob_and_check(fnames)
    fnames1 = [fname for fname in fnames if "_M1_" in fname]
    fnames2 = [fname for fname in fnames if "_M2_" in fname]
    return fnames1, fnames2

def get_run(fname, full=False):
    """Returns the run number from a str fname. If full is set it returns the
       subrun part of the str as well. (_Y_ and _I_ only)"""
    basename = path.basename(fname)
    run = ""
    if "GA_" in basename:
        if "_S_" in basename:
            # example: GA_za05to35_8_1743990to1745989_S_wr_2.root
            if "_S_" in basename:
                run = basename.split("_")[3]
            # GA_M1_za35to50_8_1740993_Y_wr.root
            else:
                run = basename.split("_")[4]
    elif ("_Y_" in basename) or ("_I_" in basename):
        # 20161008_M1_05057440.005_Y_CrabNebula-W0.40+215.root or
        # eventfiltered_20161008_05057440.005_Y_CrabNebula-W0.40+215.root
        runid = basename.split("_")[2]
        if full is False:
            run = runid.split(".")[0]
        else:
            run = runid
    elif ("_S_" in basename) or ("_Q_" in basename):
        run = basename.split("_")[1]
    else:
        raise NotImplementedError(f"Wrong filename: {fname}, runname retrieval is undefined")
    return run


class EventRemover():
    """Removes all events from calibfeatures that arent in starmergecols
       (that is post image cleaning), resulting in new calibrated files with only the events remaining
       after after the image cleaning. These can then be preprocessed and classified."""

    def __init__(self, tofiltercalibglob, starglob, superstarmcolglob, njobs=2, invert=False):
        self.validfullruns = list(set(self._get_valid_fullruns(starglob)))
        self.superstarmcolfnames = self._get_fnames(superstarmcolglob,
                                                    filetype="_S_")
        self.tofiltercalibfnames = self._get_fnames(tofiltercalibglob,
                                                    fullrunfilter=True)
        self.assert_startup_fnames()
        self.invert = False #hardcoded for now not implemented yet
        self.outfilenames = []
        self.setup_dirs(tofiltercalibglob)
        self.njobs = njobs
        if "GA_" in self.tofiltercalibfnames[0]:
            self.mc = True
        else:
            self.mc = False
        self.set_mcols(self.mc)

    def setup_dirs(self, tofiltercalibglob):
        olddir = path.dirname(tofiltercalibglob)
        self.processeddir = path.join(olddir, PROCESSEDDIR)
        os.makedirs(self.processeddir, exist_ok=True)
        self.bugdir = path.join(olddir, BUGDIR)
        os.makedirs(path.join(olddir, BUGDIR), exist_ok=True)
        os.makedirs(path.join(olddir, MISSINGDIR), exist_ok=True)
        os.makedirs(path.join(olddir, FILTERDIR), exist_ok=True)

    def set_mcols(self, mc):
        if mc is False:
            self.m1sscols = DATAM1SSTARCOLS
            self.m2sscols = DATAM2SSTARCOLS
            self.mcolidx = DATAMERGECOLSIDX
        else:
            self.m1sscols = MCM1SSTARCOLS
            self.m2sscols = MCM2SSTARCOLS
            self.mcolidx = MCMERGECOLSIDX

    def assert_startup_fnames(self):
        assert np.any(["_M1_" in f for f in self.tofiltercalibfnames]), \
            "No M1 files in tofiltercalibfnames, aborting."
        assert np.any(["_M2_" in f for f in self.tofiltercalibfnames]), \
            "No M2 files in tofiltercalibfnames, aborting."
        assert not np.any([".root" in f for f in self.tofiltercalibfnames]), \
            "Change the tofiltercalibglob, these are .root but are supposed to be csv files."

    def single_remove_events(self, ssmcoldf, ssmcolfname, tofiltercalibfnames, sscols):
        currentrun = get_run(ssmcolfname)
        tofiltercalibdf, oldname = self._merge_subruns(tofiltercalibfnames,
                                                       currentrun, getoldname=True)
        if len(tofiltercalibdf1) == 0:
            logger.warning(f"calibfiles already processed, skipping {path.basename(ssmcolfname)} current run: {currentrun} \n")
            return False
        filtermask = self.get_filtermask(tofiltercalibdf.iloc[:, self.mcolidx], ssmcoldf[sscols])
        if self.invert is False:
            filtermask, tofiltercalibdf = self.fix_idiosyncracies(filtermask, tofiltercalibdf, ssmcoldf[sscols])
        newfname = self._get_newfilename(oldname=oldname)
        nameiter.set_description(f"Saving {path.basename(newfname)}")
        # if not assert_single_fmask_and_df(filtermask, tofiltercalibdf, ssmcoldf):
            # return False
        tofiltercalibdf.iloc[filtermask].to_csv(newfname, header=False, index=False)
        del tofiltercalibdf
        return newfname

    def _move_processed_files(self, oldnames, outdir=None):
        if isinstance(oldnames, str):
            oldnames = [oldnames]
        if outdir is None:
            outdir = self.processeddir
        for oldname in oldnames:
            oldbasename = path.basename(oldname)
            newname = path.join(outdir, oldbasename)
            os.rename(oldname, newname)

    def _get_fnames(self, glob, filetype=None, fullrunfilter=False):
        filetypes = ["_Y_", "_I_", "_S_"]
        if isinstance(glob, str):
            fnames = glob_and_check(glob)
        elif isinstance(glob, list):
            fnames = glob
        else:
            raise NotImplementedError(f"Argument not implemented. type(glob): {type(glob)}")
        if filetype is None:
            for ftype in filetypes:
                if ftype in fnames[0]:
                    filetype = ftype
        if filetype not in filetypes:
            raise NotImplementedError(f"Wrong filetpye: {filetype}, glob: {glob} filename retrieval is undefined")
        allfit = np.all([filetype in f for f in fnames])
        assert allfit, \
               f"Not all files are from the same processing step, aborting. Filetype: {filetype}, glob: {glob}"

        if fullrunfilter:
            fnamesnew = [f for f in fnames
                      if get_run(f, full=True) in self.validfullruns]
            assert fnamesnew, ("fnames after fullrunfilter empty, aborting."
                   f"{path.dirname(fnames[0])} oldfnames: {[path.basename(f) for f in fnames]}")
            fnames = fnamesnew

        return sorted(fnames)

    def _get_valid_fullruns(self, starglob):
        """Takes the star filenames corresponding to the superstar files used for filtering and
           creates a list of valid runs+subruns that are to be used at the calibrated level"""
        starfnames = self._get_fnames(starglob, filetype="_I_")
        validfullruns = set([get_run(f, full=True) for f in starfnames])
        return sorted(list(validfullruns))


    def get_relevant_fullruns(self, run):
        return [f for f in self.validfullruns if run in f]


    def assert_filtermask_and_dfs(self, fmask1, fmask2, calibdf1, calibdf2, ssmcoldf):
        status = True
        # more detail inside of get_filtermask
        if (fmask1 is None) or (fmask2 is None):
            status = False
        elif (len(fmask1) == 0) or (len(fmask2) == 0):
            status = False
        elif np.sum(fmask1) != np.sum(fmask2):
            logger.warning("Number of filtered events are not the same for each telescope, skipping."
             f"#M1: {np.sum(fmask1)}, #M2: {np.sum(fmask2)}")
            status = False
        else:
            status = True
        if status is True:
            try:
                assert within_pm1(len(calibdf1.iloc[fmask1]), len(ssmcoldf)),\
                    ("len of M1 is not within +-1 of ssmcoldf, aborting.\n"
                     f"len calibdf: {len(calibdf1.iloc[fmask1])}, len ssmcoldf: {len(ssmcoldf)}")
                assert within_pm1(len(calibdf2.iloc[fmask2]), len(ssmcoldf)),\
                    ("len of M2 is not within +-1 of ssmcoldf, aborting.\n"
                     f"len calibdf: {len(calibdf2.iloc[fmask2])}, len ssmcoldf: {len(ssmcoldf)}")
            except IndexError:
                logger.critical("\n"
                  f"len ssmcoldf: {len(ssmcoldf)}\n"
                  f"true sum filtermask1: {np.sum(fmask1)}\n"
                  f"len fmask1: {len(fmask1)}\n"
                  f"len calibdf1: {len(calibdf1)}\n"
                  f"true sum filtermask2: {np.sum(fmask2)}\n"
                  f"len fmask2: {len(fmask2)}\n"
                  f"len calibdf2: {len(calibdf2)}\n")
                raise
        return status

    def fix_idiosyncracies(self, filtermask, tofiltercalibdf, ssmcoldf):
        calibmcols = tofiltercalibdf.iloc[:, self.mcolidx].values
        if np.sum(filtermask) == 0:
            logger.warning("filtermask is all false, skipping")
            return None, None
        # handle duplicates at the end of superstar file
        if len(calibmcols[filtermask]) == (len(ssmcoldf)-1):
            if np.array_equal(ssmcoldf.values[-1], ssmcoldf.values[-2]):
                calibmcols = np.vstack([calibmcols, ssmcoldf.values[-1].reshape(1, 3)])
                lastcalibrow = tofiltercalibdf.iloc[filtermask].iloc[-1]
                filtermask = np.append(filtermask, True)
                tofiltercalibdf = tofiltercalibdf.append(lastcalibrow)
                if np.array_equal(lastcalibrow.values, ssmcoldf.values[-1]):
                    return filtermask, tofiltercalibdf
                logger.warning("idk what is happening")
        # when did this happen? when the last calibrows pre or post filtermask were the same?
        # probably post filtermask, or else the condition for equality after filtermask wouldnt have worked
        elif len(calibmcols[filtermask]) == (len(ssmcoldf)+1):
            lastcalibrowequal = np.array_equal(calibmcols[-1], calibmcols[-2])
            lastssrowequal = np.array_equal(ssmcoldf.iloc[-1], ssmcoldf.iloc[-2])
            lastssandcalibrowequal = np.array_equal(calibmcols[-1], ssmcoldf.iloc[-1])
            if lastcalibrowequal and lastssandcalibrowequal and not lastssrowequal:
                filtermask[-1] = False
        if not np.array_equal(calibmcols[filtermask][:len(ssmcoldf)], ssmcoldf.values):
            logger.warning("calib and superstar still not equal, idk what is going on, skipping.")
            logger.warning(f"len fmask: {len(filtermask)}\n"
                  f"true sum filtermask: {np.sum(filtermask)}\n"
                  f"len calibmcols: {len(calibmcols)}\n"
                  f"len ssmcols: {len(ssmcoldf)}\n")
            # raise ArithmeticError("raising error for now, start bugfixing")
            return None, None
        else:
            return filtermask, tofiltercalibdf


    def assert_single_fmask_and_df(self, filtermask, tofiltercalibdf, ssmcoldf):
        pass

    def remove_events(self):
        """Filters all tofiltercalib files with the starfiles provided and throws out
           not matching subruns, merges the remaining subruns to runlevel.
           Same for the mergecolcalibfiles. Then compares each run of mcolcalibfiles
           to the superstarmergecolfile of the same run and throws out all events in
           tofiltercalib based on the filtermask. Finally writes to disk."""
        tofiltercalibfnames1, tofiltercalibfnames2 = split_into_telescopes(self.tofiltercalibfnames)

        nameiter = tqdm(self.superstarmcolfnames)
        for ssmcolfname in nameiter:
            nameiter.set_description(f"Processing {path.basename(ssmcolfname)}")
            currentrun = get_run(ssmcolfname)
            ssmcoldf = pd.read_csv(ssmcolfname, index_col=False)
            # newfname1 = self.single_remove_events(ssmcoldf, ssmcolfname, tofiltercalibfnames1, self.m1sscols)
            # newfname2 = self.single_remove_events(ssmcoldf, ssmcolfname, tofiltercalibfnames2, self.m2sscols)
                # if self.assert_filtermask_and_dfs(filtermask1, filtermask2, tofiltercalibdf1, tofiltercalibdf2, ssmcoldf) is False:
                    # self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames1, currentrun), self.bugdir)
                    # self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames2, currentrun), self.bugdir)
                    # os.remove(newfname1)
                    # os.remove(newfname2)
                    # continue
            tofiltercalibdf1, oldname1 = self._merge_subruns(tofiltercalibfnames1,
                                                             currentrun, getoldname=True)
            tofiltercalibdf2, oldname2 = self._merge_subruns(tofiltercalibfnames2,
                                                             currentrun, getoldname=True)
            if len(tofiltercalibdf1) == 0:
                logger.warning(f"calibfiles already processed, skipping {path.basename(ssmcolfname)} current run: {currentrun} \n")
                continue

            filtermask1 = self.get_filtermask(tofiltercalibdf1.iloc[:, self.mcolidx], ssmcoldf[self.m1sscols])
            filtermask2 = self.get_filtermask(tofiltercalibdf2.iloc[:, self.mcolidx], ssmcoldf[self.m2sscols])
            if self.invert is False:
                filtermask1, tofiltercalibdf1 = self.fix_idiosyncracies(filtermask1, tofiltercalibdf1, ssmcoldf[self.m1sscols])
                filtermask2, tofiltercalibdf2 = self.fix_idiosyncracies(filtermask2, tofiltercalibdf2, ssmcoldf[self.m2sscols])

                if self.assert_filtermask_and_dfs(filtermask1, filtermask2, tofiltercalibdf1, tofiltercalibdf2, ssmcoldf) is False:
                    self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames1, currentrun), self.bugdir)
                    self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames2, currentrun), self.bugdir)
                    continue

            newfname1 = self._get_newfilename(oldname=oldname1)
            newfname2 = self._get_newfilename(oldname=oldname2)

            nameiter.set_description(f"Saving {path.basename(newfname1)}")
            tofiltercalibdf1.iloc[filtermask1].to_csv(newfname1, header=False, index=False)
            nameiter.set_description(f"Saving {path.basename(newfname2)}")
            tofiltercalibdf2.iloc[filtermask2].to_csv(newfname2, header=False, index=False)
            self.outfilenames.extend([newfname1, newfname2])
            # self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames1, currentrun))
            # self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames2, currentrun))


    def parallel_remove_events(self):
        tofiltercalibfnames1, tofiltercalibfnames2 = split_into_telescopes(self.tofiltercalibfnames)
        # this doesnt work, the called function needs to be a regular function that takes the class as an argument and then does the computation on the passed class
        newfnames = Parallel(n_jobs=self.njobs)\
            (delayed(self.single_remove_events)\
             (ssmcolfname, tofiltercalibfnames1, tofiltercalibfnames2)
             for ssmcolfname in self.superstarmcolfnames)
        self.outfilenames.extend(newfnames)

    def single_remove_events(self):
        pass

    def get_filtermask(self, calibmcoldf, ssmcoldf):
        """Takes the mergecols extracted from the superstar file and compares the ones from the
        calib files to it. When calib event doesnt match superstar event, the filtermask is set
        to 0. Once ss=calib for the current row the next ss event is looked at
        till the end of the file. In the end all events not in the ss file are
        removed from the calib file. Calibdf has to contain the same run as the ssmcoldf.
        Maybe do a filename assertion."""
        if isinstance(calibmcoldf, pd.DataFrame):
            calibmcoldf = calibmcoldf.values.astype(float)
        else:
            calibmcoldf = calibmcoldf.astype(float)
        if isinstance(ssmcoldf, pd.DataFrame):
            ssmcoldf = ssmcoldf.values.astype(float)
        else:
            ssmcoldf = ssmcoldf.astype(float)

        sstarindex = 0
        filtermask = np.ones(len(calibmcoldf), dtype=bool)
        ssmcollen = len(ssmcoldf)
        missingvalues = len(calibmcoldf)-ssmcollen
        assert missingvalues >= 0, f"Number of missing values is supposed to be positive no.: {missingvalues}"
        for calibindex, calibrow in enumerate(calibmcoldf):
            if sstarindex == ssmcollen:
                filtermask[calibindex:] = False
                break
            if not np.array_equal(ssmcoldf[sstarindex], calibrow):
                if self.invert is True:
                    filtermask[calibindex] = True
                else:
                    filtermask[calibindex] = False
            else:
                sstarindex += 1
        # set as assertion once bug is fixed
        if self.invert is False:
            if not np.array_equal(calibmcoldf[filtermask], ssmcoldf):#, \
                logger.info("filtermask doesnt make equal, trying to fix")
                # raise ArithmeticError("Start debugging")
            # needs a fix for when there is a duplicate in the middle/end of the ssmcoldf (results in the filtered calibdf being too short)
            # sometimes the ss file has a duplicate event at the end which is not duplicated in the calibrated file ?!?!?
            # assert np.sum(filtermask) != 0, "something went wrong len of filtermask is 0 start debugging"

        return filtermask


    def get_fnames_for_samerun(self, fnames):
        samerun = np.all([get_run(f) in get_run(fnames[0])
                          for f in fnames])
        assert samerun, "files are not all from the same run, aborting"
        return samerun


    def _handle_fnames_for_merging(self, fnames, run=None):
        assert np.all(["_Y_" in f for f in fnames]) or np.all(["_I_" in f for f in fnames]),\
            "not all files are calibfiles, aborting"
        allM1 = np.all(["_M1_" in f for f in fnames])
        allM2 = np.all(["_M2_" in f for f in fnames])
        assert (allM1 or allM2), "files are not all from the same telescope, aborting"
        assert fnames, "List of fnames is empty aborting"
        if run is None:
            samerun = self.get_fnames_for_samerun(fnames)
        else:
            olddir = path.dirname(fnames[0])
            newfnames = [path.join(self.processeddir, path.basename(f)) for f in fnames]
            newfnames = [f for f in newfnames if get_run(f) == run]
            fnames = [f for f in fnames if run in path.basename(f)]
            if not fnames:
                if not newfnames == 0:
                    f"No star or calibfiles for run {run}"
                return []

        assert len(fnames) != 0, "len fnames is zero that shouldnt happen, aborting"
        if len(fnames) < len(self.get_relevant_fullruns(run)):
            logger.warning("missing files for current run, skipping")
            return []
        fnames =  sorted(list(set(fnames)))
        return fnames


    def _merge_subruns(self, fnames, run=None, getoldname=False, mergecolsonly=False):
        """Takes multiple fnames (subruns) from the same run and merges them in a DF. Returns the DF"""
        fnames = self._handle_fnames_for_merging(fnames, run)
        if not fnames:
            if getoldname is True:
                return [], ""
            else:
                return []

        if "mergecols" in fnames[0]:
            if len(fnames) == 1:
                rundf = pd.read_csv(fnames[0], index_col=False)
            else:
                rundf = pd.DataFrame(np.vstack([pd.read_csv(f, index_col=False).values for f in fnames]))
        else:
            # depends on the fact the merge cols extract in globRoot2csv.C sets headers, if that is changed set header=None here too
            if len(fnames) == 1:
                rundf = pd.read_csv(fnames[0], header=None, index_col=False)
            else:
                rundf = pd.DataFrame(np.vstack([pd.read_csv(f, header=None, index_col=False).values for f in fnames]))

        colnames = pd.read_csv(fnames[0]).columns
        rundf.columns = colnames
        if getoldname is True:
            oldname = self.get_oldname(fnames[0])
            return rundf, oldname
        else:
            return rundf


    @staticmethod
    def get_oldname(fname):
        """Takes fname and removes the subrun from it, leaving only the name with run identifier."""
        assert isinstance(fname, str), f"fname is not a string, aborting. fname: {fname}"
        dirname = path.dirname(fname)
        oldbasename = path.basename(fname)
        if "GA_" in oldbasename:
            return fname
        elif ("_Y_" in oldbasename) or ("_I_" in oldbasename):
            split = oldbasename.split("_")
            split[2] = split[2].split(".")[0]
            newbasename = "_".join(split)
            return path.join(dirname, newbasename)
        elif "_S_" in fname:
            return fname
        else:
            raise NotImplementedError(f"filetype not supported. fname: {fname}")


    @staticmethod
    def _get_newfilename(oldname):
        """Takes the old path and returns it with the filename prepended with `eventfiltered_`"""
            # example filename: 20160929_M1_05057144.015_I_CrabNebula-W0.40+215_mergecols.csv
            # removes subrun (bc files are merged to runlevel) and prepends eventfiltered
        dirname = path.dirname(oldname)
        newdirname = path.join(dirname, FILTERDIR)
        oldbasename = path.basename(oldname)
        if "GA_" in oldbasename:
            if ("_Y_" in oldbasename) or ("_I_" in oldbasename):
                newbasename = FILTERPREFIX + oldbasename
        elif ("_Y_" in oldbasename) or ("_I_" in oldbasename):
            splitname = oldbasename.split("_")
            splitname[2] = splitname[2].split(".")[0]
            newbasename = FILTERPREFIX + "_".join(splitname)
        newfilename = path.join(newdirname, newbasename)
        return newfilename


    def __call__(self):
        self.remove_events()


    def invert_selection(self):
        """Selects and writes a new csv file with all events that do not conform to the filtermask
        (which means they did get filtered out from calibrated->star or whatever step)"""
        pass


def process_events(**args):
    remover = EventRemover(**args)
    remover.remove_events()


@click.command()
@click.option('--tofiltercalibglob', "-fcg", default="./csv/*_Y_*.csv", type=click.Path(),
              help=('Dir from which to read the calibrated (subrun level) csv files'
                    'which are going to get filtered.'))
@click.option('--starglob', "-sg", default="./star/*_M1_*I_*.root", type=click.Path(),
              help=('Glob to the star files generated after image cleaning can be root or whatever.'
                    'Only the subrun names are important for filtering.'))
@click.option('--superstarmcolglob', "-ssg", default="./superstar/mergecols/*_S_*.csv", type=click.Path(),
              help=('CSV Files from which to read the features (mergecols) for filtering events.'
                    'Generated after merging (superstar) and extracting from root.'))
@click.option('--njobs', "-n", default=1, type=int,
              help='Number of jobs for processing.')
@click.option('--invert', "-i", default=False, type=bool,
              help='Whether to do a inverse filter (only leave events that get thrown out during star cleaning).')
def main(**args):
    try:
        if args["njobs"] == 1:
            process_events(**args)
        else:
            raise NotImplementedError("Multiprocessing needs to be implemented still.")
            # func that splits the globstirngs into new globstrings (should be in process_star.py or somewhere else)
            # newcalibglobs = split_glob(tofiltercalibglob)
            # newstarglobs = split_glob(starglob)
            # newsuperstarglobs = split_glob(superstarmcolglob)
            # globzip = zip(newcalibglobs, newstarglobs, newsuperstarglobs)
            # Parallel(n_jobs=njobs)(delayed(process_events)(ncalibglob, nstarglob, nsuperstarglob, njobs, invert)
            #                        for ncalibglob, nstarglob, nsuperstarglob in globzip)
    except:
        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
