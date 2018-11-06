from os import path
import os
import pandas as pd
import click
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from dataprepro.csv2hdf5 import glob_and_check
from dataprepro.remove_events import EventRemover


class CleaningDiffRemover(EventRemover):

    def __init__(self, tofiltercalibglob, star1glob, star2glob, processeddir="stardiff", njobs=2, normglob=None):
        self.validfullruns = list(set(self._get_valid_fullruns(star1glob)))
        self.star1mcolfnames = self._get_fnames(star1glob,
                                                filetype="_I_")
        self.star2mcolfnames = self._get_fnames(star2glob,
                                                filetype="_I_")
        self.tofiltercalibfnames = self._get_fnames(tofiltercalibglob,
                                                    filetype="_Y_",
                                                    fullrunfilter=True)
        if normglob is not None:
            self.normedfnames = self._get_fnames(normglob,
                                                 filetype="hdf5")
        else:
            self.normedfnames = None
        self.assert_startup_fnames()
        self.outfilenames = []
        olddir = path.dirname(tofiltercalibglob)
        self.processeddir = path.join(olddir, processeddir)
        os.makedirs(self.processeddir, exist_ok=True)
        self.bugdir = path.join(olddir, BUGDIR)
        os.makedirs(path.join(olddir, BUGDIR), exist_ok=True)
        os.makedirs(path.join(olddir, MISSINGDIR), exist_ok=True)
        os.makedirs(path.join(olddir, FILTERDIR), exist_ok=True)
        self.njobs = njobs
        if "GA_" in self.tofiltercalibfnames[0]:
            self.mc = True
        else:
            self.mc = False
        self.set_mcols(self.mc)
        # if self.mc is False:
            # self.m1sscols = DATAM1SSTARCOLS
            # self.m2sscols = DATAM2SSTARCOLS
            # self.mcolidx = DATAMERGECOLSIDX
        # else:
            # self.m1sscols = MCM1SSTARCOLS
            # self.m2sscols = MCM2SSTARCOLS
            # self.mcolidx = MCMERGECOLSIDX


    def get_difffiltermask(self, s1m1mcoldf, s1m2mcoldf, tofiltercalibdf1, tofiltercalibdf2):
        filtermask1m1 = self.get_filtermask(tofiltercalibdf1.iloc[:, self.mcolidx], s1m1mcoldf[self.m1sscols])
        filtermask1m2 = self.get_filtermask(tofiltercalibdf2.iloc[:, self.mcolidx], s1m2mcoldf[self.m2sscols])
        filtermask1 = filtermask1m1 ^ filtermask2m1
        return filtermask1

    def save_filtered(self, tofiltercalibdf, filtermask, oldname, nameiter, normglob=None):
        newfname = self._get_newfilename(oldname=oldname)
        nameiter.set_description(f"Saving {path.basename(newfname)}")
        if normglob is None:
            tofiltercalibdf.iloc[filtermask].to_csv(newfname, header=False, index=False)
        else:
            normedfname = normglo
            cth.readhdf5(normedfname, "r")

    def save_and_filter_hdf5(self, filtermask, oldname):
        pass


    def single_remove_events(self, s1m1mcolfname, s1m2mcolfname, s2m1mcolfname, s2m2mcolfname,
                             tofilteraclibfnames1, tofiltercalibfnames2, nameiter):

        nameiter.set_description(f"Processing {path.basename(s1mcolfname)}")
        currentrun = get_run(s1mcolfname)
        s1m1mcoldf = pd.read_csv(s1m1mcolfname, index_col=False)
        s1m2mcoldf = pd.read_csv(s1m2mcolfname, index_col=False)
        s2m1mcoldf = pd.read_csv(s2m1mcolfname, index_col=False)
        s2m2mcoldf = pd.read_csv(s2m2mcolfname, index_col=False)
        tofiltercalibdf1, oldname1 = self._merge_subruns(tofiltercalibfnames1,
                                                         currentrun, getoldname=True)
        tofiltercalibdf2, oldname2 = self._merge_subruns(tofiltercalibfnames2,
                                                         currentrun, getoldname=True)
        if len(tofiltercalibdf1) == 0:
            print(f"calibfiles already processed, skipping {path.basename(ssmcolfname)} current run: {currentrun} \n")
            return None
        filtermask1 = self.get_difffiltermask(s1m1mcoldf, s1m2mcoldf, tofiltercalibdf1, tofiltercalibdf2)
        filtermask2 = self.get_difffiltermask(s2m1mcoldf, s2m2mcoldf, tofiltercalibdf1, tofiltercalibdf2)
        # filtermask1m1 = self.get_filtermask(tofiltercalibdf1.iloc[:, self.mcolidx], s1m1mcoldf[self.m1sscols])
        # filtermask1m2 = self.get_filtermask(tofiltercalibdf2.iloc[:, self.mcolidx], s1m2mcoldf[self.m2sscols])
        # filtermask2m1 = self.get_filtermask(tofiltercalibdf1.iloc[:, self.mcolidx], s1m1mcoldf[self.m1sscols])
        # filtermask2m2 = self.get_filtermask(tofiltercalibdf2.iloc[:, self.mcolidx], s1m2mcoldf[self.m2sscols])
        # filtermask1 = filtermask1m1 ^ filtermask2m1
        # filtermask2 = filtermask1m2 ^ filtermask2m2

         if self.invert is False:
            filtermask1, tofiltercalibdf1 = self.fix_idiosyncracies(filtermask1, tofiltercalibdf1, ssmcoldf[self.m1sscols])
            filtermask2, tofiltercalibdf2 = self.fix_idiosyncracies(filtermask2, tofiltercalibdf2, ssmcoldf[self.m2sscols])
             if self.assert_filtermask_and_dfs(filtermask1, filtermask2, tofiltercalibdf1, tofiltercalibdf2, s1m1mcoldf) is False:
                self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames1, currentrun), self.bugdir)
                self._move_processed_files(self._handle_fnames_for_merging(tofiltercalibfnames2, currentrun), self.bugdir)
                return None
        self.save_filtered(tofiltercalibdf1, filtermask1, oldname1, nameiter)
        self.save_filtered(tofiltercalibdf2, filtermask2, oldname2, nameiter)

        # newfname1 = self._get_newfilename(oldname=oldname1)
        # newfname2 = self._get_newfilename(oldname=oldname2)
        # nameiter.set_description(f"Saving {path.basename(newfname1)}")
        # tofiltercalibdf1.iloc[filtermask1].to_csv(newfname1, header=False, index=False)
        # nameiter.set_description(f"Saving {path.basename(newfname2)}")
        # tofiltercalibdf2.iloc[filtermask2].to_csv(newfname2, header=False, index=False)

    def remove_events(self):
        tofiltercalibfnames1, tofiltercalibfnames2 = split_into_telescopes(self.tofiltercalibfnames)
        s1m1mcolfnames, s1m2mcolfnames = split_into_telescopes(self.star1mcolfnames)
        s2m1mcolfnames, s2m2mcolfnames = split_into_telescopes(self.star2mcolfnames)
        nameiter = tqdm(zip(star1m1mcolfnames, star1m2mcolfnames,
                            star2m1mcolfnames, star2m2mcolfnames),
                        total=len(self.star1m1mcolfnames))
        for s1m1mcolfname, s1m2mcolfname, s2m1mcolfname, s2m2mcolfname in nameiter:
            self.singe_remove_events(s1m1mcolfname, s1m2mcolfname, s2m1mcolfname, s2m2mcolfname,
                                     tofilteraclibfnames1, tofiltercalibfnames2, nameiter)


@click.command()
@click.option('--star1glob', "-sg1", default="./star/mergecols/*I_*.root", type=click.Path(),
              help=('Glob to the star mergecol files generated after image first cleaning.')
@click.option('--star2glob', "-sg2", default="./star/test/mergecols/*I_*.root", type=click.Path(),
              help=('Glob to the star mergecol files generated after different image cleaning thresholds.'))
@click.option('--njobs', "-n", default=2, type=int,
              help='Number of jobs for processing.')
@click.option('--normedfilesglob', "-nf", default=None, type=click.Path(),
              help='glob to the normalized files that will get filtered.')
def main(**args):
    try:
        remover = CleaningDiffRemover(**args)
        remover.remove_events()
    except:
        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
