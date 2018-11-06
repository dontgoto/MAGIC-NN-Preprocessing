from os import path
from itertools import accumulate
import pandas as pd
import h5py as h5
import click
from csv2hdf5 import glob_and_check
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

M1SSTARCOLS = ["nanotime_1", "id_1", "time2_1"]
M2SSTARCOLS = ["nanotime_2", "id_2", "time2_2"]
CALIBCOLS = ["nanotime", "id", "time2"]

class EventRemover():
    """Removes all events from calibfeatures that arent in starmergecols (that is post image cleaning),
       resulting in a new file with only the NN Energies that are there after the image cleaning. These can then be fed into melibea."""

    def __init__(self, tofilterdir, starmergecols, superstarfeatures, calibmergecolsdir, multifile, filtertype):
        self.tofilterdir = tofilterdir
        self.starmergecols = starmergecols
        self.superstarfeatures = superstarfeatures
        self.calibmergecolsdir = calibmergecolsdir
        self.multifile = multifile
        self.filtertype = filtertype
        self.outfilename = ""
        self.invert = False #hardcoded for now not implemented yet
        if self.multifile is True:
            pass
        self.validfullruns = [self._get_run()

    # implement slicing, slice filtermask into subruns and then slice the nnfile in the same manner
    # this slicing is for subrun files that remain after star cuts, if there are still subrun files in the dnn that get removed during star it wont work

    # starframes = [pd.read_csv(fname, sep=',') for fname in starfilenames]
    # calibframes = [pd.read_csv(fname, sep=',') for fname in calibfilenames]
    # slicing = [len(frame) for frame in calibframes]
    # print("reading done")
    # print(slicing)
    # print(len(starframes))
    # print(len(calibframes))
    # filtermasks = []

    def _get_valid_fullruns(starfnames):
        """Takes the star filenames corresponding to the superstar files used for filtering and creates a list of valid runs+subruns
           that are to be used at the calibrated level"""


        return validfullruns


    @staticmethod
    def _get_run(fname, full=False):
        if "_Y" in fname:
           runid = fname.split("_")[2]
           run = runid.split(".")[0]
        elif "_S_" in fname:
            run = fname.split("_")[1]
        else:
            raise NotImplementedError(f"Wrong filename: {fname}, runname retrieval is undefined")
        return run


    def cleaning_filter(self):
        calibfiles =
        for starfname in self.superstarfeatures:
            starrun = self._get_run(starfname)
            def subruns_from_run(fnames, run):
                fnames1 = [f for f in calibfiles
                           if "_M1_" in f
                           if self._get_run(f) in starrun]
                fnames2 = [f for f in calibfiles
                           if "_M2_" in f
                           if self._get_run(f) in starrun]
                return sorted(fnames1), sorted(fnames2)

            calibfnames1, calibfnames2 = subruns_from_run(calibfiles, starrun)
            assert (len(calibfnames1) != 0) and (len(calibfnames2) != 0), "list of calibfiles for current run empty, aborting. Run: {starrun}"
            calibfilter1 = self.filter_linearly(calibfnames1)
            calibfilter2 = self.filter_linearly(calibfnames2)
            assert len(calibfilter1) == len(calibfilter2), f"length of cleaned calibfiles is not the same, aborting. Run: {starrun}"


    @staticmethod
    def _merge_subruns(calibfiles):
        """Takes multiple calibfiles (subruns) from the same run and merges them in a DF. Returns the DF"""
        assert np.all(["_Y_" in f for f in calibfiles]), "not all files are calibfiles, aborting"
        allM1 = np.all(["_M1_" in f for f in calibfiles])
        allM2 = np.all(["_M2_" in f for f in calibfiles])
        assert (allM1 or allM2), "files are not all from the same telescope, aborting"
        samerun = np.all([self._get_run(calibfiles[0]) in self._get_run(f)
                          for f in calibfiles])
        assert samerun, "files are not all from the same run, aborting"

        calibfiles = sorted(calibfiiles)
        rundf = pd.concat([pd.read_csv(f) for f in calibfiles])

        return rundf


    @staticmethod
    def _get_linrem_mode(calibfile):
        """sets defines whether the calibfiles are from telescope 1 or 2, the comparison with the superstar file to be done accordingly"""
        assert isinstance(calibfile, basestring), f"calibfile: {print(calibfile)} is not a string, aborting""
        if isinstance(calibfile, (list, np.array)):
            calibfile = calibfile[0]
            if "_M1_" in calibfile:
                mode = "M1"
            elif "_M2_" in calibfile:
                mode = "M2"
        return mode


    def filter_linearly(self, calibfiles, superstarFile):
        """Takes the relevant cols extracted from the superstar file and compares the ones from the calib files to it.
        Every time the calib event doesnt match the superstar event it is removed.
        Once ss=calib for the current row the next ss event is looked at till the end of the file.
        In the end all events not in the superstar file are removed from the calib file.
        Calibdf has to contain the same run as the superstar file.
        Maybe do a filename assertion."""
        assert "_S_" in superstarFile, f"superstarFile: {superstarFile} is not superstar, aborting"


        calibdf = self._merge_subruns(calibfiles)
        mode = self._get_linrem_mode(calibfiles[0])
        if mode = "M1":
            sstarcols = M1SSTARCOLS
        elif mode = "M2":
            sstarcols = M2SSTARCOLS
        ssdf = pd.read_csv(superstarFile, usecols=sstarcols)


        starIndex = 0
        calibIndex = 0
        cleaningMask = np.ones(len(calibdf))
        while starIndex < len(ssdf):
            if ssdf.iloc[starIndex] != calibdf.iloc[calibIndex]:
                # calibdf.drop(pd.index(i))
                cleaningMask[calibIndex] = 0
                calibIndex += 1
            else:
                starIndex += 1
                calibIndex += 1

        assert ssdf.equals(calibdf), f"dataframes are not equal after removal, something isnt right, aborting. superstarFile: {superstarFile}"
        newfname = self._get_newfilename(oldname=calibfiles[0], mode="runmerge")
        # use to right df
        calibdf.iloc[cleaningMask].to_csv(newfname)

        return cleaningMask


    def _check_and_correct_lens(df, maxlen):
        if len(df) > maxlen:
            assert (maxlen+2) == len(df), f"idk what is happening here len calib: {len(pdCalib)}, len mergevalues: {len(mergevalues)}"
            if isinstance(df, pd.Dataframe):
                df = df.iloc[:-2]
            else:
                df = df[:-2]
        else:
            print(f"len merged: {len(df)}, maxlen: {maxlen}")
            assert len(mergedcfile) == maxlen, "Length of mergedcfile is lower than maxlen, idk what went wrong while merging both telescopes."

        return df


    @staticmethod
    def _get_nonduplicate_index(values):

        def _addition_with_reset(prev, curr):
            if curr == 0.0:
                return curr
            else:
                return prev+curr

        newindex = pd.Index(values, name=" id")
        duplicates = newindex.duplicated()
        nondupindex = np.array(list(accumulate(duplicates.astype(float)*0.01,
                                               _addition_with_reset)))
        nondupindex[duplicates]
        return values + nondupindex


    def split_into_telescopes(fnames):
        if isinstance(fnames, basestring):
            fnames = sorted(glob_and_check(fnames))
        fnames1 = [fname for fname in fnames if "_M1_" in fname]
        fnames2 = [fname for fname in fnames if "_M2_" in fname]

        return fnames1, fnames2


    def filter_csv_with_superstar(csvfnames, superstarfnames):
        """Removes all events from csvfiles and writes them to disk

           Parameters
           ==========

           csvfnames : list or str
               List of csvfilenames or globstr
           superstarfnames : list or str
               List of superstar filesnames or globstr to them"""

        csvfnames1, csvfnames2 = split_into_telescopes(csvfnames)
        mergecolfnames = sorted(get_mergecols(superstarfnames, superstar=True))

        subrun1fnames = merge_subruns(csvfnames1)
        subrun2fnames = merge_subruns(csvfnames2)
        assert len(subrun1fnames) == len(mergecolfnames), f"number of runs in csv and superstar files dont match, aborting len superstar: {len(mergecolfnames)}, len csv subrun1: {len(subrun1fnames)}"
        assert len(subrun2fnames) == len(mergecolfnames), f"number of runs in csv and superstar files dont match, aborting len superstar: {len(mergecolfnames)}, len csv subrun2: {len(subrun2fnames)}"

        for run1fnames, run2fnames, mergcolfname in zip(subrun1fnames, subrun2fnames, mergecolfnames):
            run1 = concatenatepds(run1fnames)
            run2 = concatenatepds(run1fnames)
            mergedpd.merge(run1, run2) #merge in a way that nothing gets lost, duplicates are ok, theyre removed in the next step
            superstarmergecol = pd.read_csv(mergecolfname)
            cleanedrun1 = remove_not_in_mergecol(run1, superstarmegecol)
            cleanedrun2 = remove_not_in_mergecol(run2, superstarmegecol)

            def cleaned_outfilename(fname):
                olddir = path.directory(fname)
                oldbasename = path.basename(fname)
                newbasename = "cleaned_" + oldbasename
                return path.join(olddir, newbasename)

            cleanedrun1.to_csv(cleaned_outfilename(outfilename))
            cleanedrun2.to_csv(cleaned_outfilename(outfilename))


    def merge_telescopes(csvfname1, csvfname2=None):
        if csvfname2 == None:
            if "_M1_" in csvfname1:
                csvfname2 = csvfname1.replace("_M1_", "_M2_")
            else:
                csvfname2 = csvfname1.replace("_M2_", "_M1_")

        cfile1 = pd.read_csv(csvfname1)
        cfile2 = pd.read_csv(csvfname2)
        maxlen = np.max([len(cfile1), len(cfile2)])
        mergedcfile = cfile1.merge(right=cfile2, how="left", on="id", indicator=True)
        mergedcfile = self._check_and_correct_lens(mergedcfile, maxlen)
        mergedcfile = mergedcfile[mergedcfile["_merge"] == "both"]
        mergedcfile = mergedcfile.drop("_merge")
        # telescopes might need to get merged first to remove events that only trigger one telescope
        return mergedcfile


    def get_mergestatuses(self, calibfilenames, starmergecolfnames):
        mergestatus = []
        assert len(calibfilenames) == len(starmergecolfnames), "Number of calibrated and starfiles is not the same, merging and slicing the nnfile wont work, aborting."
        print("len starframe:", len(starmergecolfnames))

        calibfilenames = sorted(calibfilenames)
        starmergecolfnames = sorted(starmergecolfnames)
        for fCalib, fStar in tqdm(zip(calibfilenames, starmergecolfnames), total=len(starmergecolfnames)):
            print(fCalib, fStar)
            # pdCalib = pd.read_csv(fCalib)
            # pdStar = pd.read_csv(fStar)
            # for some reason every .csv file contains the last event twice which makes it appear 4 times after merging, two of the four duplicates have to be removed
            # maybe dont remove because the same thing happens with the merging in the hdf5 preprocessing? need to find out
            pdCalib = self.merge_telescopes(fCalib)
            pdStar = self.merge_telescopes(fStar)
            mergevalues = pdCalib.merge(right=pdStar, how="left",
                                        indicator=True)['_merge'].values
            mergevalues = self._check_lens(mergevalues, len(pdCalib))
            mergestatus.extend(mergevalues)
        return np.array(mergestatus)


    def get_mask_from_statuses(self, mergestatus):
        filtermask = np.array(mergestatus == "both")
        if self.invert is True:
            filtermask = ~filtermask
        return filtermask


    def get_filtermask(self, calibmcolfnames=None):
        """Compares the calibrated and star files on an event by event basis and returns a mask that has the same length as the calibrated file. filtermask is True for every event that exists in both files."""
        if calibmcolfnames is None:
            calibmcolfnames = self.calibmergecolsdir
        # featuresStar = [pd.read_csv(sfeat, sep=',') for sfeat in self.starmergecols]
        # featuresCalibrated = [pd.read_csv(cfname, sep=',') for cfname in calibfilenames]
        print("getting mergestatus")
        print(np.array(calibmcolfnames).shape)
        print(np.array(self.starmergecols).shape)
        mergestatus = self.get_mergestatuses(calibfilenames, self.starmergecols)
        filtermask = self.get_mask_from_statuses(mergestatus)

        return np.array(filtermask).flatten()


    def _get_newfilename(self, oldname=None, mode="filter"):
        """Takes the old path and returns it with the filename prepended with `eventfiltered_`"""
        if mode == "filter":
            if oldname is None::
                oldname = self.tofilterdir
            newbasename = f"eventfiltered_{self.filtertype}_" + path.basename(oldname)
        elif mode =="runmerge":
            # example filename: 20160929_M1_05057144.015_I_CrabNebula-W0.40+215_mergecols.csv
            splitname = oldname.split("_")
            splitname[2] = splitname[2].split(".")[0]
            newbasename = "runmerge_" + "_".join(splitname)

        newfilename = path.join(path.dirname(oldname), newbasename)
        return newfilename


    def filter_nn_estimation(self):
        """Removes the rows that don't conform to the filtermask and saves back to .txt"""
        print("getting nnfile")
        nnFile = pd.read_csv(self.tofilterdir, 'r')
        print(nnFile.shape)
        print("getting filtermask")
        filtermask = self.get_filtermask()
        print(filtermask.shape)
        assert len(nnFile) == len(filtermask), f"Filtermask and nnFile dont have the same length. Filtermask: {len(filtermask)} nnFile: {len(nnFile)}"
        newfilename = self._get_newfilename(self.filtername)
        filteredDf = nnFile.iloc[filtermask]
        np.savetxt(filteredDf, newfilename)
        self.outfilename = newfilename


    def filter_multifile_hdf5(self):
        """Filters multiple hdf5 files with the filtermask and saves to new hdf5 files."""
        hdf5filenames = sorted(glob_and_check(path.join(self.tofilterdir, "/*")))
        if "*" in self.calibmergecolsdir:
            calibmergecolfnames = sorted(glob_and_check(self.calibmergecolsdir))
        else:
            calibmergecolfnames = sorted(glob_and_check(path.join(self.calibmergecolsdir, "/*")))
            Parallel(n_jobs=1)(delayed(self.filter_single_hdf5)(filename, calibfile)
                               for filename, calibfile in zip(hdf5filenames, calibmergecolfnames))
        self.outfilename = self.tofilterdir


    def filter_single_hdf5(self, hdf5filename=None, calibmergecolfile=None):
        """Filters the hdf5 file with the filtermask and saves to a new hdf5 file"""
        if (hdf5filename is None) and (calibmergecolfile is None):
            hdf5filename = self.tofilterdir
            calibmergecolfile = self.calibmergecolsdir
        hdf5file = h5.File(hdf5filename, 'r')
        hdfdata = np.array(hdf5file["data"])
        filtermask = self.get_filtermask(calibmergecolfile)
        hdfdata = hdfdata[filtermask]
        newfilename = self._get_newfilename(hdf5filename)
        newhdf5File = h5.File(newfilename, 'w')
        try:
            newhdf5File.create_dataset("data", data=hdfdata)
        finally:
            newhdf5File.close()
        self.outfilename = newfilename


    def remove_events(self):
        """Removes the events that don't conform to the `filtermask` and saves to a new file."""
        if self.filtertype == "NN":
            self.filter_nn_estimation()
        elif self.filtertype == "HDF5":
            if self.multifile is False:
                self.filter_single_hdf5()
            else:
                self.filter_multifile_hdf5()
        elif self.filtertpe == "SS":
            self.cleaning_filter()


    def __call__(self):
        self.remove_events()


    def invert_selection():
        """Selects and writes a new csv file with all events that do not conform to the filtermask
        (which means they did get filtered out from calibrated->star or whatever step)"""
        pass


@click.command()
@click.option('--tofilterdir', "-hd", default="./", type=click.Path(),
              help='Dir from which to read the hdf5files or nn outputfile which is going to get filtered.')
@click.option('--starmergecols', "-sf", default="./starmergecols.csv", type=click.Path(),
              help='File from which to read the features (mergecols) for filtering events. Generated after image cleaning (star).')
@click.option('--superstarfeatures', "-ssf", default="./superstarfeatures.csv", type=click.Path(),
              help='File from which to read the features (mergecols) for filtering events. Generated after merging (superstar).')
@click.option('--calibmergecolsdir', "-cf", type=click.Path(),
              help='Path from which to read the features (mergecols) for filtering events. Generated before image cleaning (calibrated).')
@click.option('--multifile', "-mf", default="True", type=bool,
              help='If True filterdir is a directory that gets globbed, if false it is a single hdf5file.')
@click.option('--filtertype', "-ft", default="SS", type=click.Choice(['NN', 'HDF5', 'SS']),
              help='What type of file to filter.')
def main(**args):
    Remover = EventRemover(**args)
    Remover.remove_events()


if __name__ == "__main__":
    main()
