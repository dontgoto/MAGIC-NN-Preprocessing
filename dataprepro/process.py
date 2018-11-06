import os
from os.path import join
from os import path
from functools import partial
import logging

from joblib import Parallel, delayed
import pandas as pd
import click
import numpy as np
from tqdm import tqdm

import mergetelescopes as mt
import csv2hdf5 as cth
import normalisation as norm
from merge_wrapper import merge_wrapper
from magicnn.parallel_apply_model import parallel_applymodel
from adjust_energy import adjust_multiple_files

logging.basicConfig(level=logging.INFO, format='%(name)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%D:%H:%M:%S', )
# logger = logging.getLogger('processing').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

yecho = partial(click.secho, fg="yellow")
PREFIX = "eventfiltered_201"
SCRIPTDIR = join(path.dirname(__file__), "")
PIXELNORMFNAME = "./pixelwise_norm.txt"

HDF = ".hdf5"
CSV = ".csv"
M1MC = "GA_*_M1_*"
M2MC = "GA_*_M2_*"
M1DATA = "*201*_M1_*"
M2DATA = "*201*_M2_*"
M12DATA = "*201*_M[12]_*"
M1 = "*_M1_*"
M2 = "*_M2_*"
M12 = "*_M[12]_*"

DIRECTORIES = ["csv/", "hdf5/", "hdfmerged/",
               "telmerged/", "normalized/", "nnenergies/"]
SUBDIR = "filtered/"
# SUBDIR = ""
DIRECTORIES = [directory+SUBDIR for directory in DIRECTORIES]

# also probably advisable to add a subdir option so everything get processed in csv/subdir/, telmerged/subdir/, so everything is still tidy when doing A/B testing for analyses

# example that shows how to do preprocessing using the functions

def flatten(outfilenames):
    outfilenames = np.array(outfilenames)
    listmask = [isinstance(outfile, (np.ndarray, list))
                for outfile in outfilenames]
    normalfiles = outfilenames[~listmask].tolist()
    filelists = outfilenames[listmask]
    filelist = [f for files in filelists for f in files]
    normalfiles.extend(filelist)
    return np.array(normalfiles)

def splitfiles(files):
    filenamesM1 = [f for f in files if "_M1_" in path.basename(f)]
    filenamesM2 = [f for f in files if "_M2_" in path.basename(f)]
    cth.check_for_equality(filenamesM1, filenamesM2)
    return filenamesM1, filenamesM2


class GeneralPreprocessing():

    def __init__(self, **args):
        self.rootdir = args.pop("rootdir")
        self.root2csv = args.pop("root2csv")
        self.csv2hdf5 = args.pop("csv2hdf5")
        self.mergehdf5 = args.pop("mergehdf5")
        self.mergetelescopes = args.pop("mergetelescopes")
        self.normalize = args.pop("normalize")
        self.normfile = args.pop("normfile")
        if self.normfile is None:
            self.normfile = ""
        self.tillend = args.pop("tillend")
        self.cleanup = args.pop("cleanup")
        self.entrypoint = args.pop("entrypoint")-1
        self.handleenergies = args.pop("handleenergies")
        self.modelpath = args.pop("modelpath")
        self.subdir = args.pop("subdir")
        if self.subdir == "":
            self.subdir = SUBDIR
        self.processingsteps = [self.csv2hdf5, self.mergehdf5,
                                self.mergetelescopes, self.normalize]
        yecho(f"rootdir: {self.rootdir}")
        self.__currentfiles = self.setup_currentfiles(args.pop("startglob", None)) # no globstr implementation for now
        # if currfilesglob is None:
            # self.currentfiles = None
        # else:
            # self.currentfiles = cth.glob_and_check(currfilesglob)
        # self.previousfiles = None

        if self.normalize == 'False':
            self.normalize = False
        elif self.normalize == 'True':
            self.normalize = True

        allTrue = ((self.rootdir and self.csv2hdf5 and self.mergehdf5) is True)
        if allTrue and (self.normalize is not False):
            logger.info("Doing every step.")
        else:
            logger.info(self.__dict__)
        # if self.doAll:
            # for flag in self.__dict__:
                # if type(self.__dict__[flag]) == bool:
                    # self.__dict__[flag] = True
        if self.tillend:
            # find first option that is set to True and set all the ones after that to True as well
            pass

    @property
    def currentfiles(self):
        return self.__currentfiles[self.entrypoint:]

    @currentfiles.getter
    def currentfiles(self):
        if self.__currentfiles is not None:
            return self.__currentfiles[self.entrypoint:]
        else:
            return self.__currentfiles

    @currentfiles.setter
    def currentfiles(self, value):
        self.previousfiles = self.__currentfiles
        self.__currentfiles = value
        self._cleanup()

    # this is not a valid design pattern
    # actually currentfiles should be a class itself with a proper init method and .M1 and .M2 properties
    def get_currentfiles(self, globstr=None):
        if self.__currentfiles is None:
            logger.warning(f"No files provided, fallback to globstr for current processing step: {globstr}")
            if globstr is None :
                logger.error(f"No globstr provided, fallback to globstr provided during script call: {self.startglob}")
                raise LookupError("No currentfiles and glob is invalid")
            else:
                self.currentfiles = self._rootdir_glob(globstr)
        elif (globstr == 1) or (globstr == 2):
            return self.split_currentfiles()[globstr-1]
        return self.currentfiles

    def setup_currentfiles(self, currfilesglob):
        if currfilesglob is None:
            self.__currentfiles = None
        else:
            self.__currentfiles = cth.glob_and_check(currfilesglob)[self.entrypoint:]
            self._cleanup()

    def _rootdir_glob(self, subdir):
        globname = join(self.rootdir, subdir)
        filenames = cth.glob_and_check(globname)
        return sorted(filenames)[self.entrypoint:]

    def _cleanup(self):
        if self.cleanup and self.previousfiles:
            for fname in self.previousfiles:
                os.remove(fname)
        self.entrypoint = 0

    def setup_dirs(self):
        for newdir in DIRECTORIES:
            os.makedirs(join(self.rootdir, newdir), exist_ok=True)
        logger.info("Directory setup done.")

    def split_currentfiles(self):
        if isinstance(self.currentfiles, list):
            filenamesM1, filenamesM2 = splitfiles(self.currentfiles)
        if not len(filenamesM1) == len(filenamesM2):
            raise NotImplementedError("currentfiles not split equally into two telescopes, not going to work")
        return filenamesM1[self.entrypoint:], filenamesM2[self.entrypoint:]

    def _generate_normfile(self):
        yecho("Generating Normfile.")
        mergedfile = cth.merge_all_hdf5_files(self.currentfiles,
                                              outpath=path.dirname(self.currentfiles[0]),
                                              outFilesize=3*10**10)
        # if len(mergedfile) == 1:
            # mergedfile = mergedfile[0]
            # normfile = norm.generate_pixelnormfile(mergedfile, opath=self.rootdir)
        # else:
            # raise NotImplementedError("Generating normfiles from multiple merged files is not implemented yet. Please merge into one big file.")
        mergedfile = mergedfile[0]
        normfile = norm.generate_pixelnormfile(mergedfile, opath=self.rootdir)
        yecho("Normfile generated.")
        return normfile

    @property
    def normfile(self):
        return self.__normfile

    @normfile.getter
    def normfile(self):
        logger.info(self.rootdir)
        if not path.isfile(self.__normfile):
            logger.warning("No valid normfile provided ({self.__normfile}). Now looking in rootdir.")
            self.__normfile = path.join(self.rootdir, PIXELNORMFNAME)
            if not path.isfile(self.__normfile) and ("GA_" in self.currentfiles[0]):
                logger.warning("No valid normfile ({self.__normfile}) in rootdir.")
                self.__normfile = self._generate_normfile()
            elif not path.isfile(self.__normfile) and ("GA_" not in self.currentfiles[0]):
                raise NotImplementedError("no valid MC normfile for normalizing real data")
        logger.info(self.__normfile)
        return self.__normfile

    @normfile.setter
    def normfile(self, value):
        self.__normfile = value

    def filter_and_convertroot2csv(self):
        merge_wrapper(processdir=self.rootdir,
                      basedir=self.rootdir,
                      starglob="/star/*_M1_*_I_*.root",
                      superstarglob="/superstar/*_S_*.root",
                      calibrootglob="/root/*_M1_*_Y_*.root")

    def convertcsv2hdf5(self):
        raise NotImplementedError("This is data/mc specific and a daugther class needs to be used accordingly")

    def mergehdf5files(self):
        yecho("Merging hdf5 files.")
        self.get_currentfiles('csv/'+self.subdir+M12+HDF)
        hdfFilenames1, hdfFilenames2 = self.split_currentfiles()
        filenamesM1 = cth.merge_all_hdf5_files(hdfFilenames1, outpath=join(self.rootdir, "hdfmerged/"+self.subdir))
        filenamesM2 = cth.merge_all_hdf5_files(hdfFilenames2, outpath=join(self.rootdir, "hdfmerged/"+self.subdir))
        filenamesM1.extend(filenamesM2)
        self.currentfiles = filenamesM1
        yecho("Merging done.")

    def mergetelescopefiles(self):
        raise NotImplementedError("This is data/mc specific and a daugther class needs to be used accordingly")

    def normalizefiles(self):
        yecho("Normalizing.")
        self.get_currentfiles(DIRECTORIES[-3]+"*"+HDF)
        if "GA_" in path.basename(self.currentfiles[0]):
            njobs = 30
        else:
            njobs = 3
        self.currentfiles = Parallel(n_jobs=njobs)\
            (delayed(norm.normalize)(ifile, self.normfile, outdir="normalized/"+self.subdir)
             for ifile in tqdm(self.currentfiles))
        yecho("Done normalizing.")

    def get_energies(self,
                     globstr=None,
                     outdir=None):
        yecho("Classifying Energies.")
        if globstr is None:
            globstr = "normalized/"+self.subdir+"normalized*.hdf5"
        if self.currentfiles:
            globstr = self.currentfiles
        self.currentfiles = parallel_applymodel(globstr=globstr,
                                    njobs=20,
                                    outdir=outdir,
                                    modelpath=self.modelpath)
        yecho("Done classifying Energies.")

    def adjust_energies(self, glob=None):
        yecho("Adjusting Energies.")
        if not self.currentfiles:
            if glob is None:
                glob = "*_application.txt"
            path = join(self.rootdir, "nnenergies")
            self.currentfiles = self._rootdir_glob(join(path, glob))
        else:
            path = path.dirname(self.currentfiles[0])
        merge = False
        self.currentfiles = adjust_multiple_files(self.normfile, self.currentfiles, path, merge)

    def preprocess(self):
        self.setup_dirs()
        # currentfiles need to match the first processing step or else bad things will happend without you noticing
        # still needs to be implemented
        # self.setup_currentfiles()
        if self.root2csv:
            self.filter_and_convertroot2csv()
        if self.csv2hdf5:
            self.convertcsv2hdf5()
        if self.mergehdf5:
            self.mergehdf5files()
        if self.mergetelescopes:
            self.mergetelescopefiles()
        if (self.normalize is not False) and (self.normalize != "False"):
            self.normalizefiles()
        if self.handleenergies:
            if self.modelpath is not None:
                self.get_energies()
            self.adjust_energies()

    def __call__(self):
        self.preprocess()


class DataPreprocessing(GeneralPreprocessing):

    def convertcsv2hdf5(self):
        yecho("Converting csv to hdf5.")
        self.get_currentfiles('csv/'+self.subdir+M12DATA+CSV)
        # assuming the subruns are not merged yet this will take 2.5GB per subrun, else x20 ram
        Parallel(n_jobs=1)(delayed(cth.create_hdf5_files)(filenames, "float32")
             for filename in tqdm(self.currentfiles, total=len(self.currentfiles)))
        yecho('Converting done.')

    def mergetelescopefiles(self):
        yecho("Merging telescope files.")
        filenamesM1, filenamesM2 = splitfiles(self.get_currentfiles('csv/'+self.subdir+M12+HDF))
        filenamesOut = [fname.replace('/csv/', '/telmerged/') for fname in filenamesM1]
        filenamesOut = [fname.replace('_M1_', '_') for fname in filenamesOut]
        filezipper = zip(filenamesM1, filenamesM2, filenamesOut)
        # consumes about X GB memory when done on subruns
        self.currentfiles = Parallel(n_jobs=1)\
            (delayed(mt.merge_telescopes)(fM1, fM2, fOut, easy_merge=True)
             for fM1, fM2, fOut in tqdm(filezipper,total=len(filenamesM1)))
        yecho("Merging done.")


class MCPreprocessing(GeneralPreprocessing):

    def convertcsv2hdf5(self):
        yecho("Converting csv to hdf5.")
        self.get_currentfiles(DIRECTORIES[0]+M12+CSV)
        Parallel(n_jobs=15)\
            (delayed(cth.create_hdf5_files)(infileName, "float16")
             for infileName in tqdm(self.currentfiles, total=len(self.currentfiles)))
        yecho("Done converting csv to hdf5.")

    def mergetelescopefiles(self):
        yecho("Merging telescope files.")
        self.get_currentfiles(DIRECTORIES[0]+M12+CSV)
        files1, files2 = self.split_currentfiles()
        self.currentfiles = Parallel(n_jobs=20)\
           (delayed(mt.mc_append_and_merge)
            ([m1], [m2], self.rootdir+DIRECTORIES[-3], easy_merge=True)
            for m1, m2 in tqdm(zip(files1, files2), total=len(files1)))
        yecho("Done merging telescope files.")


@click.command()
@click.option('--rootdir', "-rd", default="./", type=click.Path(file_okay=False, writable=True),
              help='RootDir from which to read the files')
@click.option('--subdir', "-sd", default="", type=click.Path(file_okay=False),
              help='subdir from which to read the files in each processing subdir.')
@click.option('--root2csv', "-rtc", default="False", type=bool,
              help='Convert root to csv or not. Also filters the events. Needs processed star and superstar files.')
@click.option('--csv2hdf5', "-cth", default="True", type=bool,
              help='Convert csv to hdf5 or not.')
@click.option('--mergehdf5', "-mh", default="False", type=bool,
              help='Merge the hdf5files or not.')
@click.option('--mergetelescopes', "-mt", default="True", type=bool,
              help='Merge the telescope files or not.')
@click.option('--normalize', "-no", default="True", type=click.Choice(['True', 'False', 'pixelwise', 'camerawise']),
              help='Whether to merge the telescope files. Defaults to pixelwise. `True` also defaults to pixelwise.')
@click.option('--normfile', "-nf", default=None, type=click.Path(dir_okay=False, resolve_path=True, exists=True),
              help='Location of the normfile.')
@click.option('--tillend', default="False", type=bool,
              help='Whether to process from the first step that is set to True till the end of the chain.')
@click.option('--cleanup', "-c", default="False", type=bool,
              help='Whether to delete old files.')
@click.option('--entrypoint', "-ep", default="1", type=int,
              help='The number of the first file to process in the first processing step. Useful for restarting partially failed preprocessing.')
@click.option('--handleenergies', "-he", default=True, type=bool,
              help='Whether to classify events and unnormalize their energies.')
@click.option('--modelpath', "-mp", default=None, type=click.Path(),
              help='Path to the trained model.ckpt file.')
@click.option('--mode', "-mo", default="data", type=click.Choice(["data", "mc"]),
              help='To preprocess data or monte carlo files.')
@click.option('--mc', "mode", default="data", flag_value="mc",
              help='Flag for processing MC data.')
@click.option('--data', "mode", default="data", flag_value="data",
              help='Flag for processing real data.')
@click.option('--startglob', "-sg", default=None, type=click.Path(),
              help='Glob for reading the first files.')
def main(**args):
    try:
        if args.pop("mode") == "mc":
            preprocessor = MCPreprocessing(**args)
        else:
            preprocessor = DataPreprocessing(**args)
        preprocessor.preprocess()
    except:
        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == '__main__':
    main()
