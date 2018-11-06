from os import path
from functools import partial
from joblib import Parallel, delayed
import click
from root2csv import MERGEDCSVPREFIX
from remove_events import EventRemover
from csv2hdf5 import glob_and_check
from root2csv import converter
from root2csv import check_telescope_files

yecho = partial(click.secho, fg="yellow")

def get_identifier(fname):
    if "_Y_" in fname:
        glob = "*_Y_*"
    elif "_I_" in fname:
        glob = "*_I_*"
    elif "_S_" in fname:
        glob = "*_S_*"
    else:
        glob = "*"
    if ".root" in fname:
        glob += ".root"
    elif ".csv" in fname:
        glob += ".csv"
    return glob

def split_by_dates(glob):
    if isinstance(glob, str):
        fullnames = glob_and_check(glob)
    else:
        fullnames = glob
    dirnames = [path.dirname(f) for f in fullnames]
    fnames = [path.basename(f) for f in fullnames]
    if not fnames[0].startswith("GA_"):
        dates = set([fname.split("_")[0] for fname in fnames])
    else:
        # example GA_M2_za05to35_8_1740969_Y_wr.root
        # first 5 numbers of run for splitting
        if "_S_" not in fnames[0]:
            dates = set(["*" + f.split("_")[4][:5] for f in fnames])
        # example GA_za05to35_8_1740969_Y_wr.root
        else:
            dates = set(["*" + f.split("_")[3][:5] for f in fnames])
    dates = sorted(list(dates))
    identifiers = [get_identifier(fname) for fname in fnames]
    newglobs = [date + identifier for date, identifier in zip(dates, identifiers)]
    newglobs = [path.join(dname, glob) for dname, glob in zip(dirnames, newglobs)]

    return sorted(newglobs)

def get_glob_strings(subdirglob):
    """Returns the globstrings for getting M1 and M2 files out of the subdirectory, doesnt contain the descend into the subdir anymore.
       `subdirglob` is a globstring into a subdir relative to the basedir."""
    dirname = path.dirname(subdirglob)
    basename = path.basename(subdirglob)
    assert ((("_M1_" in subdirglob) or ("_M2_" in subdirglob)) or ("_S_" in subdirglob)), \
           ("_M1_ or _M2_ not in subdirglob, cant differentiate between M1 and M2, aborting."
            f"glob: {subdirglob}")
    if ("*" not in subdirglob) and ("_S_" not in basename):
        newbasename = basename.replace("_M2_", "_M1_"), basename.replace("_M1_", "_M2_")
        return path.join(dirname, newbasename[0]), path.join(dirname, newbasename[1])
    elif ("_M1_" or "_M2_") in basename:
        newbasename =  basename.replace("_M2_", "_M1_"), basename.replace("_M1_", "_M2_")
        return path.join(dirname, newbasename[0]), path.join(dirname, newbasename[1])
    elif "_S_" in basename:
        return basename

def split_filenames(filenames):
    """Splits the filenames into those from Telescope M1 and M2"""
    fm1 = [fname for fname in filenames if "_M1_" in fname]
    fm2 = [fname for fname in filenames if "_M2_" in fname]
    return fm1, fm2

def get_dir_from_glob(basedir, globstr):
    """Returns the complete directory of the globstr when given basedir and the globstr relative to basedir."""
    directory = path.abspath(basedir) + path.dirname(globstr)
    directory = path.join(directory, "")
    return directory

def merge_wrapper(processdir, basedir, starglob, superstarglob, calibrootglob, njobs=2, invert=False):
    """extracts the mergecols from the _S_ root (superstarglob) files and merges the energies"""
    for glob in [starglob, superstarglob, calibrootglob]:
        assert path.dirname(glob), \
               f"Glob : {glob} should be/contain a subdirectory"

    superstarGlobNew = get_glob_strings(superstarglob)
    calibrootGlob1, calibrootGlob2 = get_glob_strings(calibrootglob)
    superstardir = get_dir_from_glob(processdir, superstarglob)
    calibdir = get_dir_from_glob(basedir, calibrootglob)
    starglob = processdir + starglob

    # ssmcolfnames = converter(superstardir,
                             # globstr1=superstarGlobNew,
                             # globstr2=superstarGlobNew,
                             # njobs=42,
                             # mergecolsonly=True)
    # yecho("SuperStarfiles done.")
    # tofiltercalibglob = converter(processdir,
                                  # globstr1=calibrootGlob1,
                                  # globstr2=calibrootGlob2,
                                  # njobs=42,
                                  # mergecolsonly=False)
    # yecho("Extracting done.")
    tofiltercalibglob = "./csv/*.csv"
    ssmcolfnames = glob_and_check("./superstar/mergecols/*.csv")

    yecho("Removing events.")
    if njobs > 1:
        splitcalib = split_by_dates(tofiltercalibglob)
        splitstar = split_by_dates(starglob)
        splitss = split_by_dates(ssmcolfnames)
        # needs filename output
        assert len(splitcalib) == len(splitstar) == len(splitss), "only works the first time when no calibfiles got moved, for everything else this needs a new function with more logic"
        Parallel(n_jobs=njobs)\
                           (delayed(single_remove_events)(calibglob, starglob, ssglob, njobs, invert)
                              for calibglob, starglob, ssglob in zip(splitcalib, splitstar, splitss))
        # filteredFiles = [f for arr in filteredFiles for f in arr]
    else:
        check_telescope_files(rootdir=None, globstr1=ssmcolfnames,
                              globstr2=calibmcolfnames, replacer=("_Y_", "_I_"))
        remover = EventRemover(tofiltercalibglob=tofiltercalibglob,
                               starglob=starglob,
                               superstarmcolglob=ssmcolfnames)
        remover.remove_events()
        filteredFiles = remover.outfilenames
    yecho("Removed events that get thrown out during image cleaning and superstar processing and wrote the merged runs to:")
    yecho(f"{path.basename(filteredFiles[0])}")
    # return filteredFiles

def single_remove_events(tofiltercalibglob, starglob, ssmcolfnames, njobs, invert):
    remover = EventRemover(tofiltercalibglob=tofiltercalibglob,
                           starglob=starglob,
                           superstarmcolglob=ssmcolfnames,
                           njobs=njobs, invert=invert)
    remover.remove_events()
    # filteredFiles = remover.outfilenames
    # return filterdFiles


@click.command()
@click.option('--processdir', '-pd', default=None, type=click.Path(),
              help='Directory of the mars processing chain, which contains the dirs to the star and superstar files.')
@click.option('--basedir', '-bd', default=None, type=click.Path(),
              help='Directory of the calibrated files.')
@click.option('--starglob', '-sg', default="/star/*_I_*.root", type=click.Path(),
              help='Dir of the root files after image cleaning. Relative to processdir')
@click.option('--superstarglob', '-ssg', default="/superstar/*_S_*.root", type=click.Path(),
              help='Dir of the root files after superstar processing. Relative to processdir')
@click.option('--calibrootglob', '-cg', default="/root/*_Y_*.root", type=click.Path(),
              help='Dir of the calibrated root files. Relative to processdir')
@click.option('--njobs', default=2, type=int,
              help='Number of jobs for parallel processing. Defaults to 2. Needs about 35GB RAM per process (5GB files). Only 120MB for MC files')
@click.option('--invert', default=False, type=bool,
              help='Whether to invert the filter.')
def main(**args):
    try:
        merge_wrapper(**args)
    except:
        import pdb, traceback, sys
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
