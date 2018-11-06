from glob import glob
from os import path
import os
from pprint import pprint
import click

# checks for files that are missing in one of each telescopes and removes them
# also checks for files filtered by quate and removes all files not making the quate cuts
@click.command
@click.option('--rootdir', default='./', type=click.Path(),
              help='dir from which to iterate over all dirs each containing a day of MAGIC data.')
@click.option('--remove', default=True, type=bool,
              help='Whether to remove the files that got filtered by quate. Defaults to True')
@click.option('--multidir', default=False, type=bool,
              help='Whether only process a single dir or not')


def get_basenameset(pathList):
    return {path.basename(file) for file in pathList}


def replaceSet(nameSet, old, new):
    return {name.replace(old, new) for name in nameSet}


def remove_files(quateFilterM1):
    for file in quateFilterM1:
        os.remove(quateFilterM1)
        os.remove(quateFilterM1.replace("_M1_", "_M2_"))


def filter_with_quate(rootdir, filterdir='good'):
    """Looks in a directory `rootdir` containing unprocessed MAGIC root files and applies the filters from quate. Returns the basenames of the files that don't make the cuts.

    Parameters:
    -----------

    rootdir : path
        Path to the files that also contains the quate subdir `good`.
    filterdir : path
        Path to the symlinks for the files that made the cuts from quate. Defaults to `good` if the subdir is in the same dir as the original files."""

    quateFilterM1 = glob(path.join(rootdir, subdir, "201*M1*.root"))
    quateFilterM2 = glob(path.join(rootdir, +subdir, "201*M1*.root"))
    allFilesM1 = glob(path.join(rootdir, "201*_M1_*.root"))
    allFilesM2 = glob(path.join(rootdir, "201*_M2_*.root"))
    quateFilterM1 = get_basenameset(quateFilterM1)
    quateFilterM2 = get_basenameset(quateFilterM2)
    allFilesM1 = get_basenameset(allFilesM1)
    allFilesM2 = get_basenameset(allFilesM2)
    quateFilterM1 = quateFilterM1.intersection(replaceSet(quateFilterM2, "M2", "M1"))
    pprint(quateFilterM1)
    print(len(quateFilterM1))
    return quateFilterM1


def handle_dir(rootdir, remove=True):
    quateFilterM1 = filter_with_quate(rootdir)
    if remove == True:
        remove_files(quateFilterM1)


def main(rootdir, remove, multidir=False):
    """Gets a directory where the original root files are. Checks for quates quality cuts and removes all files that don't make the cuts. Optionally iterates over multiple directories for different days.

    Parameters:
    -----------

    rootdir : path
    remove : bool
    multidir : bool
    """
    if multidir == True:
        dirs = glob(path.join(rootdir, "20*_*_*/"))
        for directory in dirs:
           handle_dir(directory, remove=remove)
    else:
        handle_dir(rootdir, remove=remove)



if __name__ == '__main__':
    main()
