from dataprepro import csv2hdf5 as cth
from os import path
import os
import click

DIRS = ["test", "train"]

def setup_dirs(indir):
    for newdir in [path.join(indir, directory) for directory in DIRS]:
        os.makedirs(newdir, exist_ok=True)


def get_fnames(indir):
    fnames = cth.glob_and_check(path.join(indir, "*_M*_*.root"))
    m1fnames = [fname for fname in fnames if "_M1_" in fname]
    m2fnames = [fname for fname in fnames if "_M2_" in fname]

    return m1fnames, m2fnames


def check_filenames(indir):
    m1fnames, m2fnames = get_fnames(indir)
    try:
        cth.check_for_equality(m1fnames, m2fnames)
    except NameError:
        newm1fnames = [f.replace("_M1_", "_M2_") for f in m1fnames]
        newm2fnames = [f.replace("_M2_", "_M1_") for f in m2fnames]
        m1fnames = set(m1fnames)
        m2fnames = set(m2fnames)
        newm1fnames = set(newm1fnames)
        newm2fnames = set(newm2fnames)

        missingm1 = newm2fnames - m1fnames
        missingm2 = newm1fnames - m2fnames
        print("\n missing from m1: ", missingm1)
        print("\n missing from m2: ", missingm2)
        for fname in missingm1:
            os.remove(fname)
        for fname in missingm2:
            os.remove(fname)

    m1fnames, m2fnames = get_fnames(indir)
    cth.check_for_equality(m1fnames, m2fnames)
    return m1fnames, m2fnames


def move_files(fnames, subdir):
    olddir = path.dirname(fnames[0])
    newdir = path.join(olddir, subdir)
    basenames = [path.basename(fname) for fname in fnames]
    newfnames = [path.join(newdir, basename) for bname in basenames]
    for fname, newfname in zip(fnames, newfnames):
        os.move(fname, newfname)
    return newfnames


def split_and_move(fnames, percentage_train):
    trainindex = int(len(fnames) * percentage_train) - 1
    trainfiles = fnames[:trainindex]
    testfiles = fnames[trainindex:]
    outtrainfiles = move_files(trainfiles, subdir="train")
    outtestfiles = move_files(testfiles, subdir="test")
    return outtrainfiles, outtestfiles


def train_test_split(m1fnames, m2fnames, precentage_train=0.6):
    """percentage_train is the percentage of train runs of total runs."""

    trainm1, testm1 = split_and_move(m1fnames)
    trainm2, testm2 = split_and_move(m2fnames)
    cth.check_for_equality(trainm1, trainm2)
    cth.check_for_equality(testm1, testm2)


def main(indir):
    setup_dirs(indir)
    m1fnames, m2fnames = check_filenames(indir)
    train_test_split(m1fnames, m2fnames, percentage_train=trainpercentage)


@click.command()
@click.option('--dir', default="./", type=click.Path(),
              help='Directory to the files that get checked.')
@click.option('--trainpercentage', "-tp", default=0.6, type=float,
              help='Percentage of total runs that should be used for training.')
if __name__ == "__main__":
    main()
