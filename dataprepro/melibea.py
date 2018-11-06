from subprocess import call
from os import path
import os
import click
import numpy as np
from joblib import Parallel, delayed
from csv2hdf5 import glob_and_check
from remove_events import get_run

# supposed to be called from the same dir where superstar etc subdirs are in
# otherwise change constants

SCRIPTDIR = path.join(path.dirname(__file__), "")
MELIBEAGLOBAL = "/opt/MAGIC/Mars_V2-18-4/melibea"
MELIBEA = "/net/big-tank/POOL/users/phoffmann/software/testmars176/Mars_V2-17-9/melibea"
MODIFIERS = "-f -q -b --stereo"
MCMODIFIERS = "-f -q -b -mc --stereo"
CONFIG = '--config=rc/melibea_stereo.rc'
LOG = '--log=melibea/melibea.log'
HADRONRF = '--rf --rftree=coach/RF.root'
ETAB = '-erec --etab=coach/Energy_Table.root'
DISP = ('--calcstereodisp --calc-disp-rf --calc-disp2-rf '
        '--disp-rf-sstrained --rfdisptree=coach/disp1/DispRF.root --rfdisp2tree=coach/disp2/DispRF.root')
ENRF = '--calc-enr --rfentree=coach/EnRF.root'

STANDARD = MODIFIERS.split(" ")
STANDARD.extend([CONFIG, LOG])
MCSTANDARD = MCMODIFIERS.split(" ")
MCSTANDARD.extend([CONFIG, LOG])


def setup_dirs(outdir):
    nnpath = path.join(outdir, "nn")
    regpath = path.join(outdir, "regular")
    os.makedirs(nnpath, exist_ok=True)
    os.makedirs(regpath, exist_ok=True)


def get_inglob(outdir):
    dirname = path.dirname(outdir)
    run = get_run(outdir)
    newbasename = f"*{run}*.root"
    newglob = path.join(dirname, newbasename)
    glob_and_check(newglob)
    return newglob


def get_nn_fargs(nnfile):
    NN = f'--nnoutputs={nnfile}'
    fargs = [MELIBEA]
    fargs.extend(STANDARD)
    fargs.extend(HADRONRF.split(" "))
    fargs.extend(ENRF.split(" "))
    fargs.append(NN)
    # fargs.extend(DISP.split(" "))
    return fargs

def get_lut_fargs(mode):
    fargs = [MELIBEAGLOBAL]
    if "mc" in mode:
        fargs.extend(MCSTANDARD)
    else:
        fargs.extend(STANDARD)
    fargs.extend(HADRONRF.split(" "))
    fargs.extend(DISP.split(" "))
    fargs.extend(ETAB.split(" "))
    return fargs

def get_second_fargs(nnfile, mode):
    "first pass has to be done with LUT and then the NN"
    # fargs = [MELIBEAGLOBAL]
    # fargs.extend(STANDARD)
    # fargs.extend(DISP.split(" "))
    NN = f'--nnoutputs={nnfile}'
    fargs = [MELIBEA]
    if "mc" in mode:
        fargs.extend(MCSTANDARD)
    else:
        fargs.extend(STANDARD)
    fargs.extend(ENRF.split(" "))
    fargs.append(NN)
    return fargs

def move_to_subdir(ssfile):
    olddir, fname = path.split(ssfile)
    subdir = fname.split(".root")[0]
    newdir = path.join(olddir, subdir)
    newfname = path.join(newdir, fname)
    os.makedirs(newdir, exist_ok=True)
    os.rename(ssfile, newfname)
    return newfname

def move_back_from_subdir(ssfile):
    olddir, subdir = path.split(path.dirname(ssfile))
    oldname = path.join(olddir, path.basename(ssfile))
    os.rename(ssfile, oldname)
    os.rmdir(path.join(olddir, subdir))
    print("moved back")


def run_melibea(ssfile, nnfile, outdir, mode):
    assert_filenames(ssfile, nnfile)
    try:
        # inglob = get_inglob(ssfile)
        newfname = move_to_subdir(ssfile)
        inglob = path.join(path.dirname(newfname), "*")
        IND = f'--ind={inglob}'
        OUTDIR = f'--out={outdir}'

        print(mode)
        print("nnfile ",nnfile)
        if "lut" in mode:
            fargs = get_lut_fargs(mode)
        elif "nn" in mode:
            fargs = get_nn_fargs(nnfile)
        elif "second" in mode:
            fargs = get_second_fargs(nnfile, mode)
        else:
            raise ValueError("mode not found")
        fargs.extend([IND, OUTDIR])

        # from IPython import embed; embed()
        print(fargs)
        call(fargs)
    finally:
        move_back_from_subdir(newfname)

    # return melibeaFname

def move_files(fnames, ids, subdir="setdiff"):
    olddir = path.dirname(fnames[0])
    newdir = path.join(olddir, subdir)
    os.makedirs(newdir, exist_ok=True)
    fnames = [fname for fname in fnames
              if np.any([i in fname for i in ids])]
    newfnames = [path.join(newdir, path.basename(fname)) for fname in fnames]
    [os.rename(fname, newfname) for fname, newfname in zip(fnames, newfnames)]

def assert_filenames(fnames1, fnames2):
    if isinstance(fnames1, list):
        runs1 = set([get_id(fname) for fname in fnames1])
        runs2 = set([get_id(fname) for fname in fnames2])
        if len(fnames1) != len(fnames2):
            f"len1: {len(fnames1)}, len2: {len(fnames2)}, setdiff of runs: {runs1^runs2}"
            move_files(fnames2, runs1^runs2)
            move_files(fnames1, runs2^runs1)
        assert runs1 >= runs2, f"Runs are not identical, aborting.\ {runs2^runs1} \n "
    elif isinstance(fnames1, str):
        assert get_id(fnames1) == get_id(fnames2), \
            f"Runnames are not identical, aborting. run1: {get_run(fnames2)}, run2: {get_run(fnames2)}"

def get_id(fname):
    if "_Q_" in fname:
        splitter = "_Q_"
    elif "_Y_" in fname:
        splitter = "_Y_"
    elif "_S_" in fname:
        splitter = "_S_"
    else:
        raise NotImplementedError(fname)
    fid = fname.split(splitter)[0].split("_")[-1]
    return fid

def make_fnames_equal(fnames, qfnames):
    fnames = np.asarray(sorted(fnames))
    qfnames = np.asarray(sorted(qfnames))

    ids = [get_id(f) for f in fnames]
    #  './melibea/lut/mergecols/GA_za05to35_8_1737992_Q_wr_mergecols.csv',
    qids = [get_id(f) for f in qfnames]
    mask = np.isin(ids, qids)
    qmask = np.isin(qids, ids)
    return fnames[mask], qfnames[qmask]


@click.command()
@click.option('--superstar', "-ss", default="./superstar/*_S_*.root",
              type=click.Path(resolve_path=True),
              help='RootDir glob from which to read the files')
@click.option('--nnenergies', "-nn", default="./nnenergies/*adjusted*.txt",
              type=click.Path(resolve_path=True),
              help='RootDir glob from which to read the files')
@click.option('--outdir', "-od", default="./melibea",
              type=click.Path(resolve_path=True),
              help='Outdir where the files get written to.')
@click.option('--njobs', "-nj", default=1, type=int,
              help='Number of jobs for parallel processing')
@click.option('--mode', "-mo", default="nn", type=click.Choice(["nn", "lut", "second", "mclut", "secondmc"]),
              help='"Wether to do LUT energies only, nn first pass, or nn second pass (obligatory for nn if you want valid StereoDispParams), mclut for processing mc files with lut energies. mclut leads to segfault idk why')
def main(superstar, nnenergies, outdir, njobs, mode):
    process_melibea(superstar, nnenergies, outdir, njobs, mode)


def process_melibea(superstar, nnenergies, outdir, njobs, mode):
    ssfiles = glob_and_check(superstar)
    if "nn" in mode or "second" in mode:
        if isinstance(nnenergies, str):
            nnfiles = glob_and_check(nnenergies)
        else:
            nnfiles = sorted(nnenergies)
        assert np.all([".root" not in fname for fname in nnfiles]), f"nnenergy files are not supposed to be rootfiles, aborting. nnglob: {nnenergies}"
        assert_filenames(ssfiles, nnfiles)
    else:
        nnfiles = ssfiles

    assert np.all([".root" in fname for fname in ssfiles]), f"superstar files are supposed to be rootfiles, aborting. superstarglob: {superstar}"

    newssfiles, newnnfiles = make_fnames_equal(ssfiles, nnfiles)
    assert len(newnnfiles) >= len(ssfiles), "Not all ssfiles get processed, something is wrong"
    setup_dirs(outdir)
    filezip = zip(ssfiles, nnfiles)
    Parallel(n_jobs=njobs)\
        (delayed(run_melibea)(ssfile, nnfile, outdir, mode) for ssfile, nnfile in filezip)

    # return melibeaFnames


if __name__ == '__main__':
    main()
