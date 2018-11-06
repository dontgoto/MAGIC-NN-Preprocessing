import os
from os import path
from IPython import embed
import click
import pandas as pd
import numpy as np
from root2csv import converter
from csv2hdf5 import glob_and_check
from remove_events import get_run
from melibea import process_melibea


def check_globs(glob1, glob2):
    if isinstance(glob1, str):
        fnames1 = glob_and_check(glob1)
        fnames2 = glob_and_check(glob2)
    elif isinstance(glob1, list):
        fnames1 = glob1
        fnames2 = glob2
    runs1 = sorted([get_run(fname) for fname in fnames1])
    runs2 = sorted([get_run(fname) for fname in fnames2])
    assert np.array_equal(runs1, runs2), "runs are not equal, aborting."


def generate_imputed_luts(basedir, superstar, lutfnames):
    """processes superstar with melibea, taking the energies from the lutfnames text files for imputation
       the extracts the lutenergies from the newly imputed files and returns their filenames"""
    outdir = path.join(basedir, "tmp")
    os.makedirs(outdir, exist_ok=True)
    process_melibea(superstar=superstar,
                    nnenergies=lutfnames,
                    outdir=outdir,
                    njobs=10,
                    mode="nn")
    glob_and_check(path.join(outdir, "*_Q_*.root"))
    imputedluts = path.join(outdir, "*_Q_*.root")
    lutenergies = converter(outdir,
                            globstr1=path.basename(imputedluts),
                            globstr2=path.basename(imputedluts),
                            multidir=False,
                            njobs=8,
                            mergecolsonly=True,
                            parallelprocessing=True)
    check_globs(lutenergies, lutfnames)

    return sorted(lutenergies)


def compare_energies(basedir, superstar, melibealut):
    """Extracts the energies from melibealut imputes them with the nn into new root files and
       extracts the energies from those and then compares the resulting energies"""
    lutdir = path.dirname(melibealut)
    try:
        lutfnames = converter(lutdir,
                              globstr1=path.basename(melibealut),
                              globstr2=path.basename(melibealut),
                              multidir=False,
                              njobs=8,
                              mergecolsonly=True,
                              parallelprocessing=True)
        lutfnames = sorted(lutfnames)
        for fname in lutfnames:
            energy = pd.read_csv(fname)["ELUT"]
            energy.to_csv(fname, header=False, index=False)

        imputedluts = generate_imputed_luts(lutdir, superstar, lutfnames)

        allequal = True
        for ilutfname, normallut in zip(imputedluts, lutfnames):
            normallutenergies = pd.read_csv(normallut, header=None).values.flatten()
            ilutenergies = pd.read_csv(ilutfname)["ELUT"].values
            ilen = len(ilutenergies)
            np.testing.assert_array_almost_equal(ilutenergies, normallutenergies[:ilen],
                                                 decimal=2)
            if not np.array_equal(ilutenergies, normallutenergies[:ilen]):
                embed()
                allequal = False
            else:
                print("files are equal, continuing")
        if allequal:
            print("All files are equal. Removing energy files")
    except Exception:
        embed()
    finally:
        for f1, f2 in zip(lutfnames, imputedluts):
            os.remove(f1)
            os.remove(f2)


@click.command()
@click.option('--superstar', "-ss", default="./superstar/*_S_*.root", type=click.Path(),
              help='Glob for the superstar files that will get processed with melibea.')
@click.option('--melibealut', "-ml", default="./melibea/*_Q_*.root", type=click.Path(),
              help='Glob for the melibea files contain the standard LUT energies')
@click.option('--basedir', "-bd", default="./",
              type=click.Path(resolve_path=True),
              help='')
def main(basedir, superstar, melibealut):
    check_globs(melibealut, superstar)
    compare_energies(basedir, superstar, melibealut)


if __name__ == '__main__':
    main()
