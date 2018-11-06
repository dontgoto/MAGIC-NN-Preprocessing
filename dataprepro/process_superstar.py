from subprocess import call
from os import path
import os
from glob import glob
import click
from joblib import Parallel, delayed
from dataprepro.csv2hdf5 import glob_and_check

SSTARVER = '/net/big-tank/POOL/users/phoffmann/software/testmars176/Mars_V2-17-9/superstar'
SSTARVER = 'superstar'
SCRIPTDIR = path.join(path.dirname(__file__), "")
SSRC = path.join(SCRIPTDIR, "rc/superstar.rc")
LOGDIR = path.join(SCRIPTDIR, "superstar/log/")
os.makedirs(LOGDIR, exist_ok=True)

def get_run_globs(fnames):
    dirnames = [path.dirname(fname) for fname in fnames]
    basenames = [path.basename(fname) for fname in fnames]

    basenames = set([name.split(".")[0] for name in basenames])
    basenames = [name + "*.root" for name in basenames]
    fnames = [path.join(dname, bname) for dname, bname in zip(dirnames, basenames)]
    assert len(fnames) > 0
    return sorted(fnames)

def process_superstar(ind1, ind2, njobs, outdir, mode):
    files1 = glob_and_check(ind1)
    files2 = glob_and_check(ind2)
    assert len(files1) == len(files2)

    outdir = path.join(path.realpath(outdir), "")
    globs1 = get_run_globs(files1)
    globs2 = get_run_globs(files2)
    glob_and_check(globs1)
    glob_and_check(globs2)

    logfiles = [path.join(LOGDIR, fname+".log") for fname in files1]
    if "data" in mode:
        fargs = [[SSTARVER, "-b", "-f", f'--config={SSRC}', f'--ind1={fname1}',
                  f'--ind2={fname2}', f'--out={outdir}', f'--log={logname}']
                  for fname1, fname2, logname in zip(globs1, globs2, logfiles)]
    elif "mc" in mode:
        fargs = [[SSTARVER, "-b", "-f", "-mc", f'--config={SSRC}',f'--ind1={fname1}',
                  f'--ind2={fname2}', f'--out={outdir}', f'--log={logname}']
                  for fname1, fname2, logname in zip(globs1, globs2, logfiles)]
    Parallel(n_jobs=njobs)(delayed(call)(farg) for farg in fargs)


@click.command()
@click.option('--ind1', default="./star/201*_M1_*_I_*.root", type=str,
              help='M1 star files glob')
@click.option('--ind2', default="./star/201*_M2_*_I_*.root", type=str,
              help='M2 star files glob')
@click.option('--njobs', '-n', default=20, type=int, help='how many jobs for parallel processing')
@click.option('--outdir', '-o', default="./superstar/", type=click.Path(file_okay=False, exists=True),
              help='outdir for the files')
@click.option('--mode', "-mo", default="data", type=click.Choice(["data", "mc"]),
              help='Apparently you have to process all mc files at once or else it doesnt find some kind of starguider tree, so this doesnt actually work multiprocessed.\n To preprocess data or monte carlo files.')
@click.option('--mc', "mode", default="data", flag_value="mc",
              help='Flag for processing MC data.')
@click.option('--data', "mode", default="data", flag_value="data",
              help='Flag for processing real data.')
def main(**args):
    process_superstar(**args)


if __name__ == "__main__":
	main()
