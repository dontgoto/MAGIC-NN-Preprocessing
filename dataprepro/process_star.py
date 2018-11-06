from subprocess import call
from os import path
import os
from glob import glob
import click
from joblib import Parallel, delayed

STARVER = '/net/nfshome/home/phoffmann/.local/Mars_V2-17-6-NN/star'
STARVER = 'star'
SCRIPTDIR = path.join(path.dirname(__file__), "")
M1RC = path.join(SCRIPTDIR, "rc/star_M1_OSA.rc")
M2RC = path.join(SCRIPTDIR, "rc/star_M2_OSA.rc")
LOGDIR = path.join(SCRIPTDIR, "star/log/")
os.makedirs(LOGDIR, exist_ok=True)


@click.command()
@click.option('--dates', default=None, type=str,
              help='Dates that will be processed. format mmdd,mmdd,mmdd')
@click.option('--njobs', '-n', default=20, type=int, help='how many jobs for parallel processing')
@click.option('--outdir', '-o', default="./star/", type=click.Path(file_okay=False, exists=True),
              help='outdir for the files')
@click.option('--mode', "-mo", default="data", type=click.Choice(["data", "mc", "mcpixels", "datapixels"]),
              help='Apparently you have to process all mc files at once or else it doesnt find some kind of starguider tree, so this doesnt actually work multiprocessed.\n To preprocess data or monte carlo files.')
@click.option('--mc', "mode", default="data", flag_value="mc",
              help='Flag for processing MC data.')
@click.option('--data', "mode", default="data", flag_value="data",
              help='Flag for processing real data.')
@click.option('--mcpixels', "mode", default="data", flag_value="mcpixels",
              help='Flag for processing MC data and keeping the pixel information.')
@click.option('--datapixels', "mode", default="data", flag_value="datapixels",
              help='Flag for processing real data and keeping the pixel information')
def main(dates, njobs, outdir, mode):
    if "data" in mode:
        if dates is not None:
            dates = dates.split(',')
            dateglob1 = ["./root/201*" + date + "*_M1_*_Y_*.root" for date in dates]
            dateglob2 = ["./root/201*" + date + "*_M2_*_Y_*.root" for date in dates]
        else:
            dateglob1 = ["201*M1*.root"]
            dateglob2 = ["201*M2*.root"]
            print("default")
        logfiles1 = [path.join(LOGDIR, date+"_M1.log") for date in dates]
        logfiles2 = [path.join(LOGDIR, date+"_M2.log") for date in dates]
    elif "mc" in mode:
        dateglob1 = [f"./root/*_M1_*{i}_Y_*.root" for i in range(10)]
        dateglob2 = [f"./root/*_M2_*{i}_Y_*.root" for i in range(10)]
        logfiles1 = [path.join(LOGDIR, f"{i}_M1.log") for i in range(10)]
        logfiles2 = [path.join(LOGDIR, f"{i}_M2.log") for i in range(10)]

    outdir = path.join(path.realpath(outdir), "")


    if "pixels" not in mode:
        fargs1 = [[STARVER, "-b", "-f", f'--config={M1RC}', f'--ind={dateglob}',
                   f'--out={outdir}', f'--log={logname}']
                  for dateglob, logname in zip(dateglob1, logfiles1)]
        fargs2 = [[STARVER, "-b", "-f", f'--config={M2RC}', f'--ind={dateglob}',
                   f'--out={outdir}', f'--log={logname}']
                  for dateglob, logname in zip(dateglob2, logfiles2)]
    else:
        fargs1 = [[STARVER, "-b", "-f", "-savecerevt", f'--config={M1RC}',
                   f'--ind={dateglob}', f'--out={outdir}', f'--log={logname}']
                  for dateglob, logname in zip(dateglob1, logfiles1)]
        fargs2 = [[STARVER, "-b", "-f", "-savecerevt",f'--config={M2RC}',
                   f'--ind={dateglob}', f'--out={outdir}', f'--log={logname}']
                  for dateglob, logname in zip(dateglob2, logfiles2)]
    fargs1.extend(fargs2)
    Parallel(n_jobs=njobs)(delayed(call)(farg) for farg in fargs1)


if __name__ == "__main__":
	main()
