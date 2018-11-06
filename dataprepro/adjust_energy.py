from os import path
import re
from glob import glob
import logging

import pandas as pd
import numpy as np
import click
from tqdm import tqdm
from dataprepro.csv2hdf5 import glob_and_check
from magicnn.parallel_apply_model import merge_outfiles

logger = logging.getLogger(__name__)
VALIDATIONSTR = "_validation_val.txt"
APPLICATIONSTR = "_application.txt"
DATACOLS = ['DL', 'eventlinking1', 'eventlinking2', 'eventlinking3']
VALCOLS = ['MCTruth', "DL"]
IMPUTECOL = ["DL"]
IMPUTESUFFIX = '_adjusted_impute.txt'
VALSUFFIX = '_adjusted_validate.txt'
TEXTSUFFIX = '.txt'

def get_mean_std(normfname):
    with open(normfname) as norm:
        if "pixel" in normfname:
            firstLine = norm.readline()
            meanlog, stdlog = firstLine.split(' ')
            meanlog = float(meanlog)
            stdlog = float(stdlog)
        else:
            raise NotImplementedError("doesnt work for camerawise normalization yet")
        logger.debug("meanlog:", meanlog)
        logger.debug("stdlog:", stdlog)
        return meanlog, stdlog

def get_mode(efname):
    mode = "data"
    if "GA_" in efname:
        mode ="mc"
    return mode

def adjust_energy(normFilename, energyFilename):
    mode = get_mode(energyFilename)
    meanlog, stdlog = get_mean_std(normFilename)
    assert path.exists(energyFilename), f"File {energyFilename} doesnt exist, aborting."
    try:
        data = pd.read_csv(energyFilename, sep=" ", dtype='float16', header=None)
    except pd.errors.EmptyDataError:
        logger.info(energyFilename)
        raise
    if mode == "data":
        data.columns = DATACOLS
    else:
        data.columns = VALCOLS
        data['MCTruth'] = (data['MCTruth'] * stdlog) + meanlog
        data["MCTruth"] = np.exp(data['MCTruth'])

    data['DL'] = (data['DL'] * stdlog) + meanlog
    data['DL'] = np.exp(data['DL'])
    return data

def adjust_multiple_files(normFilename, energyFilenames, outpath, merge=False, savemode="imputation"):
    mode = get_mode(energyFilenames[0])
    if mode == "mc":
        logger.info("mc mode removed unnecessary files")
        energyFilenames = [f for f in energyFilenames if VALIDATIONSTR in f]
    else:
        energyFilenames = [f for f in energyFilenames if APPLICATIONSTR in f]
    energyFilenames = [f for f in energyFilenames if not f.startswith("merged_")]
    energyFilenames = [f for f in energyFilenames if not f.endswith(IMPUTESUFFIX)]
    energyFilenames = [f for f in energyFilenames if not f.endswith(VALSUFFIX)]
    energyFilenames = [f for f in energyFilenames if f.endswith(TEXTSUFFIX)]
    energyFilenames = sorted(list(set(energyFilenames)))
    assert energyFilenames
    energies = [adjust_energy(normFilename, eF) for eF in energyFilenames]

    outvalnames = []
    outimputenames = []
    if merge is True:
        outpath = get_newfname(outpath, energyFilenames[0], VALSUFFIX)
        eflat = []
        frame = pd.concat(energies)
        if mode == "mc":
            frame[VALCOLS].to_csv(outpath,
                               float_format='%1.8f',
                               header=VALCOLS,
                               index=False,
                               index_label=False)
        elif mode == "data":
            pass
            # frame[DATACOLS].to_csv(outpath,
                               # float_format='%1.8f',
                               # header=False,
                               # index=False,
                               # index_label=False)
        # for e in energies:
            # eflat.extend(e)
        # np.savetxt(outpath, eflat, fmt='%1.8f')
        return outpath
    else:
        # for e, fname in zip(energies, energyFilenames):
        logger.info("saving single files")
        for frame, fname in tqdm(zip(energies, energyFilenames), total=len(energies)):
            imputename = get_newfname(outpath, fname, IMPUTESUFFIX)
            valname = get_newfname(outpath, fname, VALSUFFIX)
            if mode == "mc":
                frame[VALCOLS].to_csv(valname,
                                      float_format='%1.8f',
                                      header=VALCOLS,
                                      index=False,
                                      index_label=False,
                                      mode="w")
            outvalnames.append(valname)
            frame[IMPUTECOL].replace([np.inf, -np.inf], 0.)\
                .to_csv(imputename,
                        float_format='%1.8f',
                        header=False,
                        index=False,
                        index_label=False,
                        mode="w")
            outimputenames.append(imputename)
        merge_outfiles(outimputenames, remove=False)
        return outvalnames
    logger.info(f"Energies and saved to: \n {outpath}" )

def split_on_number(filename):
    basename = path.basename(filename)
    if filename.startswith("GA_") is False:
        key = re.split(r'(\d+)', basename)[1]
    else:
        raise NotImplementedError("Sorting MC files is not implemented yet")
    return int(key)


def get_newfname(pathname, fname, mstr):
    # GA_M2_za05to35_8_164228_Y_w0.txt
    basename = path.basename(fname)
    if len(basename.split(".")) == 2:
        newbasename = basename.split(".")[0] + mstr
    # normalized_merged_eventfiltered_GA_M1_1737991_TO_1739490_Y_wr.hdf5_validation_val.txt
    elif len(basename.split(".")) == 3:
        if "._I_" not in basename:
            newbasename = basename.split(".")[0] + mstr
        else:
        #  './normalized_20160929_05057148._I_CrabNebula-W0_adjusted_impute.txt'
            newbasename = basename.split("._I_")[0].split("_")[-1] + mstr

    # normalized_eventfiltered_20170119_05059876_I_CrabNebula-W0.40+035.hdf5_application.txt
    elif len(basename.split(".")) == 4 and ("_TO_" not in fname) and ("GA_" not in fname):
        newbasename = basename.split(".")[0] + mstr
    # normalized_20170120_05059934.001_I_CrabNebula-W0.40+215.hdf5_application.txt
    elif len(basename.split(".")) == 5:
        newbasename = (basename.split(".")[0]
                       + "."
                       + basename.split(".")[1]
                       + mstr)
    else:
        raise NotImplementedError(f"idk how to handle files that dont have 1,2 or 4 dots \n file: {fname}")
    ofname = path.join(pathname, newbasename)
    return ofname


@click.command()
@click.option('--outfileglob', default="./*_application.txt", type=click.Path(),
              help='glob that gives all outputfiles containing the energy from the nn')
@click.option('--normfile', default="./pixelwise_norm.txt", type=click.Path(dir_okay=False, resolve_path=True, exists=True),
              help='File that contains the normalization constants')
@click.option('--merge', default=False, type=bool,
              help='Whether to merge the files or not. Default False.')
@click.option('--opath', default=None, type=click.Path(file_okay=False, writable=True, resolve_path=True),
              help='Outpath for the files.')
@click.option('--savemode', "-sm", default="imputation", type=click.Choice(["imputation", "validation"]),
              help='Whether to save for value imputation with melibea or for validation of mc files.')
def main(outfileglob, normfile, opath, merge, savemode):
    if opath is None:
        opath = path.dirname(outfileglob)
    filenames = glob_and_check(outfileglob)
    if len(filenames) > 1:
        filenames.sort(key=split_on_number)
    logger.info([path.basename(filename) for filename in filenames])
    adjust_multiple_files(normfile, filenames, opath, merge, savemode)

if __name__ == '__main__':
    main()
