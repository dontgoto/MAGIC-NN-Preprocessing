### apply the normalisation (acquired on MC-simulations) to the data. For documentation, see the MC-equivalent
from functools import partial
from os import path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py as h5
from dataprepro import csv2hdf5 as cth
# import dask.array as da

logger = logging.getLogger(__name__)
tqdm = partial(tqdm, mininterval=30.0)

NORMPREFIX = "normalized_"
PIXELFNAME = "pixelwise_norm.txt"
MERGEDFNAME = "mean_pixelwise_datafordata_norm.txt"
DATAFORMCNORM = "mean_pixelwise_dataformc_norm.txt"
OLDDIR = "/telmerged/"
NEWDIR = "/normalized/"
MEAN_MC_NORMNAME = "mean_pixelwise_mc_norm.txt"
MEAN_DATA_NORMNAME = "mean_pixelwise_data_norm.txt"
STANDARD_NORMFILENAME = "pixelwise_norm.txt"

def write_pixelnormfile(mean, std_dev, opath, fname=PIXELFNAME):
    outfname = path.join(opath, fname)
    df = pd.DataFrame(np.vstack([mean, std_dev]).T)
    logger.info(f"saving to {path.abspath(outfname)}")
    df.to_csv(outfname, header=None, index=None, mode="w", sep=" ")
    return outfname

def generate_pixelnormfile(infilename, opath, fname=PIXELFNAME):
    h5file = h5.File(infilename, "r")
    if cth.get_shape(h5file)[0] > 5*10**5:
        indices = np.random.choice(range(0,cth.get_shape(h5file)[0]),
                                   5*10**5, replace=False)
        data = h5file["data"].value[indices,::]
    else:
        data = h5file["data"].value #read the dataset
    basename = path.basename(infilename)
    # logger.debug("number of nans before norm: ", np.sum(np.isnan(data.flatten())))
    if "GA_" in basename:
        for i in tqdm(range(0, data.shape[0])):
            data[i][0] = np.log(data[i][0]) #logarithmize the energy
    else:
        # if the norm is generated from real data, cut of the eventlinking
                data = data[::,:-3]
    # logger.debug("number of nans before norm: ", np.sum(np.isnan(data.flatten())))
    mean = np.mean(data, axis=0, dtype='float64') #calculate the mean for all features
    std_dev = np.std(data, axis=0, dtype='float64') #calculate the standard deviation for all features

    normfile = write_pixelnormfile(mean, std_dev, opath, fname=fname)
    return normfile, mean, std_dev
    # needs to be read from path.join(path.dirname(path.dirname(infilename)), PIXELFNAME)

def add_mean_mcenergy(mean_mean, mean_std_dev, mcnormfile):
    mcnorm = pd.read_csv(mcnormfile, sep=" ", header=None)
    mcnorm.columns = ['mean', 'std']
    mean_mean = np.insert(mean_mean, 0, mcnorm["mean"].values[0])
    mean_std_dev = np.insert(mean_std_dev, 0, mcnorm["std"].values[0])
    return mean_mean, mean_std_dev

def generate_normfiles(glob, opath, mcnormfile=PIXELFNAME):
    infilenames = cth.glob_and_check(glob)
    means, std_devs = [], []

    for infilename in tqdm(infilenames):
        fname = path.basename(infilename) + "_norm.txt"
        normfile, mean, std_dev = generate_pixelnormfile(infilename, opath, fname)
        means.append(mean)
        std_devs.append(std_dev)
    means = np.array(means)
    std_devs = np.array(std_devs)
    assert means.shape == std_devs.shape
    mean_mean = np.nanmean(means, axis=0)
    mean_std_dev = np.nanmean(std_devs, axis=0)
    pathname = path.dirname(path.dirname(infilenames[0]))

    if "GA" in infilenames[0]:
        write_pixelnormfile(mean_mean, mean_std_dev, opath, fname=MEAN_MC_NORMNAME)
    else:
        mean_mean, mean_std_dev = add_mean_mcenergy(mean_mean, mean_std_dev, mcnormfile)
        write_pixelnormfile(mean_mean, mean_std_dev,
                            opath, fname=MEAN_DATA_NORMNAME)

def normalize(infilePath, normfile=None, outdir=None):
    if normfile is None:
        normfile = STANDARD_NORMFILENAME
        # normfile = path.join(path.dirname(path.dirname(infilename)), PIXELFNAME)

    f2 = pd.read_csv(normfile, sep=" ", header=None, names=["mean", "std"])
    mean = f2["mean"].values
    std = f2["std"].values
    data = h5.File(infilePath)["data"].value.astype("float16")

    if "GA_" in path.basename(infilePath):
        diff = 0
        data[::,0] = np.log(data[::,0])
        upper = data.shape[1] -3
    # if not mc file ignore the mc energy at index 0 in norm.txt
    else:
        diff = 1
        upper = f2.shape[0] -3
# -2 for excluding az and zd
    # for i in range(diff,upper-2):
        # if std[i] != 0:
            # testdata[:,i-diff] -= mean[i]
            # testdata[:,i-diff] /= std[i]

    infmaskShort = np.isinf(mean) & np.isinf(std) & np.isnan(mean) & np.isnan(std)
    infmask = np.zeros(data.shape[1], dtype=bool)
    infmask[:len(infmaskShort)] = infmaskShort

    stdmask = (std == 0)
    std[stdmask] = 1.
    mean[stdmask] = 0.
    data[::,0:upper-diff-2] -= mean[diff:upper-2]
    data[::,0:upper-diff-2] /= std[diff:upper-2]
    data[::,infmask] = 0.

    # np.testing.assert_array_almost_equal(testdata ,data, decimal=5)

    data[::,-1] = (data[::,-1]-(360./2))/105.
    data[::,-2] = (data[::,-2]-(45.))/26.

    newBasename = NORMPREFIX + path.basename(infilePath)
    newDirname = ""
    if outdir is None:
        newDirname = path.dirname(infilePath).replace(OLDDIR, NEWDIR)
        newDirname = path.dirname(infilePath).replace(OLDDIR, NEWDIR)
    else:
        newDirname = outdir
    outfilePath = path.join(newDirname, newBasename)
    assert outfilePath != infilePath

    np.nan_to_num(data, copy=False)
    outfilename = cth.create_hdf5_from_dataset(data, outfilePath)
    logger.info('normalized ', path.basename(outfilePath))
    del data
    return outfilename
