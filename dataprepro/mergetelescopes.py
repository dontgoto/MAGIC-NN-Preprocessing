from os import path
from glob import glob
import logging
from functools import partial

import pandas as pd
import h5py
import numpy as np
from dataprepro import csv2hdf5 as cth

MERGEIDS = ("0","1", "6404", "6406")
EMERGEIDS = ("1")
DATAMERGEIDS = ("0", "1", "4")

logger = logging.getLogger(__name__)

def merge_telescopes(filenameM1, filenameM2, filename_out, easy_merge=True):
    logger.info("merging", path.basename(filenameM1))
    assert filenameM1 != filenameM2, "Merging the same telescope files does not make sense, aborting."

    h = h5py.File(filenameM1, 'r')
    data_1 = h['data'].value
    h = h5py.File(filenameM2, 'r')
    data_2 = h['data'].value

    shape = data_1.shape[1]
    order = [str(i) for i in range(2*shape)]

    cols1 = [str(2*i) for i in range(shape)]
    df_1 = pd.DataFrame(data_1, columns=cols1)
    del data_1
    df_1 = df_1.rename(columns={'2': '1'})

    cols2 = [str(2*i+1) for i in range(shape)]
    df_2 = pd.DataFrame(data_2, columns=cols2)
    del data_2
    df_2 = df_2.rename(columns={'1': '0', '3': '1', '5': '4'})

    if easy_merge is False:
        maxlen = max(len(df_1), len(df_2))
        minlen = min(len(df_1), len(df_2))
        logger.info("no easy merge")
        # leads to a loss of 20% of events b/c times are not neccessarily the same for both telescopes
        # please dont do this
        assert df_1.duplicated().sum() == 0, f"duplicates in {filenameM1}"
        assert df_2.duplicated().sum() == 0, f"duplicates in {filenameM2}"
        merged = pd.merge(df_1, df_2, on=DATAMERGEIDS)
        assert len(merged) <= maxlen, \
            ("Merged df is longer than the maximum length of both"
             " telescope files, this should not be possible."
            f"len df1: {len(df_1)}, df2: {len(df_2)}, merged: {len(merged)}")
    else:
        assert len(df_1) == len(df_2),\
            ("lens are not equal, aborting. This means the csv files werent "
             "filtered properly at the start of the processing chain"
            f"len df1: {len(df_1)}, df2: {len(df_2)}")
        lastequal1 = np.all(df_1.iloc[-1] == df_1.iloc[-2])
        lastequal2 = np.all(df_2.iloc[-1] == df_2.iloc[-2])
        assert lastequal1 == lastequal2, \
            "The same rows are not both duplicates/ non duplicate, something is wrong idk what though"

        # leads to 2x2 for the duplicated values -> merged is 2 rows too long
        # if lastequal1:
            # oldmerged = pd.merge(df_1.iloc[:-1], df_2.iloc[:-1], on=("4"))
            # maxlen -= 1
        # else:
            # oldmerged = pd.merge(df_1, df_2, on=("4"))
        # oldmerged = oldmerged.drop(columns=["0_x", "1_x"])
        # oldmerged = oldmerged.rename(columns={'0_y': '0', '1_y': '1'})
        merged = df_2.drop(columns=DATAMERGEIDS, inplace=True)
        merged = pd.merge(df_1, df_2, left_index=True, right_index=True)
        # merged = merged.drop(columns=["0_x", "1_x", "4_x"])
        # merged = merged.rename(columns={'0_y': '0', '1_y': '1', "4_y": "4"})

        # assert np.array_equal(merged.values, oldmerged.values), f"the new merge method doesnt work properly len newmerged: {len(merged), len oldmerged: {len(oldmerged)}}"
        assert len(merged) == len(df_1), \
        ("Merged df is longer than the maximum length of both telescope files, this should not be possible."
         f"len df1: {len(df_1)}, df2: {len(df_2)}, merged: {len(merged)}")
        logger.info(f"merged len is the same as that of each telescope")
    del df_1
    del df_2
    merged = merged.reindex_axis(order, axis=1)

    temp = merged["0"]
    temp2 = merged["1"]
    temp5 = merged["4"]
    temp3 = merged["6406"]
    temp4 = merged["6407"]

    for col in [*DATAMERGEIDS, "2", "3", "5", "6406", "6407", "6408", "6409"]:
        del merged[col]

    merged = pd.concat([merged, temp3, temp4, temp, temp2, temp5], axis=1)
    data_2 = np.array(merged)
    del merged

    h = h5py.File(filename_out, "w")
    h.create_dataset("data", data=data_2)
    h.close()
    assert path.isfile(filename_out)
    logger.info(path.basename(filename_out), " merged")
    del data_2
    return filename_out

def append_seeds(hdfFilenames):
    """Appends the seeds to one array

       hdfFilenames : list
           list of names of hdf5 files that get appended"""

    assert isinstance(hdfFilenames, list), f"filenames not a list, aborting. fnames: {hdfFilenames}"
    hdf5_file = h5py.File(hdfFilenames[0], 'r') # read first .hdf5-file
    data_0 = np.array(hdf5_file['data']) # extract data
    if len(hdfFilenames) > 1:
        for hdfFname in hdfFilenames[1:]:
            hdf5_file = h5py.File(hdfFname, 'r') # repeat for second .hdf5-file
            data_1 = np.array(hdf5_file['data'])
            data_0 = np.append(data_0, data_1, axis = 0) # append
        del data_1
    return data_0

def mc_append_and_merge(filenamesM1, filenamesM2, outpath=None, easy_merge=False):
    """appends the different seeds and merges the resulting telescope arrays. Order of filenames needs to be:
        First all different seeds of Telescope1 then all seeds of Telescope2 in the same order as those of Telescope1.

        filenamesM1 : list
            A list containing the different filenames for all the seeds

    """
    M1 = append_seeds(filenamesM1)
    M2 = append_seeds(filenamesM2)
    assert M1.shape[1] == M2.shape[1], "shapes dont match, aborting. M1 shape: {M1.shape} M2 shape: {M2.shape}"
    logger.debug(f"m1 shape {M1.shape}")
    logger.debug(f"m2 shape {M2.shape}")
    basename = path.basename(filenamesM1[0])
    if outpath is None:
        outfilename = 'merged/merged_' + basename.replace("_M1_", "_")
    else:
        outfilename = path.join(outpath, 'merged_'+path.basename(filenamesM1[0]))
    outfilename = mc_merge_telescopes(M1, M2, outfilename, easy_merge)
    return outfilename

def mc_merge_telescopes(M1, M2, outfilename, easy_merge=True):
    """Takes a two telescope arrays from append_seeds and merges them

        M1 : np.array
            array of all events for a single telescope
        outfilename : str
           path+filename of the resulting merged file"""
    shape = M1.shape[1]
    order = [str(i) for i in range(2*shape)]

    cols1 = [str(2*i) for i in range(shape)]
    cols2 = [str(2*i+1) for i in range(shape)]
    df_1 = pd.DataFrame(M1, columns=cols1) #create DataFrame from first np.array, columns named with numbers (every second column is empty)
    df_1 = df_1.rename(columns={'2': '1'}) #rename second column to first
    df_2 = pd.DataFrame(M2, columns=cols2) #create DataFrame from second np.array, columns named with numbers (every (alternating to the first) second column is empty)
    df_2 = df_2.rename(columns={'1': '0', '3': '1', '6405': '6404', '6407': '6406'}) #rename a few columns, so that the datasets can be merged afterwards

    # merged = pd.merge(df_1, df_2, on=MERGEIDS) #merge the datasets on energy, ID, zenith and azimuth

    maxlen = max(len(df_1), len(df_2))
    if easy_merge is False:
        assert df_1.duplicated().sum() == 0
        assert df_2.duplicated().sum() == 0
        merged = pd.merge(df_1, df_2, on=MERGEIDS) #merge the datasets on energy, ID, zenith and azimuth
        assert len(merged) <= maxlen, \
            ("Merged df is shorter than the maximum length of both telescope files, this should not be possible."
             f"len df1: {len(df_1)}, df2: {len(df_2)}, merged: {len(merged)}")
    else:
        lastequal1 = np.all(df_1.iloc[-1] == df_1.iloc[-2])
        lastequal2 = np.all(df_2.iloc[-1] == df_2.iloc[-2])
        # leads to 2x2 for the duplicated values -> merged is 2 rows too long
        assert lastequal1 == lastequal2, \
            "The same rows are not both duplicates/ non duplicate, something is wrong idk what though"
        if not lastequal1:
            # assert np.array_equal(df_1[EMERGEIDS].values, df_2[EMERGEIDS].values)
            # merged = pd.merge(df_1, df_2, on=EMERGEIDS)
            df_1.drop(columns=["0", "1", "6404", "6406"], inplace=True)
            merged = pd.merge(df_1, df_2, left_index=True, right_index=True)
            # merged = pd.concat((df_1, df_2), axis=1)
        else:
            df_1.drop(columns=["0", "1", "6404", "6406"], inplace=True)
            merged = pd.merge(df_1, df_2, left_index=True, right_index=True)
            # merged = pd.concat((df_1.iloc[:-1], df_2.iloc[:-1]), axis=1)
            # merged = pd.merge(df_1.iloc[:-1], df_2.iloc[:-1], on=EMERGEIDS)
            maxlen -= 1
        # merged = merged.drop(columns=["0_x", "6404_x", "6406_x"])
        # merged = merged.rename(columns={'0_y': '0', '6404_y': '6404', "6406_y": "6406"})
        assert len(merged) == maxlen, \
            ("Merged df is not equal to the maximum length of both telescope files, this should not be possible."
             f"len df1: {len(df_1)}, df2: {len(df_2)}, merged: {len(merged)}")

    merged = merged.reindex_axis(order, axis=1) #reindex the axis

    # for col in ["1", "2", "3", "6408", "6409", "6405", "6407"]:
    delcols = ["1", "2", "3", "6405", "6407"]
    for col in delcols:
        del merged[col] #delete duplicated columns

    # assert len(df_1.columns)+len(df_2.columns)-len(delcols) == len(merged.columns), f"{len(df_1.columns)} {len(df_2.columns)} {len(delcols)}"
    assert len(df_1.columns)+len(df_2.columns)-1 == len(merged.columns), f"{len(df_1.columns)} {len(df_2.columns)} {len(delcols)}"

    array = np.array(merged) #convert back to np.array
    cth.create_hdf5_from_dataset(array, outfilename)
    assert path.isfile(outfilename), f"outfile {outfilename} doesnt exist"
    return outfilename
