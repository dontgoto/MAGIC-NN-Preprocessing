### import packages

import numpy as np
from tqdm import tqdm
from os import path
from dataprepro import normalisation as norm
from dataprepro import csv2hdf5 as cth

PIXELFNAME = "pixelwise_norm.txt"

def camerawise_normalisation(infilename, truncate=False):
    """
    truncate : int or False
        number of columns to truncate the data to, defaults to False for no truncation
    """
    data = np.array(norm.read_hdf5_dataset(infilename)) #read the dataset

    for i in range(0, data.shape[0]):
        data[i][0] = np.log(data[i][0]) #logarithmize the energy

    mean1 = np.mean(data, axis=0, dtype='float64') #calculate the mean for all features
    std_dev1 = np.std(data, axis=0, dtype='float64') #calculate the standard deviation for all features

    mean_charge1 = 0 #initialise variables for the camerawise normalisation
    mean_charge2 = 0
    mean_time1 = 0
    mean_time2 = 0
    stddev_charge1 = 0
    stddev_charge2 = 0
    stddev_time1 = 0
    stddev_time2 = 0

    for i in range(0, 1600):
        mean_charge1 += mean1[4*i + 1] #calculate the sum for all cameras and variables
        mean_charge2 += mean1[4*i + 2]
        mean_time1 += mean1[4*i + 3]
        mean_time2 += mean1[4*i + 4]
        stddev_charge1 += std_dev1[4*i + 1]
        stddev_charge2 += std_dev1[4*i + 2]
        stddev_time1 += std_dev1[4*i + 3]
        stddev_time2 += std_dev1[4*i + 4]

    mean_charge1 /= 1039 #calculate the mean and standard deviation for all cameras/features
    mean_charge2 /= 1039
    mean_time1 /= 1039
    mean_time2 /= 1039
    stddev_charge1 /= 1039
    stddev_charge2 /= 1039
    stddev_time1 /= 1039
    stddev_time2 /= 1039

    for i in tqdm(range(0, 1600)): #normalise the pixels according to camerawise normalisation
        if (std_dev1[4*i + 1] != 0):
            data[:,4*i + 1] -= mean_charge1
            data[:,4*i + 1] /= stddev_charge1
        if (std_dev1[4*i + 2] != 0):
            data[:,4*i + 2] -= mean_charge1
            data[:,4*i + 2] /= stddev_charge1
        if (std_dev1[4*i + 3] != 0):
            data[:,4*i + 3] -= mean_charge1
            data[:,4*i + 3] /= stddev_charge1
        if (std_dev1[4*i + 4] != 0):
            data[:,4*i + 4] -= mean_charge1
            data[:,4*i + 4] /= stddev_charge1

    data[:,0] -= mean1[0] #normalise the other features
    data[:,0] /= std_dev1[0]
    data[:,6401] -= mean1[6401]
    data[:,6401] /= std_dev1[6401]
    data[:,6402] -= mean1[6402]
    data[:,6402] /= std_dev1[6402]

    np.nan_to_num(data, copy=False)
    if truncate:
        data = data[:truncate]
        print("Data truncated.")

    basename = path.basename(infilename)
    pathname = path.dirname(path.dirname(infilename))
    print(data.shape)

    pathname = path.abspath(pathname)
    opath = path.join(pathname, '/normalized/camerawise_normalized_'+basename)
    cth.create_hdf5_from_dataset(data, opath) #save the file


def write_pixelnormfile(mean, std_dev, pathname):
    outfname = path.join(pathname, PIXELFNAME)
    with open(outfname, "a") as f:
        #create a file for the normalisation (for the opposite direction after the estimation and the analysis of the inputs)
        for i in range(0, mean.shape[0]): #go over all features
            f.write(str(mean[i]) + " " + str(std_dev[i]) + "\n") #extract mean and standard deviation


def pixelwise_normalisation(infilename, opath, truncate=False):
    """
    truncate : int or False
        number of columns to truncate the data to, defaults to False for no truncation
    """
    data = np.array(cth.read_hdf5_dataset(infilename)) #read the dataset
    basename = path.basename(infilename)
    pathname = path.dirname(path.dirname(infilename))

    print("number of nans before norm: ", np.sum(np.isnan(data.flatten())))
    for i in tqdm(range(0, data.shape[0])):
        data[i][0] = np.log(data[i][0]) #logarithmize the energy
    test = np.log(data[:,0])
    print( np.array_equal(test, data[:,0]))
    print("number of nans before norm: ", np.sum(np.isnan(data.flatten())))

    mean = np.mean(data, axis=0, dtype='float64') #calculate the mean for all features
    std_dev = np.std(data, axis=0, dtype='float64') #calculate the standard deviation for all features
    write_pixelnormfile(mean, std_dev, pathname)

    for i in range(0,data.shape[1]-3): #go over all features
        if (std_dev[i] != 0):
            data[:,i] -= mean[i] #apply the normalisation
            data[:,i] /= std_dev[i]

    print("number of nans after norm: ", np.sum(np.isnan(data.flatten())))
    print("number of data points: ", data.shape[0]*data.shape[1])

    np.nan_to_num(data, copy=False)
    if truncate:
        data = data[:truncate]
        print("Data truncated.")
    print(data.shape)
    opath = path.abspath(opath)
    opath = path.join(opath, '/pixelwise_normalised_'+basename)
    cth.create_hdf5_from_dataset(data, opath) #save the file
