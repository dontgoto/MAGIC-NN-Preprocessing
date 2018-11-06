### file to read .csv, group them and save them as .hdf5
from os import path
from os import remove
from dataprepro import csv2hdf5 as cth

logger = logging.getLogger(__name__)

def create_hdf5(infileName):
    outfileName = infileName.replace(".csv", ".hdf5")
    newOutfileName = outfileName.replace('csv/', 'hdf5/')
    cth.create_hdf5_from_csv(infileName, newOutfileName, dtype="float16") #load .csv, save as .hdf5
    logger.info(path.basename(newOutfileName), "done")
    return newOutfileName
