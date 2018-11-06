import h5py as h5
import numpy as np
from os import path
from csv2hdf5 import glob_and_check
import click


def print_linenumber(fname):
    h5file = h5.File(fname)
    print(f"{path.basename(fname)} has shape: {h5file['data'].shape}")
    h5file.close()


@click.command()
@click.option('--glob', default="./*.hdf5", type=click.Path(),
              help='glob that gives all outputfiles containing the energy from the nn')
def main(glob):
    filenames = glob_and_check(glob)
    ish5 = np.all(["h5" in fname for fname in filenames])
    ishdf5 = np.all(["hdf5" in fname for fname in filenames])
    ishdf = np.all(["hdf" in fname for fname in filenames])
    if not (ish5 or ishdf5 or ishdf):
        print("files are not ending with .hdf or something like that, this should fail soon.")
    for fname in filenames:
        print_linenumber(fname)


if __name__ == '__main__':
    main()
