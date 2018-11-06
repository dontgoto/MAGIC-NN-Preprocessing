# dataprepro/ #
Contains all the subroutines needed for preprocessing files for the NN and processing them after the NN has classified them.
All routines are called by process.py 
It is used for real data, but can now also process MC data, but the refactoring is still in progress.

1. globroot2csv.C
    - is used for converting the calibrated .root files to csv
    - is called by root2csv.py
2. remove_events.py
    - 
3. csv2hdf5.py
    - used for converting csv files to hdf5
    - can merge multiple hdf5 files of different (sub)runs
4. mergetelescopes.py
    - merges both telescope files to one
5. normalisation.py
    - applies to norm.txt file on the data and standardizes all events pixel by pixel
6. adjust_energy.py
    - is used for de-standerdizing the event energies once they have been estimated by the NN

# mc/ #
Contains some subroutines that are used in the preprocessing of MC data.