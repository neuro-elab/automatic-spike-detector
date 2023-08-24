import h5py
import numpy as np

FILEPATH = '../../../../Data/'
FILENAME = "Sz2.h5"


def read_h5_file(path):
    return h5py.File(path, "r")


h5_file = read_h5_file(FILEPATH + FILENAME)
