import time

import h5py
import numpy as np
from h5py import AttributeManager

from src.domain.Trace import Trace

FILEPATH = '../../../../Data/'
FILENAME = "Sz2.h5"
TRACES = 'traces'
RAW_DATA = 'raw'


def create_trace(label: str, dataset: h5py.Dataset, attributes: AttributeManager):
    return Trace(label,
                 attributes.get('duration'),
                 attributes.get('n_samples'),
                 attributes.get('processing'),
                 dataset.attrs.get('sfreq'),
                 dataset.attrs.get('unit'),
                 attributes.get('start_date'),
                 attributes.get('start_time'),
                 attributes.get('start_timestamp'),
                 dataset[:])


def read_h5_file(path):
    h5_file = h5py.File(path, "r")
    raw_traces = h5_file.get(TRACES).get(RAW_DATA)
    attributes = raw_traces.attrs
    return [create_trace(label, raw_traces.get(label), attributes) for label in raw_traces.keys()]


start = time.time()
datasets = read_h5_file(FILEPATH + FILENAME)
size = len(datasets)
print(f'elapsed time: {(time.time() - start)/1000} seconds')
