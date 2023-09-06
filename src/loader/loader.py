
import h5py
import numpy as np
from h5py import AttributeManager

from src.domain.Trace import Trace

FILEPATH = '../../../../Data/'
FILENAME = "Sz2.h5"
TRACES = 'traces'
RAW_DATA = 'raw'


def create_trace(label: str, dataset: h5py.Dataset, attributes: AttributeManager):
    """
    Create a Trace object from a recording of a particular electrode with corresponding label

    :param label: label of the trace
    :param dataset: numerical representation of the recording
    :param attributes: set of attributes of a recording
    :return: Trace object representing a recording from an electrode with the corresponding label
    """
    return Trace(label,
                 attributes.get('duration'),
                 attributes.get('n_samples'),
                 attributes.get('processing'),
                 dataset.attrs.get('sfreq'),
                 dataset.attrs.get('unit'),
                 attributes.get('start_date'),
                 attributes.get('start_time'),
                 attributes.get('start_timestamp'),
                 dataset[:].astype(np.float64))


def read_h5_file(path):
    """
    Load a file in hdf5 format and transform its content to a list of Traces

    :param path: path to file
    :return: List of Trace objects
    """
    h5_file = h5py.File(path, "r")

    # Extract the raw datasets from the hdf5 file
    raw_traces = h5_file.get(TRACES).get(RAW_DATA)

    # Currently attributes are on the level of the set of traces,
    # will later on be on the level of individual traces (recordings)
    attributes = raw_traces.attrs

    return [create_trace(label, raw_traces.get(label), attributes) for label in raw_traces.keys()]


#start = time.time()
#datasets = read_h5_file(FILEPATH + FILENAME)
#size = len(datasets)
#print(f'elapsed time: {(time.time() - start)/1000} seconds')
