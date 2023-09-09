import h5py
import mne.io
import numpy as np
from loguru import logger

from src.domain.Trace import Trace

TRACES = "traces"
RAW_DATA = "raw"


def create_trace(label: str, dataset: np.array, attributes: dict):
    """
    Create a Trace object from a recording of a particular electrode with corresponding label

    :param label: label of the trace
    :param dataset: numerical representation of the recording
    :param attributes: set of attributes of a recording
    :return: Trace object representing a recording from an electrode with the corresponding label
    """
    return Trace(
        label,
        attributes.get("sfreq"),
        attributes.get("unit"),
        dataset[:10000].astype(np.float64),
    )


def read_h5_file(path: str):
    """
    Loads a file in hdf5 format and transform its content to a list of Traces.

    :param path: path to file
    :return: List of Trace objects
    """
    h5_file = h5py.File(path, "r")

    # Extract the raw datasets from the hdf5 file
    raw_traces = h5_file.get(TRACES).get(RAW_DATA)

    # Currently attributes are on the level of the set of traces,
    # will later on be on the level of individual traces (recordings)
    attributes = raw_traces.attrs

    return [
        create_trace(label, raw_traces.get(label), attributes)
        for label in raw_traces.keys()
    ]


def read_file(path: str):
    # TODO: add doc
    filename = path[path.rfind("/") + 1 :]
    logger.debug(f"Loading file {filename}")
    ext = path[path.rfind(".") + 1 :].lower()

    if ext == "h5":
        return read_h5_file(path)
    elif ext == "fif" or ext == "edf":
        file = mne.io.read_raw_fif(path) if ext == "fif" else mne.io.read_raw_edf(path)
        raw_traces = file.get_data()
        attributes = dict({"sfreq": file.info["sfreq"], "unit": None})

        return [
            create_trace(label, times, attributes)
            for label, times in zip(file.ch_names, raw_traces)
        ]
    else:
        raise Exception(f"Data format {ext} ist not supported by this application")
