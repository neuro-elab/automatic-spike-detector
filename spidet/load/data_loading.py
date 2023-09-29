from typing import List, Tuple

import h5py
import mne.io
import numpy as np
from h5py import Dataset, File
from loguru import logger
from mne.io import RawArray

from spidet.domain.Trace import Trace

# Supported file formats
HDF5 = "h5"
EDF = "edf"
FIF = "fif"


def read_file(
    path: str,
    dataset_paths: List[str] = None,
    bipolar_reference: bool = False,
    leads: List[str] = None,
) -> List[Trace]:
    """
    Read EEG data from a file and return a list of Trace objects,
    containing the EEG data of each channel.

    Reads EEG data from a file specified by 'path'.
    The supported file formats include '.h5', '.fif', and '.edf'.

    Parameters
    ----------
    path : str
        The file path of the EEG data file.
    dataset_paths : List[str] (default None)
        If the file is an '.h5' file, these are the absolute paths
        to the datasets within the file.
    bipolar_reference: bool (default False)
        A boolean indicating whether bipolar references between respective
        channels should be calculated and subsequently considered as traces
    leads: List[str] (default None)
        The leads for whose channels to perform bipolar referencing.
        NOTE: 'leads' cannot be None if 'bipolar_reference' is True

    Returns
    -------
    List[Trace]
        A list of Trace objects containing EEG data.

    Raises
    ------
    Exception
        If the file format is not supported.
    """
    filename = path[path.rfind("/") + 1 :]
    logger.debug(f"Loading file {filename}")
    file_format = path[path.rfind(".") + 1 :].lower()

    if file_format == HDF5:
        return read_h5_file(path, dataset_paths, bipolar_reference, leads)
    elif file_format in [EDF, FIF]:
        return read_edf_or_fif_file(path, file_format, bipolar_reference, leads)
    else:
        raise Exception(
            f"The file format {file_format} ist not supported by this application"
        )


def read_h5_file(
    file_path: str,
    dataset_paths: List[str],
    bipolar_reference: bool = False,
    leads: List[str] = None,
) -> List[Trace]:
    """
    Loads a file in HDF5 format and transforms its content to a list of Trace objects.
    Provides the option to perform bipolar referencing for channels within a lead,
    if the leads are provided as argument.

    Parameters
    ----------
    file_path : str
        The path to the HDF5 file.
    dataset_paths : List[str]
        The absolute paths to the datasets within an HDF5 file.
    bipolar_reference: bool (default False)
        A boolean indicating whether bipolar references between respective channels
        should be calculated and subsequently considered as traces
    leads: List[str] (default None)
        The leads for whose channels to perform bipolar referencing.
        NOTE: 'leads' cannot be None if 'bipolar_reference' is True

    Returns
    -------
    List[Trace]
        A list of Trace objects representing the content of the HDF5 file.
    """
    if dataset_paths is None:
        raise Exception("Paths to the dataset within the HDF5 file can not be None")

    h5_file = h5py.File(file_path, "r")

    # Extract the raw datasets from the hdf5 file
    raw_traces: List[Dataset] = [h5_file.get(path) for path in dataset_paths]

    # Extract start timestamps for datasets
    start_timestamps: List[float] = [
        extract_start_timestamp(path, h5_file) for path in dataset_paths
    ]

    # Extract channel names from the dataset paths
    channel_names = [
        channel_path[channel_path.rfind("/") + 1 :] for channel_path in dataset_paths
    ]

    # Extract frequencies from datasets
    frequencies: List[float] = [
        raw_trace.attrs.get("sfreq") for raw_trace in raw_traces
    ]

    if bipolar_reference:
        # Generate an instance of mne.io.RawArray from the h5 Datasets
        # in order to generate bipolar references
        raw: RawArray = RawArray(
            np.array(raw_traces),
            info=mne.create_info(
                ch_names=channel_names,
                ch_types="eeg",
                sfreq=frequencies[0],
                verbose=False,
            ),
            verbose=False,
        )
        raw = generate_bipolar_references(raw, leads)
        raw_traces = raw.get_data()

    return [
        create_trace(label, data, freq, ts)
        for label, data, freq, ts in zip(
            channel_names, raw_traces, frequencies, start_timestamps
        )
    ]


def read_edf_or_fif_file(
    file_path: str,
    file_format: str,
    bipolar_reference: bool = False,
    leads: List[str] = None,
):
    """
    Loads a file in either FIF or EDF format and transforms its content to a list of Trace objects.
    Provides the option to perform bipolar referencing for channels within a lead,
    if the leads are provided as argument.

    Parameters
    ----------
    file_path : str
        The path to the file.
    file_format : str
        format indicating whether the file is of type FIF or EDF
    bipolar_reference: bool (default False)
        A boolean indicating whether bipolar references between respective channels
        should be calculated and subsequently considered as traces
    leads: List[str] (default None)
        The leads for whose channels to perform bipolar referencing.
        NOTE: 'leads' cannot be None if 'bipolar_reference' is True

    Returns
    -------
    List[Trace]
        A list of Trace objects representing the content of the file.
    """
    raw: RawArray = (
        mne.io.read_raw_fif(file_path, verbose=False)
        if file_format == FIF
        else mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    )
    if bipolar_reference:
        raw = generate_bipolar_references(raw, leads)

    return [
        create_trace(label, times, raw.info["sfreq"], raw.info["meas_date"].timestamp())
        for label, times in zip(raw.ch_names, raw.get_data())
    ]


def create_trace(
    label: str, dataset: np.array, sfreq: int, start_timestamp: float
) -> Trace:
    """
    Create a Trace object from a recording of a particular electrode with a corresponding label.

    Parameters
    ----------
    label : str
        The label of the trace.

    dataset : array_like
        Numerical representation of the recording.

    sfreq : int
        Sampling frequency of the recording.

    start_timestamp: float
        Start timestamp of the recording (UNIX timestamp).

    Returns
    -------
    Trace
        A Trace object representing a recording from an electrode with the corresponding label.
    """
    return Trace(
        label,
        sfreq,
        start_timestamp,
        dataset[:].astype(np.float64),
    )


def extract_start_timestamp(dataset_path: str, file: File) -> float:
    sub_path = dataset_path[dataset_path.find("traces/") + 7 :]
    subgroup = sub_path[: sub_path.find("/")]
    subgroup_attributes = file.get(f"traces/{subgroup}").attrs
    return subgroup_attributes["start_timestamp"]


def generate_bipolar_references(raw: RawArray, leads: List[str]) -> RawArray:
    if leads is None:
        raise Exception(
            "bipolar_reference is true but no leads were passed for whose channels to perform the referencing"
        )
    anodes, cathodes = get_anodes_and_cathodes(leads, raw.ch_names)
    raw = mne.set_bipolar_reference(
        raw, anode=anodes, cathode=cathodes, drop_refs=True, copy=False, verbose=False
    )
    return raw


def get_anodes_and_cathodes(
    leads: List[str], channel_names: List[str]
) -> Tuple[List[str], List[str]]:
    anodes, cathodes = [], []
    for prefix in leads:
        channels = list(
            filter(lambda channel_name: channel_name.startswith(prefix), channel_names)
        )
        for idx in range(len(channels) - 1):
            anodes.append(channels[idx])
            cathodes.append(channels[idx + 1])

    return anodes, cathodes
