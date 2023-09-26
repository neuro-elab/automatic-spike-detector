from typing import List, Tuple

import h5py
import mne.io
import numpy as np
from h5py import Group, Dataset
from loguru import logger
from mne.io import RawArray

from spidet.domain.Trace import Trace

# TODO: remove patient labels
LABELS_EL010 = [
    "Hip21",
    "Hip22",
    "Hip23",
    "Hip24",
    "Hip25",
    "Hip26",
    "Hip27",
    "Hip28",
    "Hip29",
    "Hip210",
    "Temp1",
    "Temp2",
    "Temp3",
    "Temp4",
    "Temp5",
    "Temp6",
    "Temp7",
    "Temp8",
    "Temp9",
    "Temp10",
    "FrOr1",
    "FrOr2",
    "FrOr3",
    "FrOr4",
    "FrOr5",
    "FrOr6",
    "FrOr7",
    "FrOr8",
    "FrOr9",
    "FrOr10",
    "FrOr11",
    "FrOr12",
    "In An1",
    "In An2",
    "In An3",
    "In An4",
    "In An5",
    "In An6",
    "In An7",
    "In An8",
    "In An9",
    "In An10",
    "In An11",
    "In An12",
    "InPo1",
    "InPo2",
    "InPo3",
    "InPo4",
    "InPo5",
    "InPo6",
    "InPo7",
    "InPo8",
    "InPo9",
    "InPo10",
    "InPo11",
    "InPo12",
    "Hip11",
    "Hip12",
    "Hip13",
    "Hip14",
    "Hip15",
    "Hip16",
    "Hip17",
    "Hip18",
    "Hip19",
    "Hip110",
    "Hip111",
    "Hip112",
]

LEAD_PREFIXES = ["Hip2", "Temp", "FrOr", "In An", "InPo", "Hip1"]

HDF5 = "h5"
EDF = "edf"
FIF = "fif"


def read_file(path: str, dataset_paths: List[str] = None) -> List[Trace]:
    """
    Read EEG data from a file and return a list of Trace objects, containing the EEG data of each channel.

    Reads EEG data from a file specified by 'path'. The supported file formats
    include '.h5', '.fif', and '.edf'.

    Parameters
    ----------
    path : str
        The file path of the EEG data file.
    dataset_paths : List[str]
        The absolute paths to the datasets within an HDF5 file.

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
        return read_h5_file(path, dataset_paths)
    elif file_format in [EDF, FIF]:
        return read_edf_or_fif_file(path, file_format)
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
    h5_file = h5py.File(file_path, "r")

    # Extract the raw datasets from the hdf5 file
    raw_traces: List[Dataset] = [h5_file.get(path) for path in dataset_paths]

    # Retrieve the channel names from the parent group of the traces
    traces_parent_group: Group = raw_traces[0].parent
    channel_names = traces_parent_group.keys()

    # Generate an instance of mne.io.RawArray from the h5 Datasets
    # in order to generate possible bipolar references
    raw: RawArray = RawArray(
        np.array(raw_traces), info=mne.create_info(ch_names=channel_names)
    )
    if bipolar_reference:
        raw = generate_bipolar_references(raw, leads)

    # Currently attributes are on the level of the set of traces,
    # will later on be on the level of individual traces (recordings)
    attributes = traces_parent_group.attrs

    return [
        create_trace(label, times, attributes)
        for label, times in zip(raw.ch_names, raw.get_data())
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
        mne.io.read_raw_fif(file_path)
        if file_format == FIF
        else mne.io.read_raw_edf(file_path, preload=True)
    )
    if bipolar_reference:
        raw = generate_bipolar_references(raw, leads)
    attributes = dict({"sfreq": raw.info["sfreq"], "unit": None})

    return [
        create_trace(label, times, attributes)
        for label, times in zip(raw.ch_names, raw.get_data())
    ]


def create_trace(label: str, dataset: np.array, attributes: dict) -> Trace:
    """
    Create a Trace object from a recording of a particular electrode with a corresponding label.

    Parameters
    ----------
    label : str
        The label of the trace.

    dataset : array_like
        Numerical representation of the recording.

    attributes : dict
        A set of attributes of the recording.

    Returns
    -------
    Trace
        A Trace object representing a recording from an electrode with the corresponding label.
    """
    return Trace(
        label,
        attributes.get("sfreq"),
        attributes.get("unit"),
        dataset.astype(np.float64),
    )


def generate_bipolar_references(raw: RawArray, leads: List[str]) -> RawArray:
    if leads is None:
        raise Exception(
            "bipolar_reference is true but no leads were passed for whose channels to perform the referencing"
        )
    anodes, cathodes = get_anodes_and_cathodes(leads)
    raw = mne.set_bipolar_reference(
        raw, anode=anodes, cathode=cathodes, drop_refs=True, copy=False
    )
    return raw


def get_anodes_and_cathodes(leads: List[str]) -> Tuple[List[str], List[str]]:
    anodes, cathodes = [], []
    for prefix in LEAD_PREFIXES:
        channels = list(
            filter(lambda channel_name: channel_name.startswith(prefix), leads)
        )
        for idx in range(len(channels) - 1):
            anodes.append(channels[idx])
            cathodes.append(channels[idx + 1])

    return anodes, cathodes
