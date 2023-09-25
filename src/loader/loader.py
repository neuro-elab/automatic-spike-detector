from typing import List, Tuple

import h5py
import mne.io
import numpy as np
from loguru import logger
from mne.io.edf.edf import RawEDF

from src.domain.Trace import Trace

TRACES = "traces"
RAW_DATA = "raw"
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

PREFIX_LEAD_GROUPS = ["Hip2", "Temp", "FrOr", "In An", "InPo", "Hip1"]


def create_trace(label: str, dataset: np.array, attributes: dict) -> Trace:
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
        dataset.astype(np.float64),
    )


def read_h5_file(path: str) -> List[Trace]:
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


def read_file(path: str) -> List[Trace]:
    # TODO: add doc
    filename = path[path.rfind("/") + 1 :]
    logger.debug(f"Loading file {filename}")
    ext = path[path.rfind(".") + 1 :].lower()

    if ext == "h5":
        return read_h5_file(path)
    elif ext == "fif" or ext == "edf":
        raw: RawEDF = (
            mne.io.read_raw_fif(path)
            if ext == "fif"
            else mne.io.read_raw_edf(path, preload=True)
        )
        raw_re_referenced = re_reference(raw)
        raw_traces = raw_re_referenced.get_data()
        attributes = dict({"sfreq": raw.info["sfreq"], "unit": None})

        return [
            create_trace(label, times, attributes)
            for label, times in zip(raw_re_referenced.ch_names, raw_traces)
        ]
    else:
        raise Exception(f"Data format {ext} ist not supported by this application")


def re_reference(raw: RawEDF) -> RawEDF:
    anodes, cathodes = get_anodes_and_cathodes()
    raw = mne.set_bipolar_reference(
        raw, anode=anodes, cathode=cathodes, drop_refs=True, copy=False
    )
    return raw


def get_anodes_and_cathodes() -> Tuple[List[str], List[str]]:
    anodes, cathodes = [], []
    for prefix in PREFIX_LEAD_GROUPS:
        channels = list(
            filter(lambda channel_name: channel_name.startswith(prefix), LABELS_EL010)
        )
        for idx in range(len(channels) - 1):
            anodes.append(channels[idx])
            cathodes.append(channels[idx + 1])

    return anodes, cathodes
