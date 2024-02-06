from dataclasses import dataclass

import numpy as np


@dataclass
class Trace:
    """
    This class represents the recording from a given channel.

    Attributes
    ----------

    label: str
        The name of the channel.

    sfreq: int
        The sampling frequency of the data.

    start_timestamp: float
        The start timestamp of the recording as a UNIX timestamp.

    data: numpy.ndarray[numpy.dtype[float]]
        An array containing the EEG data of the given channel.
    """

    label: str
    sfreq: int
    start_timestamp: float
    data: np.ndarray[np.dtype[float]]
