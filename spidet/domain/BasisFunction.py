from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BasisFunction:
    """
    This class represents a single EEG metapattern extracted from the preprocessed EEG data and contains
    the expression levels of each channel in the given BasisFunction.

    Attributes
    ----------

    label: str
        The label of the BasisFunction; the label of a given BasisFunction contains the column index
        in the :math:`W` matrix prefixed by a capital W.

    unique_id: str

    data_array: numpy.ndarray[numpy.dtype[float]]
        An array containing the expression levels of each channel in the BasisFunction

    channel_names: List[str]
        An list with the names of the channels contained in the BasisFunction

    """

    label: str
    unique_id: str
    channel_names: List[str]
    data_array: np.ndarray[np.dtype[float]]
