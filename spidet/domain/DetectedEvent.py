from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectedEvent:
    """
    This class represents a detected period of abnormal activity in a given
    :py:class:`~spidet.domain.ActivationFunction`.

    Attributes
    ----------

    times: numpy.ndarray[Any, numpy.dtype[float]]
        An array of UNIX timestamps representing the points in time for each data point
        within the detected event period.

    values: numpy.ndarray[Any, numpy.dtype[float]]
        The activation levels at each point in time within the detected event period.
    """

    times: np.ndarray[Any, np.dtype[float]]
    values: np.ndarray[Any, np.dtype[float]]
