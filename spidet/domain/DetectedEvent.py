from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedEvent:
    """
    This class represents a detected period of abnormal activity in a given
    :py:class:`~spidet.domain.ActivationFunction`.

    Attributes
    ----------

    times: numpy.ndarray[numpy.dtype[float]]
        An array of UNIX timestamps representing the points in time for each data point
        within the detected event period.

    values: numpy.ndarray[numpy.dtype[float]]
        The activation levels at each point in time within the detected event period.
    """

    times: np.ndarray[np.dtype[float]]
    values: np.ndarray[np.dtype[float]]
