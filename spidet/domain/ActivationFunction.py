from dataclasses import dataclass
from typing import List

import numpy as np

from spidet.domain.DetectedEvent import DetectedEvent


@dataclass
class ActivationFunction:
    """
    This class represents the activation levels of a given EEG metapattern (i.e., a
    :py:class:`~spidet.domain.BasisFunction`) in the time domain and contains periods of abnormal EEG activity,
    represented as detected events (see :py:class:`~spidet.domain.DetectedEvent`).

    Attributes
    ----------

    label: str
        The label of the ActivationFunction; the label of a given ActivationFunction contains the row index
        in the :math:`H` matrix prefixed by a capital H.

    unique_id: str

    times: numpy.ndarray[numpy.dtype[float]]
        An array containing the timestamps of each data point

    data_array: numpy.ndarray[numpy.dtype[float]]
        An array containing the activation level at each point in time

    detected_events_on: numpy.ndarray[numpy.dtype[int]]
        An array with the indices in the data array corresponding to the onsets of the detected events.

    detected_events_off: numpy.ndarray[numpy.dtype[int]]
        An array with the indices in the data array corresponding to the offsets of the detected events.

    event_threshold: float
        the threshold used for the computation of the detected events.
    """

    label: str
    unique_id: str
    times: np.ndarray[np.dtype[float]]
    data_array: np.ndarray[np.dtype[float]]
    detected_events_on: np.ndarray[np.dtype[int]]
    detected_events_off: np.ndarray[np.dtype[int]]
    event_threshold: float

    def get_sub_period(
        self, offset: float, duration: float
    ) -> np.ndarray[np.dtype[float]]:
        """
        Computes a sub period of the :py:class:`ActivationFunction`.

        Parameters
        ----------
        offset: float
            Offset from the start of the recording in seconds.

        duration: float
            Duration of the sub period in seconds.

        Returns
        -------
        numpy.ndarray[numpy.dtype[float]]
            Array containing the data points within the defined sub period.
        """
        # Find indices corresponding to offset and end of duration
        start_idx = (np.abs(self.times - offset)).argmin()
        end_index = (np.abs(self.times - (offset + duration))).argmin()
        return self.data_array[start_idx:end_index]

    def get_detected_events(
        self,
    ) -> List[DetectedEvent]:
        """
        Returns a list of :py:class:`~spidet.domain.DetectedEvent` objects representing the
        computed events detected on the given :py:class:`ActivationFunction`.
        """
        detected_events = []

        for idx, (on, off) in enumerate(
            zip(self.detected_events_on, self.detected_events_off)
        ):
            detected_period = DetectedEvent(
                self.times[on : off + 1], self.data_array[on : off + 1]
            )
            detected_events.append(detected_period)

        return detected_events

    def get_event_mask(self):
        """
        Returns a binary numpy array indicating the indices of all detected events of the given
        :py:class:`ActivationFunction`.
        """
        event_mask = np.zeros(len(self.data_array))
        for on, off in zip(self.detected_events_on, self.detected_events_off):
            event_mask[on : off + 1] = 1

        return event_mask.astype(int)
