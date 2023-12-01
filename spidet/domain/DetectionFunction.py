from dataclasses import dataclass
from typing import Any, List

import numpy as np

from spidet.domain.DetectedEvent import DetectedEvent


@dataclass
class DetectionFunction:
    label: str
    unique_id: str
    times: np.ndarray[Any, np.dtype[float]]
    data_array: np.ndarray[Any, np.dtype[float]]
    detected_events_on: np.ndarray[Any, np.dtype[int]]
    detected_events_off: np.ndarray[Any, np.dtype[int]]
    event_threshold: float

    def get_sub_period(self, offset: float, duration: float):
        # Find indices corresponding to offset and end of duration
        start_idx = (np.abs(self.times - offset)).argmin()
        end_index = (np.abs(self.times - (offset + duration))).argmin()
        return self.data_array[start_idx:end_index]

    def get_detected_events(
        self,
    ) -> List[DetectedEvent]:
        detected_events = []

        for idx, (on, off) in enumerate(
            zip(self.detected_events_on, self.detected_events_off)
        ):
            detected_period = DetectedEvent(
                self.times[on : off + 1], self.data_array[on : off + 1]
            )
            detected_events.append(detected_period)

        return detected_events
