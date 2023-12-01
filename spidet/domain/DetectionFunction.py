from dataclasses import dataclass
from typing import Any, List

import numpy as np

from spidet.domain.DetectedPeriod import DetectedPeriod


@dataclass
class DetectionFunction:
    label: str
    unique_id: str
    times: np.ndarray[Any, np.dtype[float]]
    data_array: np.ndarray[Any, np.dtype[float]]
    detected_periods_on: np.ndarray[Any, np.dtype[int]]
    detected_periods_off: np.ndarray[Any, np.dtype[int]]
    threshold: float

    def get_sub_period(self, offset: float, duration: float):
        # Find indices corresponding to offset and end of duration
        start_idx = (np.abs(self.times - offset)).argmin()
        end_index = (np.abs(self.times - (offset + duration))).argmin()
        return self.data_array[start_idx:end_index]

    def get_detected_periods(
        self,
    ) -> List[DetectedPeriod]:
        detected_periods = []

        for idx, (on, off) in enumerate(
            zip(self.detected_periods_on, self.detected_periods_off)
        ):
            detected_period = DetectedPeriod(
                self.times[on : off + 1], self.data_array[on : off + 1]
            )
            detected_periods.append(detected_period)

        return detected_periods
