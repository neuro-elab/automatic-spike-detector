from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeDetectionFunction:
    label: str
    unique_id: str
    times: np.ndarray
    data_array: np.ndarray

    def get_sub_period(self, offset: float, duration: float):
        # Find indices corresponding to offset and end of duration
        start_idx = (np.abs(self.times - offset)).argmin()
        end_index = (np.abs(self.times - (offset + duration))).argmin()
        return self.data_array[start_idx:end_index]
