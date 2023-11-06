from dataclasses import dataclass
from typing import Any, Tuple, List
import numpy as np

from spidet.domain.SpikeDetectionFunction import SpikeDetectionFunction


@dataclass
class CoefficientsFunction(SpikeDetectionFunction):
    spikes_on_indices: np.ndarray[Any, np.dtype[int]]
    spikes_off_indices: np.ndarray[Any, np.dtype[int]]
    spike_threshold: float
    codes_for_spikes: bool

    def get_spike_points(
        self,
    ) -> np.ndarray[Any, np.dtype[float]]:
        spike_points = []

        for idx in range(len(self.spikes_on_indices)):
            spike_time_on, spike_time_off = (
                self.spikes_on_indices[idx],
                self.spikes_off_indices[idx],
            )

            spike_times = self.times[spike_time_on : spike_time_off + 1]
            spike_values = self.data_array[spike_time_on : spike_time_off + 1]

            if idx == 0:
                spike_points = np.column_stack((spike_times, spike_values))
            else:
                spike_points = np.vstack(
                    (spike_points, np.column_stack((spike_times, spike_values)))
                )

        return spike_points
