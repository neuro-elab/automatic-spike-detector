from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from spidet.domain.SpikeDetectionFunction import SpikeDetectionFunction


@dataclass
class CoefficientsFunction(SpikeDetectionFunction):
    spikes_on_indices: np.ndarray[Any, np.dtype[int]]
    spikes_off_indices: np.ndarray[Any, np.dtype[int]]
    spike_threshold: float
    codes_for_spikes: bool
