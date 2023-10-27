from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from spidet.domain.SpikeDetectionFunction import SpikeDetectionFunction


@dataclass
class CoefficientsFunction(SpikeDetectionFunction):
    spikes_on: np.ndarray[Any, np.dtype[float]]
    spikes_off: np.ndarray[Any, np.dtype[float]]
