from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectedEvent:
    event_times: np.ndarray[Any, np.dtype[float]]
    event_values: np.ndarray[Any, np.dtype[float]]
