from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectedPeriod:
    period_times: np.ndarray[Any, np.dtype[float]]
    period_values: np.ndarray[Any, np.dtype[float]]
