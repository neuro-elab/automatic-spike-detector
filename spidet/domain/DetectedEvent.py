from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectedEvent:
    times: np.ndarray[Any, np.dtype[float]]
    values: np.ndarray[Any, np.dtype[float]]
