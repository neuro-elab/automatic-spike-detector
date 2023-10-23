from typing import Any

import numpy as np


def compute_rescaled_timeline(
    start_timestamp: float, length: int, sfreq: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    times = np.linspace(start=0, stop=length, num=length) / sfreq
    return times + start_timestamp
