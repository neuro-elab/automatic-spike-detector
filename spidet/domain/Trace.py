from dataclasses import dataclass

import numpy as np


@dataclass
class Trace:
    label: str
    # duration: float
    # n_samples: int
    # processing: str
    sfreq: int
    # unit: str
    # start_date: date
    # start_time: time
    start_timestamp: float
    data: np.array
