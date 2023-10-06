from dataclasses import dataclass

import numpy as np


@dataclass
class Artifacts:
    bad_times: np.ndarray
    bad_channels: np.ndarray
