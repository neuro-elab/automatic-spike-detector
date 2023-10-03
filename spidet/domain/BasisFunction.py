from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BasisFunction:
    label: str
    unique_id: str
    channel_names: List[str]
    data_array: np.ndarray
