from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np


@dataclass
class PreprocessedData:
    channel_names: List[str]
    times: np.ndarray
    line_length: np.ndarray
    std_line_length: np.ndarray
