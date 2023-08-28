import numpy as np
from mne.io import RawArray


def resample_data(data: np.array, resampling_freq: int) -> np.array:
    """
    Resamples the data with the desired frequency

    :param data: data to be resampled
    :param resampling_freq:
    :return: resampled data
    """
    return RawArray(data).resample(sfreq=resampling_freq)
