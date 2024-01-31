from typing import List

import mne
import numpy as np

from mne.io import RawArray


def resample_data(
    data: np.array, channel_names: List[str], sfreq: int, resampling_freq: int
) -> np.array:
    """
    Resamples the data with the desired frequency.

    Parameters
    ----------
    sfreq : float
        Original frequency of the data.

    channel_names : list of str
        Labels of the channels.

    data : array-like
        Data to be resampled.

    resampling_freq : float
        Target resampling frequency.

    Returns
    -------
    array-like
        Resampled data.
    """
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq)
    resampled_data = RawArray(data, info=info, verbose=False).resample(
        sfreq=resampling_freq, verbose=False
    )
    return resampled_data.get_data()
