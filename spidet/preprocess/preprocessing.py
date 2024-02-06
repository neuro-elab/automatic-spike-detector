from typing import List

import numpy as np
from loguru import logger

from spidet.domain.Trace import Trace
from spidet.preprocess.filtering import filter_signal, notch_filter_signal
from spidet.preprocess.resampling import resample_data
from spidet.preprocess.rescaling import rescale_data


def apply_preprocessing_steps(
    traces: List[Trace],
    notch_freq: int,
    resampling_freq: int,
    bandpass_cutoff_low: int,
    bandpass_cutoff_high: int,
) -> np.ndarray[np.dtype[float]]:
    """
    Applies the necessary preprocessing steps to the original iEEG data. This involves:

        1.  Bandpass-filtering with a butterworth forward-backward filter of order 2
        2.  Notch-filtering
        3.  Rescaling
        4.  Resampling

    Parameters
    ----------
    traces : List[Trace]
        The original iEEG data as a list of Traces objects. Each trace corresponds to the recording of single channel.

    notch_freq : int
        The frequency of the notch filter; data will be notch-filtered at this frequency
        and at the corresponding harmonics,
        e.g. notch_freq = 50 Hz -> harmonics = [50, 100, 150, etc.]

    resampling_freq: int
        The frequency to resample the data after filtering and rescaling

    bandpass_cutoff_low : int
        Cut-off frequency at the lower end of the passband of the bandpass filter.

    bandpass_cutoff_high : int
        Cut-off frequency at the higher end of the passband of the bandpass filter.

    Returns
    -------
    numpy.ndarray[np.dtype[float]]
        2-dimensional numpy array containing the preprocessed data where the rows correspond to the input traces.
    """

    # Extract channel names
    channel_names = [trace.label for trace in traces]

    logger.debug(f"Channels processed by worker: {channel_names}")

    # Extract data sampling freq
    sfreq = traces[0].sfreq

    # Extract raw data from traces
    traces = np.array([trace.data for trace in traces])

    # 1. Bandpass filter
    logger.debug(
        f"Bandpass filter data between {bandpass_cutoff_low} and {bandpass_cutoff_high} Hz"
    )

    bandpass_filtered = filter_signal(
        sfreq=sfreq,
        cutoff_freq_low=bandpass_cutoff_low,
        cutoff_freq_high=bandpass_cutoff_high,
        data=traces,
    )

    # 2. Notch filter
    logger.debug(f"Apply notch filter at {notch_freq} Hz")
    notch_filtered = notch_filter_signal(
        eeg_data=bandpass_filtered,
        notch_frequency=notch_freq,
        low_pass_freq=bandpass_cutoff_high,
        sfreq=sfreq,
    )

    # 3. Scaling channels
    logger.debug("Rescale filtered data")
    scaled_data = rescale_data(
        data_to_be_scaled=notch_filtered, original_data=traces, sfreq=sfreq
    )

    # 4. Resampling data
    logger.debug(f"Resample data at sampling frequency {resampling_freq} Hz")
    resampled_data = resample_data(
        data=scaled_data,
        channel_names=channel_names,
        sfreq=sfreq,
        resampling_freq=resampling_freq,
    )

    return resampled_data
