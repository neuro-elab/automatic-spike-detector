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
) -> np.ndarray:
    # TODO add documentation, clean up

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
