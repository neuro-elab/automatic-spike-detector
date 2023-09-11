import multiprocessing
from typing import List

import numpy as np
from loguru import logger

from src.domain.Trace import Trace
from src.preprocessing.filtering import filter_signal, notch_filter_signal
from src.preprocessing.line_length import apply_line_length
from src.preprocessing.resampling import resample_data
from src.preprocessing.rescaling import rescale_data


def apply_preprocessing_steps(
    traces: List[Trace],
    notch_freq: int,
    resampling_freq: int,
    bandpass_cutoff_low: int,
    bandpass_cutoff_high: int,
):
    # TODO add documentation, clean up

    # channel names
    channel_names = [trace.label for trace in traces]

    # frequency of data
    data_freq = traces[0].sfreq

    # extract data from traces
    traces = np.array([trace.data for trace in traces])

    # 1. bandpass filter
    logger.debug(
        f"Bandpass filter data between {bandpass_cutoff_low} and {bandpass_cutoff_high} Hz"
    )

    bandpass_filtered = filter_signal(
        sfreq=data_freq,
        cutoff_freq_low=bandpass_cutoff_low,
        cutoff_freq_high=bandpass_cutoff_high,
        data=traces,
    )

    # 2. notch filter
    logger.debug(f"Apply notch filter at {notch_freq} Hz")
    notch_filtered = notch_filter_signal(
        eeg_data=bandpass_filtered,
        notch_frequency=notch_freq,
        low_pass_freq=bandpass_cutoff_high,
        sfreq=data_freq,
    )

    # 3. scaling channels
    logger.debug("Rescale filtered data")
    scaled_data = rescale_data(
        data_to_be_scaled=notch_filtered, original_data=traces, sfreq=data_freq
    )

    # 4. resampling data
    logger.debug(f"Resample data at sampling frequency {resampling_freq} Hz")

    resampled_data = resample_data(
        data=scaled_data,
        channel_names=channel_names,
        sfreq=data_freq,
        resampling_freq=resampling_freq,
    )

    # 5. compute line length
    logger.debug("Apply line length computations")
    line_length_eeg = apply_line_length(eeg_data=resampled_data, sfreq=data_freq)

    return line_length_eeg


def parallel_preprocessing(
    traces: List[Trace],
    notch_freq: int = 50,
    resampling_freq: int = 500,
    bandpass_cutoff_low: int = 1,
    bandpass_cutoff_high: int = 200,
    n_processes: int = 4,
):
    # TODO: add documentation
    logger.debug(f"Starting preprocessing pipeline on {n_processes} parallel processes")

    # Using all available cores for process pool
    n_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=n_cores) as pool:
        preprocessed_data = pool.starmap(
            apply_preprocessing_steps,
            [
                (
                    data,
                    notch_freq,
                    resampling_freq,
                    bandpass_cutoff_low,
                    bandpass_cutoff_high,
                )
                for data in np.array_split(traces, 4)
            ],
        )

    data = np.concatenate(preprocessed_data, axis=0)
    logger.debug("Preprocessing pipeline finished successfully, returning data")
    return data
