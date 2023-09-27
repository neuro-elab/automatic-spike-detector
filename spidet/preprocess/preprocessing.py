import multiprocessing
from typing import List

import numpy as np
from loguru import logger

from spidet.domain.Trace import Trace
from spidet.domain.PreprocessedData import PreprocessedData
from spidet.preprocess.filtering import filter_signal, notch_filter_signal
from spidet.preprocess.line_length import apply_line_length
from spidet.preprocess.resampling import resample_data
from spidet.preprocess.rescaling import rescale_data
from spidet.load.data_loading import read_file


class Preprocessor:
    def __init__(
        self,
        file_path: str,
        dataset_paths: List[str] = None,
        bipolar_reference: bool = False,
        leads: List[str] = None,
    ):
        self.file_path = file_path
        self.dataset_paths = dataset_paths
        self.bipolar_reference = bipolar_reference
        self.leads = leads

    def apply_preprocessing_steps(
        self,
        traces: List[Trace],
        notch_freq: int,
        resampling_freq: int,
        bandpass_cutoff_low: int,
        bandpass_cutoff_high: int,
    ) -> np.ndarray:
        # TODO add documentation, clean up

        # Channel names
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

        # 5. Compute line length
        logger.debug("Apply line length computations")
        line_length = apply_line_length(eeg_data=resampled_data, sfreq=resampling_freq)

        # 6. Downsample to 50 hz
        logger.debug("Resample line length at 50 Hz")
        resampled_data = resample_data(
            data=line_length,
            channel_names=channel_names,
            sfreq=resampling_freq,
            resampling_freq=50,
        )

        # Resampling produced some negative values, replace by 0
        resampled_data[resampled_data < 0] = 0

        return resampled_data

    def parallel_preprocessing(
        self,
        notch_freq: int = 50,
        resampling_freq: int = 500,
        bandpass_cutoff_low: int = 0.1,
        bandpass_cutoff_high: int = 200,
        n_processes: int = 8,
        line_length_freq: int = 50,
    ) -> PreprocessedData:
        # Load the eeg traces from the given file
        traces: List[Trace] = read_file(
            self.file_path, self.dataset_paths, self.bipolar_reference, self.leads
        )

        # TODO: add documentation
        logger.debug(
            f"Starting preprocess pipeline on {n_processes} parallel processes"
        )

        # Using all available cores for process pool
        n_cores = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=n_cores) as pool:
            preprocessed_data = pool.starmap(
                self.apply_preprocessing_steps,
                [
                    (
                        data,
                        notch_freq,
                        resampling_freq,
                        bandpass_cutoff_low,
                        bandpass_cutoff_high,
                    )
                    for data in np.array_split(traces, n_processes)
                ],
            )

        line_length = np.concatenate(preprocessed_data, axis=0)

        # Compute standard deviation between line length traces
        std_line_length = np.std(line_length, axis=0)

        # Compute rescaled timeline
        start_timestamp = traces[0].start_timestamp
        times = (
            np.linspace(start=0, stop=line_length.shape[1], num=line_length.shape[1])
            / line_length_freq
        )
        times = times + start_timestamp

        # Channel names
        channel_names = [trace.label for trace in traces]

        logger.debug("Preprocessing pipeline finished successfully, returning data")
        return PreprocessedData(channel_names, times, line_length, std_line_length)
