from typing import List

import numpy as np
from loguru import logger

from spidet.domain.Artifacts import Artifacts
from spidet.domain.Trace import Trace
from spidet.load.data_loading import DataLoader


class ArtifactDetector:
    @staticmethod
    def __detect_bad_times(
        data: np.ndarray,
        sfreq: int,
    ):
        logger.debug("Computing bad times")
        bad_times = None
        times = data.shape[1]

        medians_channels = np.median(np.abs(data), axis=0)

        # Binary array indicating where channel medians pass critical threshold
        sat = medians_channels > 100 * np.median(medians_channels)

        # Binary array indicating where sum of channels is zero
        flat = np.sum(data, axis=0) == 0

        # Calculate start and end points of bad times
        on_bad_times = np.where(
            np.diff(np.concatenate([[False], np.bitwise_or(flat, sat)]).astype(int))
            == 1
        )[0].squeeze()

        off_bad_times = np.where(
            np.diff(np.concatenate([[False], np.bitwise_or(flat, sat)]).astype(int))
            == -1
        )[0].squeeze()

        # Correct for unequal number of elements
        if len(on_bad_times) > len(off_bad_times):
            off_bad_times = np.append(off_bad_times, times)
        elif len(off_bad_times) > len(on_bad_times):
            on_bad_times = np.append(0, on_bad_times)

        if on_bad_times.size != 0:
            # Extract periods between artifacts
            gaps = on_bad_times[1:] - off_bad_times[:-1]

            # Only consider gaps of a certain minimum length
            relevant_gaps = gaps >= 0.1 * sfreq

            on_indices = np.append(1, relevant_gaps).nonzero()[0]
            on_bad_times = on_bad_times[on_indices]

            off_indices = np.append(relevant_gaps, 1).nonzero()[0]
            off_bad_times = off_bad_times[off_indices]

            bad_times = np.vstack((on_bad_times, off_bad_times)).T

        logger.debug(f"Identified {bad_times.shape[0]} periods as bad times")
        return bad_times

    @staticmethod
    def __detect_bad_channels(data: np.ndarray, bad_times: np.ndarray):
        logger.debug("Computing bad channels")

        nr_channels, times = data.shape

        # Binary array indicating which channels are considered empty
        empty_channels = np.sum(data == 0, axis=1) > 0.1 * data.shape[1]

        # Detect white noise
        if bad_times is not None:
            white_noise = np.zeros(times)
            for idx in range(bad_times.shape[0]):
                white_noise[bad_times[idx, 0] : bad_times[idx, 1]] = 1
            relevant_data = data[:, (1 - white_noise).nonzero()[0]]
        else:
            relevant_data = data

        sum_per_channel = np.sum(np.abs(relevant_data), axis=1)

        # Calculate the interquartile range
        q1 = np.percentile(relevant_data, 25)
        q3 = np.percentile(relevant_data, 75)
        iqr = q3 - q1

        # Transform the channel-sums in a z-score manner
        sum_per_channel = (sum_per_channel - np.median(sum_per_channel)) / iqr

        white_noise_channels = sum_per_channel > 3

        bad_channels = np.bitwise_or(
            np.zeros(nr_channels, dtype=bool), empty_channels, white_noise_channels
        )

        logger.debug(f"Identified {np.sum(bad_channels)} channels as potentially bad")
        return bad_channels

    def run_on_data(
        self,
        data: np.ndarray,
        sfreq: int,
        detect_bad_times: bool = True,
        detect_bad_channels: bool = True,
    ) -> Artifacts:
        bad_times = None
        bad_channels = None

        logger.debug("Running artifact detection")

        # Calculate bad times, i.e. times corresponding to possible artifact
        if detect_bad_times:
            bad_times = self.__detect_bad_times(data, sfreq)

        # Calculate bad channels, i.e. channels that could be considered an artifact
        if detect_bad_channels:
            bad_channels = self.__detect_bad_channels(data, bad_times)

        return Artifacts(bad_times=bad_times, bad_channels=bad_channels)

    def run(self, file_path: str, channel_paths: List[str]) -> Artifacts:
        # Read data from file
        data_loader = DataLoader()
        traces: List[Trace] = data_loader.read_file(
            path=file_path, dataset_paths=channel_paths
        )

        # Perform artifact detection
        sfreq = traces[0].sfreq
        artifacts: Artifacts = self.run_on_data(
            data=np.array([trace.data for trace in traces]), sfreq=sfreq
        )

        return artifacts
