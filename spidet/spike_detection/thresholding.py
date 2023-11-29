from typing import Dict, Tuple

import numpy as np
from loguru import logger


class ThresholdGenerator:
    def __init__(
        self,
        h_matrix: np.ndarray,
        preprocessed_data: np.ndarray = None,
        sfreq: int = 50,
        z_threshold: int = 10,
    ):
        self.preprocessed_data = preprocessed_data
        self.h_matrix = h_matrix
        self.sfreq = sfreq
        self.z_threshold = z_threshold

    def __determine_involved_channels(
        self, spikes_on: np.ndarray, spikes_off: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preprocessed_data is None:
            logger.warning(
                "Cannot determine involved channels as preprocessed data is None"
            )
            return np.array([]), spikes_on, spikes_off
        nr_events = len(spikes_on)

        # Return empty arrays if no events available
        if nr_events == 0:
            return tuple((np.array([]), np.array([]), np.array([])))
        nr_channels = self.preprocessed_data.shape[0]
        channels_involved = np.zeros((nr_events, nr_channels))

        # Calculate background
        background = np.zeros((self.preprocessed_data.shape[1]))

        if spikes_on[0] > 1:
            background[: spikes_on[0]] = 1

        for idx in range(nr_events - 1):
            background[spikes_off[idx] : spikes_on[idx + 1]] = 1

        # TODO: check why np.median returns all zeros
        # Get mean and standard deviation of background for each channel
        median_channels = np.median(
            self.preprocessed_data[:, background.nonzero()[0]], axis=1
        )
        std_channels = np.std(
            self.preprocessed_data[:, background.nonzero()[0]], axis=1
        )

        # For each event determine the involved channels
        for event in range(nr_events):
            event_window = self.preprocessed_data[
                :, spikes_on[event] : spikes_off[event]
            ]

            # Calculate z-scores for channels along the event window
            z_scores = (event_window - median_channels[:, None]) / std_channels[:, None]

            # Get maximum z-scores along event window and respective indices for each channel
            max_z, channel_lags = np.max(z_scores, axis=1), np.argmax(z_scores, axis=1)

            # Include channels having z-score higher than z-threshold
            channels = max_z > self.z_threshold

            if not any(channels):
                continue

            # Set value to maximum lag for channels not included
            not_included = np.nonzero((channels + 1) % 2)[0]
            channel_lags[not_included] = np.max(channel_lags)

            # Get the channel that first reaches max z-score
            min_lag = np.min(channel_lags)

            channels_involved[event, :] = channels * (channel_lags - min_lag + 1)

        if nr_channels > 50:
            # For large nr of channels, only consider events associated with multiple channels
            relevant_channels = [
                event
                for event in range(nr_events)
                if np.sum(channels_involved[event]) > 1
            ]
        else:
            # Remove events not associated with any channel
            relevant_channels = [
                event
                for event in range(nr_events)
                if np.sum(channels_involved[event]) > 0
            ]

        return (
            channels_involved[relevant_channels, :],
            spikes_on[relevant_channels],
            spikes_off[relevant_channels],
        )

    def generate_threshold(self) -> float:
        # TODO: add doc
        # Calculate number of bins
        nr_bins = min(round(0.1 * self.h_matrix.shape[1]), 1000)

        # Create histogram of data_matrix
        hist, bin_edges = np.histogram(self.h_matrix, bins=nr_bins)

        # TODO: check whether disregard bin 0 (Epitome)
        # Get rid of bin 0
        hist, bin_edges = hist[1:], bin_edges[1:]

        # Smooth hist with running mean of 10 dps
        hist_smoothed = np.convolve(hist, np.ones(10) / 10, mode="same")

        # Smooth hist 10 times with running mean of 3 dps
        for _ in range(10):
            hist_smoothed = np.convolve(hist_smoothed, np.ones(3) / 3, mode="same")

        # TODO: check whether disregard 10 last dp, depending on smoothing
        hist, hist_smoothed, bin_edges = (
            hist[:-10],
            hist_smoothed[:-10],
            bin_edges[:-10],
        )

        # Compute first differences
        first_diff = np.diff(hist_smoothed, 1)

        # Correct for size of result array of first difference, duplicate first value
        first_diff = np.append(first_diff[0], first_diff)

        # Smooth first difference matrix 10 times with running mean of 3
        # data points
        first_diff_smoothed = first_diff
        for _ in range(10):
            first_diff_smoothed = np.convolve(
                first_diff_smoothed, np.ones(3) / 3, mode="same"
            )

        # Get first 2 indices of localized modes in hist
        modes = np.nonzero(np.diff(np.sign(first_diff), 1) == -2)[0][:2]

        # Get index of first mode that is at least 10 dp to the right
        idx_mode = modes[modes > 9][0]

        # Index of first inflection point to the right of the mode
        idx_first_inf = np.argmin(first_diff_smoothed[idx_mode:])

        # Get index in original hist
        idx_first_inf += idx_mode - 1

        # Second difference of hist
        second_diff = np.diff(first_diff_smoothed, 1)

        # Correct for size of result array of differentiation, duplicate first column
        second_diff = np.append(second_diff[0], second_diff)

        # Get index of max value in second diff to the right of the first peak
        # -> corresponds to values around spikes
        idx_second_peak = np.argmax(second_diff[idx_first_inf:])

        # Get index in original hist
        idx_second_peak += idx_first_inf - 1

        # Fit a line in hist
        start_idx = np.max(
            [
                idx_mode,
                idx_first_inf
                - np.rint((idx_second_peak - idx_first_inf) / 2).astype(int),
            ],
        )
        end_idx = idx_second_peak

        if end_idx - start_idx <= 1:
            end_idx = [end_idx + 3, start_idx + 3][
                np.argmax(np.array([end_idx - start_idx + 3, 3]) > 2)
            ]
            logger.warning(
                f"End index for threshold line fit either before or too close to start index, modified to: {end_idx}"
            )

        threshold_fit = np.polyfit(
            bin_edges[start_idx:end_idx],
            hist_smoothed[start_idx:end_idx],
            deg=1,
        )

        threshold = -threshold_fit[1] / threshold_fit[0]

        return threshold

    def find_spikes(self, threshold: float) -> Dict[(int, Dict)]:
        # TODO: add doc

        # Create spike mask indicating whether specific time point belongs to spike
        spike_mask = self.h_matrix > threshold

        # Process rows of H sequentially
        spikes = dict()
        for idx, h_row in enumerate(spike_mask):
            # Find starting time points of spikes
            spikes_on = np.array(np.diff(np.append(0, h_row), 1) == 1).nonzero()[0]

            # Find ending time points of spikes (i.e. blocks of 1s)
            spikes_off = np.array(np.diff(np.append(0, h_row), 1) == -1).nonzero()[0]

            # Correct for any starting spike not ending within recording period
            if len(spikes_on) > len(spikes_off):
                spikes_on = spikes_on[:-1]

            spike_durations = spikes_off - spikes_on

            # Consider only events having a duration of at least 20 ms
            spikes_on = spikes_on[spike_durations >= 0.02 * self.sfreq]
            spikes_off = spikes_off[spike_durations >= 0.02 * self.sfreq]

            # Likewise, if gaps between events are < 40 ms, they are considered the same event
            gaps = spikes_on[1:] - spikes_off[:-1]
            gaps_mask = gaps >= 0.04 * self.sfreq
            spikes_on = spikes_on[np.append(1, gaps_mask).nonzero()[0]]
            spikes_off = spikes_off[np.append(gaps_mask, 1).nonzero()[0]]

            # Add +/- 40 ms on either side of the events, zeroing out any negative values
            # and upper bounding values by maximum time point
            spikes_on = np.maximum(0, spikes_on - 0.04 * self.sfreq).astype(int)
            spikes_off = np.minimum(len(h_row), spikes_off + 0.04 * self.sfreq).astype(
                int
            )

            # Determine which channels were involved in measuring which events
            (
                channel_event_assoc,
                spikes_on,
                spikes_off,
            ) = self.__determine_involved_channels(spikes_on, spikes_off)

            spikes.update(
                {
                    idx: dict(
                        {
                            "spikes_on": spikes_on,
                            "spikes_off": spikes_off,
                            "channels_involved": channel_event_assoc,
                        }
                    )
                }
            )

        return spikes
