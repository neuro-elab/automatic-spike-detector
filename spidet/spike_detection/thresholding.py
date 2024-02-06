from typing import Dict, Tuple

import numpy as np
from loguru import logger


class ThresholdGenerator:
    """
    This class is the primary entity for computing detected events on a given single activation function or
    set of activation functions.

    Parameters
    ----------
    activation_function_matrix: numpy.ndarray[numpy.dtype[float]]
        A single or set of activation functions for which to compute events

    preprocessed_data: np.ndarray[numpy.dtype[float]]
        The preprocessed iEEG data, produced by applying the preprocessing steps listed in the preprocessing section.

    sfreq: int
        The sampling frequency of the data contained in the activation functions, defaults to 50 Hz.

    z_threshold: int
        The z-threshold used for computing the channels involved in a particular event.

    """

    def __init__(
        self,
        activation_function_matrix: np.ndarray[np.dtype[float]],
        preprocessed_data: np.ndarray[np.dtype[float]] = None,
        sfreq: int = 50,
        z_threshold: int = 10,
    ):
        self.activation_function_matrix = (
            activation_function_matrix
            if len(activation_function_matrix.shape) > 1
            else activation_function_matrix[np.newaxis, :]
        )
        self.preprocessed_data = preprocessed_data
        self.sfreq = sfreq
        self.z_threshold = z_threshold
        self.thresholds = dict()

    def __determine_involved_channels(
        self, events_on: np.ndarray, events_off: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preprocessed_data is None:
            logger.warning(
                "Cannot determine involved channels as preprocessed data is None"
            )
            return np.array([]), events_on, events_off
        if len(events_on) == 0:
            logger.debug(
                "Cannot determine involved channels as as no events were found"
            )
            return np.array([]), events_on, events_off

        nr_events = len(events_on)

        # Return empty arrays if no events available
        if nr_events == 0:
            return tuple((np.array([]), np.array([]), np.array([])))
        nr_channels = self.preprocessed_data.shape[0]
        channels_involved = np.zeros((nr_events, nr_channels))

        # Calculate background
        background = np.zeros((self.preprocessed_data.shape[1]))

        if events_on[0] > 1:
            background[: events_on[0]] = 1

        for idx in range(nr_events - 1):
            background[events_off[idx] : events_on[idx + 1]] = 1

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
                :, events_on[event] : events_off[event]
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
            relevant_events = [
                event
                for event in range(nr_events)
                if np.sum(channels_involved[event]) > 1
            ]
        else:
            # Remove events not associated with any channel
            relevant_events = [
                event
                for event in range(nr_events)
                if np.sum(channels_involved[event]) > 0
            ]

        return (
            channels_involved[relevant_events, :],
            events_on[relevant_events],
            events_off[relevant_events],
        )

    def generate_individual_thresholds(self) -> None:
        """
        Computes the threshold for each individual activation function based on
        :py:func:`~generate_threshold`
        """
        for idx, activation_function in enumerate(self.activation_function_matrix):
            threshold = self.generate_threshold(data=activation_function)
            self.thresholds.update({idx: threshold})

    def generate_threshold(self, data: np.ndarray[np.dtype[float]] = None) -> float:
        """
        Computes the threshold for individual activation functions. The threshold is defined as the
        zero-crossing of the line that is fitted to the right of the histogram of a given
        activation function.

        Parameters
        ----------

        data: np.ndarray[np.dtype[float]]
            This represents the data for which to compute the threshold. If None, the threshold is computed
            for the activation_function_matrix passed to the ThresholdGenerator at initialization.

        Returns
        -------
        float
            The threshold computed for either the data passed as a function argument or the activation function
            passed to the ThresholdGenerator at initialization.

        """
        # TODO: add doc
        # Determine data to compute threshold for
        data = data if data is not None else self.activation_function_matrix

        # Calculate number of bins
        nr_bins = min(round(0.1 * data.shape[-1]), 1000)

        # Create histogram of data_matrix
        hist, bin_edges = np.histogram(data, bins=nr_bins)

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
        candidates = modes[np.where((modes > 9) & (modes < len(bin_edges) / 10))]
        idx_mode = modes[0] if len(candidates) == 0 else candidates[0]

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

    def find_events(self, threshold: float = None) -> Dict[(int, Dict)]:
        """
        Computes the events for the activation functions in the activation_function_matrix, which was
        passed to the ThresholdGenerator at initialization. If the threshold argument is None,
        the computation is based on the thresholds generated for each activation function
        by :py:func:`~generate_individual_thresholds`

        Parameters
        ----------
        threshold: float
            The threshold used to compute events for the activation_function_matrix. This can be useful e.g.
            if the activation_function_matrix contains a single activation function and events need to be
            computed based on a custom threshold.

        Returns
        -------
        Dict[(int, Dict)]
            A nested dictionary containing the events for each activation function. A given activation function
            in the dictionary can be accessed by its respective index in the :py:attr:`activation_function_matrix`.
            The events for a given activation function are represented by a dictionary containing two index arrays
            corresponding to the onset, accessible by the "events_on"-key, and offset,
            accessible by the "events_off"-key, indices of the events, and one binary masking array
            indicating the indices of all detected events, accessible via the "event_mask"-key.
        """
        # Process rows sequentially
        events = dict()
        for idx, activation_function in enumerate(self.activation_function_matrix):
            # Determine threshold
            threshold = threshold if threshold is not None else self.thresholds.get(idx)

            # Create event mask indicating whether specific time point belongs to event
            event_mask = activation_function > threshold

            # Find starting time points of events
            events_on = np.array(np.diff(np.append(0, event_mask), 1) == 1).nonzero()[0]

            # Find ending time points of events (i.e. blocks of 1s)
            events_off = np.array(np.diff(np.append(0, event_mask), 1) == -1).nonzero()[
                0
            ]

            # Correct for any starting event not ending within recording period
            if len(events_on) > len(events_off):
                events_on = events_on[:-1]

            event_durations = events_off - events_on

            # Consider only events having a duration of at least 20 ms
            events_on = events_on[event_durations >= 0.02 * self.sfreq]
            events_off = events_off[event_durations >= 0.02 * self.sfreq]

            # Likewise, if gaps between events are < 40 ms, they are considered the same event
            gaps = events_on[1:] - events_off[:-1]
            gaps_mask = gaps >= 0.04 * self.sfreq
            channel_event_assoc = []
            if not len(events_on) == 0:
                events_on = events_on[np.append(1, gaps_mask).nonzero()[0]]
                events_off = events_off[np.append(gaps_mask, 1).nonzero()[0]]

                # Add +/- 40 ms on either side of the events, zeroing out any negative values
                # and upper bounding values by maximum time point
                events_on = np.maximum(0, events_on - 0.04 * self.sfreq).astype(int)
                events_off = np.minimum(
                    len(event_mask) - 1, events_off + 0.04 * self.sfreq
                ).astype(int)

                # Merge overlapping events
                gaps = events_on[1:] - events_off[:-1]
                gaps_mask = gaps > 0
                events_on = events_on[np.append(1, gaps_mask).nonzero()[0]]
                events_off = events_off[np.append(gaps_mask, 1).nonzero()[0]]

                # Determine which channels were involved in measuring which events
                (
                    channel_event_assoc,
                    events_on,
                    events_off,
                ) = self.__determine_involved_channels(events_on, events_off)

            # Create event mask
            event_mask = np.zeros(len(activation_function))
            for on, off in zip(events_on, events_off):
                event_mask[on : off + 1] = 1

            events.update(
                {
                    idx: dict(
                        {
                            "events_on": events_on,
                            "events_off": events_off,
                            "event_mask": event_mask.astype(int),
                            "channels_involved": channel_event_assoc,
                        }
                    )
                }
            )

        return events
