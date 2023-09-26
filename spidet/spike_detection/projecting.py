from typing import Tuple, List

import numpy as np


# TODO add doc, refactor
class Projector:
    def __init__(
        self,
        h_matrix: np.ndarray,
        w_matrix: np.ndarray,
        downsampling_freq: int = 500,
        ll_freq: int = 50,
    ):
        self.h_matrix = h_matrix
        self.w_matrix = w_matrix
        self.rank = self.h_matrix.shape[0]
        self.duration = self.h_matrix.shape[1]
        self.downsampling_freq = downsampling_freq
        self.line_length_freq = ll_freq
        self.rate = self.downsampling_freq / self.line_length_freq
        self.window = 0.5 * self.line_length_freq

    def __compute_peaks_per_activation_vector(self, matrix: np.ndarray) -> np.ndarray:
        zeros = np.zeros((self.rank, 1))
        first_diff = np.diff(matrix, 1, 1)
        signs_first_diff = np.sign(np.concatenate((zeros, first_diff), axis=1))
        first_diff_signs = np.diff(signs_first_diff, 1, 1)
        peaks_mask = np.concatenate(
            (
                np.bitwise_and(first_diff_signs == -2, matrix[:, :-1] > 0),
                zeros,
            ),
            axis=1,
        )
        return peaks_mask

    def find_and_project_peaks(
        self, preprocessed_data: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        h = self.h_matrix
        n_channels = preprocessed_data.shape[0]

        # Difference to other BFs -> most distinctive features
        h_differences = np.zeros(h.shape)
        for idx in range(self.rank):
            h_differences = h[idx, :] - np.sum(h[np.arange(self.rank) != idx, :], 0)

        # Compute peaks per activation vector in h_differences and h respectively
        peaks_mask_h_diff = self.__compute_peaks_per_activation_vector(h_differences)
        peaks_mask_h = self.__compute_peaks_per_activation_vector(h)

        # PROJECT peaks onto the preprocessed data matrix
        data_projections = list()
        w_projection = np.zeros((n_channels, 100))
        for idx in range(self.rank):
            if peaks_mask_h_diff[idx, :].any():
                peaks_mask = peaks_mask_h_diff[idx, :]
            else:
                # Some basis functions are never activated above the others
                peaks_mask = peaks_mask_h[idx, :]

            # Get indices of peaks, corresponds to peak times
            indices_peaks = np.nonzero(peaks_mask)[0].squeeze()

            # Bound content of array to range (1, duration)
            indices_peaks = indices_peaks[
                (indices_peaks - self.window >= 1)
                & (indices_peaks + self.window <= self.duration)
            ]

            # Get number of peaks, at least 100
            n_peaks = min(100, len(indices_peaks))

            # Retrieve peaks from row idx in H matrix
            peaks = h[idx, indices_peaks]

            # Sort peaks in ascending order and get index array
            idx_sorted_peaks = np.argsort(peaks)

            # Get sorted peak times
            sorted_peak_times = indices_peaks[idx_sorted_peaks[:n_peaks]]

            samples = np.zeros((n_channels, 2 * self.window + 1, n_peaks))

            for idx_peak in range(n_peaks):
                # Project onto preprocessed data
                samples[:, :, idx_peak] = preprocessed_data[
                    :,
                    sorted_peak_times[idx_peak]
                    - self.window : sorted_peak_times[idx_peak]
                    + self.window,
                ]

            # Create mask for involved channels from W
            involved = self.w_matrix[:, idx] > np.median(self.w_matrix.flatten())

            # Set channels (rows of preprocessed_data) not involved in basis functions to NaN
            samples[not involved, :, :] = np.nan

            data_projections.append(samples)
            w_projection = np.concatenate(
                (
                    w_projection,
                    20 * np.mean(samples, axis=2),
                    np.zeros((n_channels, 100)),
                ),
                axis=1,
            )

        return w_projection, data_projections
