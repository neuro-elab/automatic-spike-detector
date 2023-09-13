import multiprocessing
import os

import numpy as np
from numpy import genfromtxt


def generate_threshold(data_matrix):
    # TODO: add doc
    # Calculate number of bins
    nr_bins = min(round(0.1 * data_matrix.shape[1]), 1000)

    # Create histogram of data_matrix
    hist, bin_edges = np.histogram(data_matrix, bins=nr_bins)

    # TODO: check whether disregard bin 0 (Epitome)

    # Smooth each histogram 10 times with running mean of 3 data points
    hist_smoothed = hist
    for _ in range(10):
        hist_smoothed = np.convolve(hist_smoothed, np.ones(3) / 3, mode="same")

    # TODO: check whether disregard 10 last dp, depending on smoothing

    # Compute first differences
    first_diff = np.diff(hist, 1)

    # TODO: Epitome duplicates first dp in first diff
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
    idx_mode = modes[modes > 10][0]

    # Index of minimum value to the right of the mode
    idx_min_right = np.argmin(first_diff_smoothed[idx_mode:])

    # Get index in original hist
    idx_min_right += idx_mode - 1

    # Second difference of hist
    second_diff = np.diff(first_diff_smoothed, 1)

    # TODO: Epitome duplicates first dp in second diff, for array size
    # Correct for size of result array of differentiation, duplicate first column
    second_diff = np.append(second_diff[0], second_diff)

    # Get index of max value in second diff, right of the first peak
    # -> corresponds to values around spikes
    idx_second_peak = np.argmax(second_diff[idx_min_right:])

    # Get index in original hist
    idx_second_peak += idx_min_right - 1

    # Fit a line in hist
    threshold_fit = np.polyfit(
        bin_edges[
            idx_min_right
            - round((idx_second_peak - idx_min_right) / 2) : idx_second_peak
        ],
        hist_smoothed[
            idx_min_right
            - round((idx_second_peak - idx_min_right) / 2) : idx_second_peak
        ],
        deg=1,
    )

    threshold = -threshold_fit[1] / threshold_fit[0]

    return threshold


def annotate_spikes(k_dir: str):
    data_matrix = genfromtxt(k_dir + "/H_best.csv", delimiter=",")

    # generate threshold that separates spiking from non-spiking periods
    threshold = generate_threshold(data_matrix)

    # Get row wise indices in data matrix indicating data points belonging to a spike
    spike_annotations = [
        np.argwhere(data_matrix[row_idx, :] > threshold).T
        for row_idx in range(data_matrix.shape[0])
    ]
    return spike_annotations


def parallel_thresholding(experiment_dir: str):
    # TODO: add doc
    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        experiment_dir + "/" + k_dir
        for k_dir in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, k_dir)) and "k=" in k_dir
    ]

    # Using all available cores for process pool
    n_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=n_cores) as pool:
        spike_annotations = pool.map(
            annotate_spikes,
            rank_dirs,
        )

    return spike_annotations
