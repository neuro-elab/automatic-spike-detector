import numpy as np


def apply_line_length(eeg_data: np.array, sfreq: int):
    """
    Performs the line length operation on the eeg data. Slices the data into evenly spaced intervals
    along the time axis and processes each block separately. For details on the line length operation see
    The line length is the summed absolute difference of the data along consecutive time points over a
    predefined segment.

    :param eeg_data: input data
    :param sfreq: frequency of the input eeg
    :return: line length representation of the input eeg
    """
    # shape of the data: number of channels x duration
    nr_channels, duration = np.shape(eeg_data)

    # window size for line length calculations, default 40 ms
    window = 40

    # effective window size: round to nearest even in the data points
    w_eff = 2 * round(sfreq * window / 2000)

    # to optimize computation, calculations are performed on intervals built from 40000 evenly spaced
    # discrete time points along the duration of the signal
    time_points = np.round(
        np.linspace(0, duration - 1, max(2, round(duration / 40000)))
    ).astype(dtype=int)
    line_length_eeg = np.empty((nr_channels, time_points.take(-1)))

    # iterate over time points
    for idx in range(len(time_points) - 1):
        # extract a segment of eeg data containing the data of a single time interval
        # (i.e. time_points[idx] up to time_points[idx + 1])
        if idx == len(time_points) - 2:
            eeg_interval = np.concatenate(
                (
                    eeg_data[:, time_points[idx] : time_points[idx + 1]],
                    np.zeros((nr_channels, w_eff)),
                ),
                axis=1,
            )
        else:
            # add a pad to the time dimension of size w_eff
            eeg_interval = np.array(
                eeg_data[:, time_points[idx] : time_points[idx + 1] + w_eff]
            )

        # build cuboid containing w_eff number of [nr_channels, interval_length]-planes,
        # where each plane is shifted by a millisecond w.r.t. the preceding plane
        eeg_cuboid = np.empty((eeg_interval.shape[0], eeg_interval.shape[1], w_eff))
        for j in range(w_eff):
            # TODO check whether we need j-1 or simply j in the nan part
            eeg_cuboid[:, :, j] = np.concatenate(
                (eeg_interval[:, j:], np.zeros((nr_channels, j))), axis=1
            )

        # perform line length computations
        line_length_interval = np.nansum(np.abs(np.diff(eeg_cuboid, 1, 2)), 2)

        # remove the pad
        line_length_eeg[
            :, time_points[idx] : time_points[idx + 1]
        ] = line_length_interval[:, : line_length_interval.shape[1] - w_eff]

    # center the data a window
    line_length_eeg = np.concatenate(
        (
            np.zeros((nr_channels, np.floor(w_eff / 2).astype(int))),
            line_length_eeg[:, : -np.ceil(w_eff / 2).astype(int)],
        ),
        axis=1,
    )

    # TODO: check whether downsampling for line length should be available (in Epitome only done when freq < 50 Hz)

    return line_length_eeg
