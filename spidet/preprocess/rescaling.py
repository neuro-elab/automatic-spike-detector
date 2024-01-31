import numpy as np

from spidet.preprocess.filtering import filter_signal


def rescale_data(
    data_to_be_scaled: np.array, original_data: np.array, sfreq: int
) -> np.array:
    """
    Rescales the bandpass filtered data with respect to the original data matrix.

    Parameters
    ----------
    data_to_be_scaled : numpy.array
        Data to be scaled.

    original_data : numpy.array
        Original data.

    sfreq : float
        Sampling frequency of the channels in the original data.

    Returns
    -------
    array-like
        Scaled version of the filtered data.
    """
    # PROCESS ORIGINAL DATA
    # bandpass filter
    filtered_original_data = filter_signal(
        sfreq,
        cutoff_freq_low=8,
        cutoff_freq_high=30,
        data=original_data,
        zero_center=False,
    )

    # make data non-negative
    non_negative = np.abs(filtered_original_data)

    # vector of column means (i.e. means across channels for each point in time)
    col_means = np.mean(non_negative, 0)

    # consider only columns in original data with col_mean < median(col_means)
    processed_original_data = non_negative[:, col_means < np.median(col_means)]

    # SCALING STEPS
    # vector of means of channels (i.e. means across time for each channel)
    channel_means = np.mean(processed_original_data, 1)

    # calculate scaling parameter as the mean of the channel divided by 20
    scaling_param = np.mean(channel_means) / 20

    # normalize channel means by dividing by its own mean
    normalized_channel_means = channel_means / np.mean(channel_means)

    # scale and normalize data
    scaled_data = (
        scaling_param * data_to_be_scaled / normalized_channel_means[:, np.newaxis]
    )

    return scaled_data
