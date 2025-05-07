import numpy as np

from spidet.preprocess.filtering import filter_signal, notch_filter_signal
from spidet.preprocess.resampling import resample_data
from spidet.preprocess.rescaling import rescale_data


def apply_preprocessing_steps(
    channel_names: list,
    sfreq: int,
    data: np.ndarray,
    notch_freq: int,
    resampling_freq: int,
    bandpass_cutoff_low: int,
    bandpass_cutoff_high: int,
) -> np.ndarray[np.dtype[float]]:
    """
    Applies the necessary preprocessing steps to the original iEEG data. This involves:

        1.  Bandpass-filtering with a butterworth forward-backward filter of order 2
        2.  Notch-filtering
        3.  Rescaling
        4.  Resampling

    Parameters
    ----------
    channel_names : list
        The channel names as strings corresponding to the rows in data. The number of channel names should be equal to the number of rows in data.

    data : np.ndarray
        The data matrix representing eeg traces to be preprocessed. The data will be rowwise scaled and filtered.

    notch_freq : int
        The frequency of the notch filter; data will be notch-filtered at this frequency
        and at the corresponding harmonics,
        e.g. notch_freq = 50 Hz -> harmonics = [50, 100, 150, etc.]

    resampling_freq: int
        The frequency to resample the data after filtering and rescaling

    bandpass_cutoff_low : int
        Cut-off frequency at the lower end of the passband of the bandpass filter.

    bandpass_cutoff_high : int
        Cut-off frequency at the higher end of the passband of the bandpass filter.

    Returns
    -------
    numpy.ndarray[np.dtype[float]]
        2-dimensional numpy array containing the preprocessed data where the rows correspond to the input traces.
    """

    # 1. Bandpass filter
    processed = filter_signal(
        sfreq=sfreq,
        cutoff_freq_low=bandpass_cutoff_low,
        cutoff_freq_high=bandpass_cutoff_high,
        data=data,
    )

    # 2. Notch filter
    processed = notch_filter_signal(
        eeg_data=processed,
        notch_frequency=notch_freq,
        low_pass_freq=bandpass_cutoff_high,
        sfreq=sfreq,
    )

    # 3. Scaling channels
    processed = rescale_data(
        data_to_be_scaled=processed, original_data=data, sfreq=sfreq
    )

    # 4. Resampling data
    processed = resample_data(
        data=processed,
        channel_names=channel_names,
        sfreq=sfreq,
        resampling_freq=resampling_freq,
    )

    return processed
