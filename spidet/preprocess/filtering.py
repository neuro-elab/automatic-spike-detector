import mne
import numpy as np
from scipy import signal


def filter_signal(
    sfreq: int,
    cutoff_freq_low: int,
    cutoff_freq_high: int,
    data: np.array,
    zero_center: bool = True,
) -> np.array:
    """
    Filter the provided signal with a bandpass butterworth forward-backward filter
    at specified cut-off frequencies. The order of the butterworth filter is predefined to be 2,
    which effectively results in an order of 4 as the data is forward-backward filtered.
    Additionally, the possibility to zero-center the data is provided.

    Parameters
    ----------
    sfreq : int
        Sampling frequency of the input signal/-s.

    cutoff_freq_low : int
        Lower end of the frequency passband.

    cutoff_freq_high : int
        Upper end of the frequency passband.

    data : array-like
        Signal/-s to be filtered.
    zero_center : bool, optional
        If True, re-centers the signal/-s, defaults to True.

    Returns
    -------
    array-like
        Bandpass filtered zero-centered signal/-s at cut-off frequency 200 Hz.
    """
    # Create an iir (infinite impulse response) butterworth filter
    iir_params = dict(order=2, ftype="butter", btype="bandpass")
    iir_filter = mne.filter.create_filter(
        data,
        sfreq,
        l_freq=cutoff_freq_low,
        h_freq=cutoff_freq_high,
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # Forward-backward filter
    filtered_eeg = signal.sosfiltfilt(iir_filter["sos"], data)

    if zero_center:
        # Zero-center the data
        filtered_eeg -= np.median(filtered_eeg, 1, keepdims=True)

    return filtered_eeg


def notch_filter_signal(
    eeg_data: np.array, notch_frequency: int, low_pass_freq: int, sfreq: int
):
    """
    Creates a notch-filter and runs it over the provided data.

    Parameters
    ----------
    eeg_data : array-like
        Data to be filtered.

    notch_frequency : int
        The frequency of the notch filter; data will be notch-filtered at this frequency
        and at the corresponding harmonics,
        e.g. notch_freq = 50 Hz -> harmonics = [50, 100, 150, etc.]

    low_pass_freq : int
        Frequency above which the signal is ignored.

    sfreq : int
        Baseline frequency of the signal.

    Returns
    -------
    array-like
        Filtered signal.
    """
    # get harmonics of the notch frequency within low pass freq, max first 4 harmonics
    harmonics = np.arange(notch_frequency, low_pass_freq, notch_frequency)
    harmonics = harmonics[:4] if harmonics.size > 4 else harmonics

    eeg_data = mne.filter.notch_filter(
        x=eeg_data, Fs=sfreq, freqs=harmonics, verbose=False
    )

    return eeg_data
