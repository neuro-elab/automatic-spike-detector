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

    :param sfreq: sampling frequency of the input signal/-s
    :param cutoff_freq_low: lower end of the frequency passband
    :param cutoff_freq_high: upper end of the frequency passband
    :param data: signal/-s to be filtered
    :param zero_center: if True, re-centers the signal/-s, defaults to True
    :return: bandpass filtered zero-centered signal/-s at cut-off frequency 200 Hz
    """
    # TODO: remove nyq

    # Nyquist frequency (i.e. half the sampling frequency)
    nyq = sfreq / 2

    # cut-off frequencies
    f_l = cutoff_freq_low
    f_h = cutoff_freq_high

    # TODO: how should normalized freq be used with mne, if at all

    # Normalize frequency
    np.array([f_l, f_h]) / nyq

    # create an iir (infinite impulse response) butterworth filter
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

    # forward-backward filter
    filtered_eeg = signal.sosfiltfilt(iir_filter["sos"], data)

    if zero_center:
        # zero-center the data
        filtered_eeg -= np.median(filtered_eeg, 1, keepdims=True)

    return filtered_eeg


def notch_filter_signal(
    eeg_data: np.array, notch_frequency: int, low_pass_freq: int, sfreq: int
):
    """
    Creates a notch-filter and runs it over the provided data

    :param eeg_data: data to be filtered
    :param notch_frequency: frequency (and its harmonics) to filter
    :param low_pass_freq: frequency above which signal is ignored
    :param sfreq: baseline frequency of the signal
    :return: filtered signal
    """
    # TODO complete/rework documentation; check for correctness

    # get harmonics of the notch frequency within low pass freq, max first 4 harmonics
    harmonics = np.arange(notch_frequency, low_pass_freq, notch_frequency)
    harmonics = harmonics[:4] if harmonics.size > 4 else harmonics

    eeg_data = mne.filter.notch_filter(
        x=eeg_data, Fs=sfreq, freqs=harmonics, verbose=False
    )

    return eeg_data
