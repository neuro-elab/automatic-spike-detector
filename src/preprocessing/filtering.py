import mne
import numpy as np
from scipy import signal


def filter_signal(sfreq: int, cutoff_freq_low: int, cutoff_freq_high: int, data: np.array, zero_center: bool = True) \
        -> np.array:
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
    # Nyquist frequency (i.e. half the sampling frequency)
    nyq = sfreq / 2

    # cut-off frequencies
    f_l = cutoff_freq_low
    f_h = cutoff_freq_high

    # Normalize frequency
    norm_freq = np.array(f_l, f_h) / nyq

    # create an iir (infinite impulse response) butterworth filter
    iir_params = dict(order=2, ftype="butter")
    iir_filtered = mne.filter.create_filter(
        data, nyq, l_freq=norm_freq[0], h_freq=norm_freq[1], method="iir", iir_params=iir_params, btype="bandpass",
        verbose=True
    )

    # forward-backward filter
    filtered_eeg = signal.sosfiltfilt(iir_filtered["sos"], data)

    if zero_center:
        # zero-center the data
        filtered_eeg -= np.median(filtered_eeg, 1, keepdims=True)

    return filtered_eeg
