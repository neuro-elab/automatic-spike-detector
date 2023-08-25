import mne
import numpy as np
from scipy import signal


def filter_signal(sfreq: int, data: np.array) -> np.array:
    """
    Filter the provided signal with a low-pass butterworth forward-backward filter
    at cut-off frequency 200 Hz

    :param sfreq: sample frequency of the input signal/-s
    :param data: signal/-s to be filtered
    :return: low-pass filtered signal at cut-off frequency 200 Hz
    """
    # Nyquist frequency
    nyq = sfreq / 2

    # cut-off frequencies
    cutoff_freq_low = 1
    cutoff_freq_high = 200

    # Normalize frequency
    norm_freq = np.array(cutoff_freq_low, cutoff_freq_high) / nyq

    # create an iir (infinite impulse response) butterworth filter
    iir_params = dict(order=2, ftype="butter")
    iir_filtered = mne.filter.create_filter(
        data, nyq, l_freq=norm_freq[0], h_freq=norm_freq[1], method="iir", iir_params=iir_params, verbose=True
    )

    # forward-backward filter
    data_steep = signal.sosfiltfilt(iir_filtered["sos"], data)

    return data_steep
