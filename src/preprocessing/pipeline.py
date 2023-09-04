import numpy as np

from src.domain.Trace import Trace
from src.preprocessing.filtering import filter_signal, notch_filter_signal
from src.preprocessing.rescaling import rescale_data
from src.preprocessing.resampling import resample_data
from src.preprocessing.line_length import apply_line_length


def apply_preprocessing_steps(trace: Trace):
    #TODO add documentation

    # 1. bandpass filter
    bandpass_filtered = filter_signal(trace.sfreq, cutoff_freq_low=1, cutoff_freq_high=200, data=trace.data)

    # 2. notch filter
    notch_filtered = notch_filter_signal(eeg_data=bandpass_filtered, notch_frequency=50, low_pass_freq=200, sfreq=trace.sfreq)

    # 3. scaling channels
    scaled_data = rescale_data(data_to_be_scaled=notch_filtered, original_data=trace.data, sfreq=trace.sfreq)

    # 4. resampling data
    resampled_data = resample_data(data=scaled_data, resampling_freq=500)

    # 5. compute line length
    line_length_eeg = apply_line_length(eeg_data=resampled_data, sfreq=trace.sfreq)

    return line_length_eeg
