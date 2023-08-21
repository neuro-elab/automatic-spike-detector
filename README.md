# automatic-spike-detection

This module performs automatic spike detection on eeg data. The input data already went through some preprocessing, though further steps need to be applied to prepare for spike detection.

### Preprocessing
Steps performed within this module to prepare for  automatic spike detection:
1. Bandpass filter 1 - 200 Hz
2. Rescaling
3. Resampling
4. Line length

### Detection algorithms
After data preparation, the module offers the following methods to perform automatic spike detection:
* Non-negative matrix factorization (NMF)