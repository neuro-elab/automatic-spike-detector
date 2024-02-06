# automatic-spike-detection

The automatic-spike-detection package is a Python library for automatically detecting interictal epileptiform
discharges (IEDs) in intracranial EEG (iEEG) recordings.


IEDs, also referred to as ”spikes”, are a characteristic of
the epileptic brain that are recognizable as large transient events in the electroencephalogram
of patients living with epilepsy [[1]](#1). Whereas, for clinicians, IEDs can provide valuable information
about the epileptogenic zone, for researchers, they can also be a source of noise and need to be excluded,
such as in [[2]](#2) where Cusinato and Alnes et al. studied how the human brain processes sounds. Regardless of
the context, the localization of IEDs in EEG recordings is a very time-consuming task.

This package aims to contribute to this issue by building on an algorithm previously developed by
Baud et al. [[3]](#3) that employs nonnegative matrix factorization (NMF) to automatically detect IEDs,
an unsupervised machine-learning algorithm that produces a lower-dimensional approximation of the input.

It is important to note, that the algorithm used by this package is optimized for and was solely tested
on iEEG recordings. Intracranial EEG is an invasive technique with implanted electrodes
that is used for clinical monitoring, e.g. to identify the epileptogenic zone and prepare for epilepsy surgery.
The primary characteristic of iEEG is that it provides high spatial and temporal resolution of the
electrical activity in the brain, which makes it a valuable resource for neuroscientific research as well [[4]](#4).

Please consult the automatic-spike-detection [Documentation](https://automatic-spike-detection.readthedocs.io/en/latest/index.html)
for details on the [underlying concepts](https://automatic-spike-detection.readthedocs.io/en/latest/documentation/index.html)
of the algorithm, the [Installation and Usage](https://automatic-spike-detection.readthedocs.io/en/latest/usage/index.html#usage)
, and the application programming interface [(API)](https://automatic-spike-detection.readthedocs.io/en/latest/reference/index.html)

## Contributions
In addition to the dependencies in the `requirements.txt` file, you need to install the dependencies listed in the
`dev-requirements.txt` file, which provides some formatting tools:

````
pip install requirements.txt
pip install dev-requirements.txt
````

## References
<a id="1">[1]</a>
Marco de Curtis and Giuliano Avanzini. "Interictal spikes in focal epileptogenesis".
Progress in Neurobiology 63, no.5 (2001): 541-567.

<a id="2">[2]</a>
Riccardo Cusinato, Sigurd L. Alnes, Ellen van Maren, Ida Boccalaro, Debora Ledergerber, Antoine
Adamantidis, Lukas L. Imbach, Kaspar Schindler, Maxime O. Baud, and Athina Tzovara. Intrinsic
neural timescales in the temporal lobe support an auditory processing hierarchy. Journal of
Neuroscience, 43(20):3696–3707, 2023.

<a id="3">[3]</a>
Maxime O. Baud, Jonathan K. Kleen, Gopala K. Anumanchipalli, Liberty S. Hamilton, Yee-Leng
Tan, Robert Knowlton, and Edward F. Chang. Unsupervised learning of spatiotemporal interictal
discharges in focal epilepsy. Neurosurgery, 83(4), 2018.

<a id="4">[4]</a>
Elizabeth L Johnson, Julia W Y Kam, Athina Tzovara, and Robert T Knight. Insights into human
cognition from intracranial eeg: A review of audition, memory, internal cognition, and causality.
Journal of Neural Engineering, 17(5):051001, oct 2020.
