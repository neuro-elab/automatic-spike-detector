.. module:: spidet

.. _usage:

=====
Usage
=====

This section gives instructions on how to install the automatic-spike-detection package and contains
examples of how to use it.

Installation
^^^^^^^^^^^^
The automatic-spike-detection package is hosted on `Python Package Index (PyPi) <https://pypi.org/>`_ repository and can be installed
with the `package installer for Python <https://pip.pypa.io/en/stable/>`_ ``pip`` via

.. code-block:: bash

    pip install automatic-spike-detection

and updated via

.. code-block:: bash

    pip install automatic-spike-detection --upgrade


Code Examples
^^^^^^^^^^^^^

The :class:`~spidet.spike_detection.spike_detection_pipeline.SpikeDetectionPipeline` class builds the core entity
of the automatic-spike-detection package and provides a complete pipeline for spike detection that includes

    1.  reading the data from the provided file (supported file formats are .h5, .edf, .fif) and
        transforming the data into a list of :mod:`~spidet.domain.Trace` objects,
    2.  performing the necessary preprocessing steps by means of the :mod:`~spidet.preprocess.preprocessing` module,
    3.  applying the line-length transformation using the :mod:`~spidet.spike_detection.line_length` module,
    4.  performing Nonnegative Matrix Factorization to extract the most discriminating metappatterns,
        done by the :mod:`~spidet.spike_detection.nmf` module and
    5.  computing periods of abnormal activity by means of the :mod:`~spidet.spike_detection.thresholding` module.

where preprocessing is optimized for the task of detection spikes.

An application example could look like

.. code-block:: Python

    # Define the file path
    file: str  = "/home/User/intracranial_EEG_recording.h5"

    # Define the sparseness parameter in [0, 1] if NMF should run with sparseness constraints
    # Note that running NMF with sparseness constraints typically increases running time
    sparseness: float = 0.25

    # Set the number of NMF runs performed for each rank in the rank_range
    # This is optional as there is a default of 100
    nmf_runs = 100

    # Set the range of rank for which to perform NMF
    # This, again, is optional and the default includes ranks [2, 3, 4, 5]
    k_min = 2
    k_max = 5

    # Initialize the spike detection pipeline
    spike_detection_pipeline = SpikeDetectionPipeline(
        file_path=file,
        save_nmf_matrices=True,
        sparseness=sparseness,
        nmf_runs=runs_per_rank,
        rank_range=(k_min, k_max),
    )

    # In case of an .h5 file, the channel paths within the file need to be defined
    channel_paths: List[str] = [
        "/traces/raw_bipolar/lead/Amy/Amy01-Amy02",
        "/traces/raw_bipolar/lead/Amy/Amy02-Amy03",
        "/traces/raw_bipolar/lead/Amy/Amy03-Amy04",
        "/traces/raw_bipolar/lead/Amy/Amy04-Amy05",
        ...]

    # Run the detection pipeline
    basis_functions: List[BasisFunction], activation_functions: List[ActivationFunction] =
        spike_detection_pipeline.run(
            channel_paths=channel_paths,
        )

Pleas check out the :ref:`API Reference <reference>` for further details on how to use the :class:`~spidet.spike_detection.spike_detection_pipeline.SpikeDetectionPipeline`.
