import multiprocessing
from typing import List, Tuple

import numpy as np
from loguru import logger
from scipy.signal.windows import hann

from spidet.domain.ActivationFunction import ActivationFunction
from spidet.domain.Trace import Trace
from spidet.load.data_loading import DataLoader
from spidet.preprocess.preprocessing import apply_preprocessing_steps
from spidet.preprocess.resampling import resample_data
from spidet.spike_detection.thresholding import ThresholdGenerator
from spidet.utils.times_utils import compute_rescaled_timeline


class LineLength:
    """
    This class provides all operations regarding the line-length transformation.

    Parameters
    ----------

    file_path: str
        Path to the file containing the iEEG data.

    bad_times: numpy.ndarray[numpy.dtype[float]]
        An optional N x 2 numpy array containing periods that must be excluded before applying
        the line-length transformation. Each of th N rows in the array represents a period to be excluded,
        defined by the start and end indices of the period in the original iEEG data.
        The defined periods will be set to zero with the transitions being smoothed by applying a hanning window
        to prevent spurious patterns.

    dataset_paths: List[str], mandatory when the file is in .h5 format
        A list of paths to the traces to be included within an h5 file. This is only necessary in the case
        of h5 files.

    bipolar_reference: bool, optional, default = False
        If True, the bipolar references of the included channels will be computed. If channels already are
        in bipolar form this needs to be False.

    exclude: List[str], optional
        A list of channel names that need to be excluded. This only applies in the case of .edf and .fif files.

    leads: List[str]
        A list of the leads included. Only necessary if bipolar_reference is True, otherwise can be None.
    """

    def __init__(
        self,
        file_path: str,
        dataset_paths: List[str] = None,
        exclude: List[str] = None,
        bipolar_reference: bool = False,
        leads: List[str] = None,
        bad_times: np.ndarray = None,
    ):
        self.file_path = file_path
        self.dataset_paths = dataset_paths
        self.exclude = exclude
        self.bipolar_reference = bipolar_reference
        self.leads = leads
        self.bad_times = bad_times
        self.line_length_window: int = 40
        self.line_length_freq: int = 50

    def dampen_bad_times(
        self,
        data: np.ndarray[np.dtype[float]],
        sfreq: int,
        orig_sfreq: int,
        window_length: int = 100,
    ) -> np.ndarray:
        """
        Dampens bad times within preprocessed iEEG data by setting values of bad times intervals to zero
        and applying hann windows (https://en.wikipedia.org/wiki/Hann_function) around starting and ending
        points in order to get smoothed transitions

        Parameters
        ----------
        data : numpy.ndarray[np.dtype[float]]
            The preprocessed iEEG data.

        sfreq : int
            The sampling frequency of the preprocessed iEEG data.

        orig_sfreq : int
            The sampling frequency of the original iEEG data.

        window_length : int, optional, default = 100
            The length of the smoothed transition periods in milliseconds

        Returns
        -------
        smoothed_data : numpy.ndarray[np.dtype[float]]
            The preprocessed iEEG data wih artifacts being zeroed and having smoothed transition periods.

        """
        if len(self.bad_times.shape) == 1:
            self.bad_times = self.bad_times[np.newaxis, :]

        self.bad_times = np.rint(self.bad_times * sfreq / orig_sfreq).astype(int)

        # Create window
        window = 2 * np.rint(window_length / 1000 * sfreq).astype(int)

        # Make window length even
        window = window if window % 2 == 0 else window + 1

        # Create hanning window
        hann_w = 1 - hann(window)

        left_hann = hann_w[0 : int(window / 2)]
        right_hann = hann_w[int(window / 2) + 1 :]

        # Bound to limits
        self.bad_times[:, 0] = np.maximum(
            1 + window / 2, self.bad_times[:, 0] - window / 2
        )
        self.bad_times[:, 1] = np.minimum(
            data.shape[1] - window / 2, self.bad_times[:, 1] + window / 2
        )

        # Create the hann mask matrix
        hann_mask = np.ones(data.shape)
        for event_idx in range(self.bad_times.shape[0]):
            hann_mask[
                :,
                int(self.bad_times[event_idx, 0] - window / 2) : int(
                    self.bad_times[event_idx, 1] + window / 2
                ),
            ] = np.hstack(
                (
                    left_hann,
                    np.zeros((np.diff(self.bad_times[event_idx]).astype(int)[0] + 1)),
                    right_hann,
                )
            )
        return hann_mask * data

    def compute_line_length(self, eeg_data: np.ndarray, sfreq: int):
        """
        Performs the line-length transformation on the input EEG data..

        Parameters
        ----------
        eeg_data : numpy.ndarray
            Input EEG data.

        sfreq : int
            Sampling frequency of the input EEG data.

        Returns
        -------
        numpy.ndarray[Any,
            Line length representation of the input EEG data.

        Notes
        -----
        The line length operation involves slicing the input data into evenly spaced intervals
        along the time axis and processing each block separately. It computes the summed absolute
        difference of the data along consecutive time points over a predefined segment. [1]_

        References
        ----------
        .. [1]
        Koolen, N., Jansen, K., Vervisch, J., Matic, V., De Vos, M., Naulaers, G., & Van Huffel, S. (2014).
        Line length as a robust method to detect high-activity events:
        Automated burst detection in premature EEG recordings.
        Clinical Neurophysiology, 125(10), 1985–1994. https://doi.org/https://doi.org/10.1016/j.clinph.2014.02.015
        """
        # shape of the data: number of channels x duration
        nr_channels, duration = np.shape(eeg_data)

        # window size for line length calculations, default 40 ms
        window = self.line_length_window

        # effective window size: round to nearest even in the data points
        w_eff = 2 * round(sfreq * window / 2000)

        # to optimize computation, calculations are performed on intervals built from 40000 evenly spaced
        # discrete time points along the duration of the signal
        time_points = np.round(
            np.linspace(0, duration - 1, max(2, round(duration / 40000)))
        ).astype(dtype=int)
        line_length_eeg = np.empty((nr_channels, time_points.take(-1)))

        # iterate over time points
        for idx in range(len(time_points) - 1):
            # extract a segment of eeg data containing the data of a single time interval
            # (i.e. time_points[idx] up to time_points[idx + 1])
            if idx == len(time_points) - 2:
                eeg_interval = np.concatenate(
                    (
                        eeg_data[:, time_points[idx] : time_points[idx + 1]],
                        np.zeros((nr_channels, w_eff)),
                    ),
                    axis=1,
                )
            else:
                # add a pad to the time dimension of size w_eff
                eeg_interval = np.array(
                    eeg_data[:, time_points[idx] : time_points[idx + 1] + w_eff]
                )

            # build cuboid containing w_eff number of [nr_channels, interval_length]-planes,
            # where each plane is shifted by a millisecond w.r.t. the preceding plane
            eeg_cuboid = np.empty((eeg_interval.shape[0], eeg_interval.shape[1], w_eff))
            for j in range(w_eff):
                eeg_cuboid[:, :, j] = np.concatenate(
                    (eeg_interval[:, j:], np.zeros((nr_channels, j))), axis=1
                )

            # perform line length computations
            line_length_interval = np.nansum(np.abs(np.diff(eeg_cuboid, 1, 2)), 2)

            # remove the pad
            line_length_eeg[
                :, time_points[idx] : time_points[idx + 1]
            ] = line_length_interval[:, : line_length_interval.shape[1] - w_eff]

        # center the data a window
        line_length_eeg = np.concatenate(
            (
                np.zeros((nr_channels, np.floor(w_eff / 2).astype(int))),
                line_length_eeg[:, : -np.ceil(w_eff / 2).astype(int)],
            ),
            axis=1,
        )

        return line_length_eeg

    def line_length_pipeline(
        self,
        traces: List[Trace],
        notch_freq: int,
        resampling_freq: int,
        bandpass_cutoff_low: int,
        bandpass_cutoff_high: int,
    ) -> np.ndarray:
        # Extract channel names
        channel_names = [trace.label for trace in traces]

        # Preprocess the data
        preprocessed = apply_preprocessing_steps(
            traces=traces,
            notch_freq=notch_freq,
            resampling_freq=resampling_freq,
            bandpass_cutoff_low=bandpass_cutoff_low,
            bandpass_cutoff_high=bandpass_cutoff_high,
        )

        # Zero out bad times if any
        if self.bad_times is not None:
            logger.debug("Dampening bad times on preprocessed EEG with hann windows")
            preprocessed = self.dampen_bad_times(
                data=preprocessed, sfreq=resampling_freq, orig_sfreq=traces[0].sfreq
            )

        # Compute line length
        logger.debug("Apply line length computations")
        line_length = self.compute_line_length(
            eeg_data=preprocessed, sfreq=resampling_freq
        )

        # Downsample to line_length_freq (default 50 Hz)
        logger.debug(f"Resample line length at {self.line_length_freq} Hz")
        line_length_resampled_data = resample_data(
            data=line_length,
            channel_names=channel_names,
            sfreq=resampling_freq,
            resampling_freq=self.line_length_freq,
        )

        # Resampling produced some negative values, replace by 0
        line_length_resampled_data[line_length_resampled_data < 0] = 0

        return line_length_resampled_data

    def apply_parallel_line_length_pipeline(
        self,
        notch_freq: int = 50,
        resampling_freq: int = 500,
        bandpass_cutoff_low: int = 0.1,
        bandpass_cutoff_high: int = 200,
        line_length_freq: int = 50,
        line_length_window: int = 40,
    ) -> Tuple[float, List[str], np.ndarray[np.dtype[float]]]:
        """
        This function launches the line length pipeline, which first carries out the necessary preprocessing steps
        and then performs the line-length transformation of the preprocessed EEG data. The individual steps include

            1.  reading the data from the provided file (supported file formats are .h5, .edf, .fif)
                using the :py:mod:`~spidet.load.data_loading` module, which transforms the data
                into a list of :py:mod:`~spidet.domain.Trace` objects,
            2.  performing the necessary preprocessing steps by means of the
                :py:mod:`~spidet.preprocess.preprocessing` module,
            3.  and applying the line-length transformation.

        To optimize computation, the channels are split into subsets and processed in parallel.

        Parameters
        ----------
        notch_freq: int, optional, default = 50
            The frequency of the notch filter; data will be notch-filtered at this frequency
            and at the corresponding harmonics,
            e.g. notch_freq = 50 Hz -> harmonics = [50, 100, 150, etc.]

        resampling_freq: int, optional, default = 500
            The frequency to resample the data after filtering and rescaling

        bandpass_cutoff_low: int, optional, default = 0.1
            Cut-off frequency at the lower end of the passband of the bandpass filter.

        bandpass_cutoff_high: int, optional, default = 200
            Cut-off frequency at the higher end of the passband of the bandpass filter.

        line_length_freq: int, optional, default = 50
            Sampling frequency of the line-length transformed data

        line_length_window: int, optional, default = 40
            Window length used to for the line-length operation (in milliseconds).

        Returns
        -------
        Tuple[float, List[str], numpy.ndarray[numpy.dtype[float]]]
            Tuple containing, the start timestamp of the recording, a list of channel names
            corresponding to the channels in the line-length transformed data,
            the line-length transformed data
        """
        # Set optional line length params
        self.line_length_freq = line_length_freq
        self.line_length_window = line_length_window

        # Load the eeg traces from the given file
        data_loader = DataLoader()
        start_timestamp = None
        labels = []
        line_length_list = []

        # Sequentially load, preprocess and line-length transform subsets of channels due to memory limitations
        nr_channel_subsets = (
            1 if len(self.dataset_paths) // 10 == 0 else len(self.dataset_paths) // 10
        )
        for channel_set in np.array_split(self.dataset_paths, nr_channel_subsets):
            traces: List[Trace] = data_loader.read_file(
                self.file_path,
                list(channel_set),
                self.exclude,
                self.bipolar_reference,
                self.leads,
            )
            # Extract the channel names
            labels.extend([trace.label for trace in traces])

            # Start time of the recording
            start_timestamp = traces[0].start_timestamp

            # Using all available cores for process pool
            n_cores = multiprocessing.cpu_count()

            # Define the number of parallel process used for preprocessing and line-length transformation
            n_processes = min(5, len(channel_set))

            with multiprocessing.Pool(processes=n_cores) as pool:
                line_length = pool.starmap(
                    self.line_length_pipeline,
                    [
                        (
                            data,
                            notch_freq,
                            resampling_freq,
                            bandpass_cutoff_low,
                            bandpass_cutoff_high,
                        )
                        for data in np.array_split(traces, n_processes)
                    ],
                )

            # Combine results from parallel processing
            line_length_subset = np.concatenate(line_length, axis=0)
            line_length_list.append(line_length_subset)

        return start_timestamp, labels, np.concatenate(line_length_list, axis=0)

    def compute_unique_line_length(
        self,
        notch_freq: int = 50,
        resampling_freq: int = 500,
        bandpass_cutoff_low: int = 0.1,
        bandpass_cutoff_high: int = 200,
        n_processes: int = 5,
        line_length_freq: int = 50,
        line_length_window: int = 40,
    ) -> ActivationFunction:
        """
        This function computes the standard deviation of the data after performing
        the line-length transformation using the :py:func:`apply_parallel_line_length_pipeline` method
        and wraps it into a single :py:class:`~spidet.domain.ActivationFunction` object.
        The defined parameters will be passed on to the :py:func:`apply_parallel_line_length_pipeline` method.

        Parameters
        ----------
        notch_freq: int, optional, default = 50
            The frequency of the notch filter; data will be notch-filtered at this frequency
            and at the corresponding harmonics,
            e.g. notch_freq = 50 Hz -> harmonics = [50, 100, 150, etc.]

        resampling_freq: int, optional, default = 500
            The frequency to resample the data after filtering and rescaling

        bandpass_cutoff_low: int, optional, default = 0.1
            Cut-off frequency at the lower end of the passband of the bandpass filter.

        bandpass_cutoff_high: int, optional, default = 200
            Cut-off frequency at the higher end of the passband of the bandpass filter.

        n_processes: int, optional, default = 5
            Number of parallel processes to use for the line-length pipeline

        line_length_freq: int, optional, default = 50
            Sampling frequency of the line-length transformed data

        line_length_window: int, optional, default = 40
            Window length used to for the line-length operation (in milliseconds).

        Returns
        -------
        :py:class:`~spidet.domain.ActivationFunction`
            ActivationFunction containing the standard deviation of the line-length transformed data.
        """
        # Compute line length for each channel (done in parallel)
        start_timestamp, _, line_length = self.apply_parallel_line_length_pipeline(
            notch_freq=notch_freq,
            resampling_freq=resampling_freq,
            bandpass_cutoff_low=bandpass_cutoff_low,
            bandpass_cutoff_high=bandpass_cutoff_high,
            n_processes=n_processes,
            line_length_freq=line_length_freq,
            line_length_window=line_length_window,
        )

        # Compute standard deviation between line length channels which is our unique line length
        std_line_length = np.std(line_length, axis=0)

        # Compute times for x-axis
        times = compute_rescaled_timeline(
            start_timestamp=start_timestamp,
            length=line_length.shape[1],
            sfreq=line_length_freq,
        )

        # Generate threshold and detect periods
        threshold_generator = ThresholdGenerator(
            activation_function_matrix=std_line_length, sfreq=line_length_freq
        )
        threshold = threshold_generator.generate_threshold()
        detected_periods = threshold_generator.find_events(threshold)

        # Create unique id
        filename = self.file_path[self.file_path.rfind("/") + 1 :]
        unique_id = f"{filename[:filename.rfind('.')]}_std_line_length"

        return ActivationFunction(
            label="Std Line Length",
            unique_id=unique_id,
            times=times,
            data_array=std_line_length,
            detected_events_on=detected_periods.get(0)["events_on"],
            detected_events_off=detected_periods.get(0)["events_off"],
            event_threshold=threshold,
        )
