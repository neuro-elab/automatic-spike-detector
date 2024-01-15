import multiprocessing
from typing import List, Tuple, Any

import numpy as np
from loguru import logger
from scipy.signal.windows import hann

from spidet.domain.DetectionFunction import DetectionFunction
from spidet.domain.Trace import Trace
from spidet.load.data_loading import DataLoader
from spidet.preprocess.preprocessing import apply_preprocessing_steps
from spidet.preprocess.resampling import resample_data
from spidet.spike_detection.thresholding import ThresholdGenerator
from spidet.utils.times_utils import compute_rescaled_timeline


class LineLength:
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
        data: np.ndarray[Any, np.dtype[np.float64]],
        sfreq: int,
        orig_sfreq: int,
        window_length: int = 100,
    ) -> np.ndarray:
        """
        Dampens bad times within preprocessed EEG data by setting values of bad times intervals to zero
        and applying hann windows (https://en.wikipedia.org/wiki/Hann_function) around starting and ending
        points in order to get smoothed transitions

        Parameters
        ----------
        data : numpy.ndarray[Any, np.dtype[np.float64]]
            The preprocessed EEG data.

        sfreq : int
            The sampling frequency of the preprocessed EEG data.

        orig_sfreq : int
            The sampling frequency of the original EEG data.

        window_length : int
            The length of the smoothed transition periods in ms

        Returns
        -------
        smoothed_data : numpy.ndarray[Any, np.dtype[np.float64]]
            The preprocessed EEG data wih artifacts being zeroed and having smoothed transition periods.

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

    def compute_line_length(self, eeg_data: np.array, sfreq: int):
        """
        Computes the line length of the input EEG data.

        Parameters
        ----------
        eeg_data : numpy.ndarray
            Input EEG data.

        sfreq : float
            Frequency of the input EEG data.

        Returns
        -------
        numpy.ndarray
            Line length representation of the input EEG data.

        Notes
        -----
        The line length operation involves slicing the input data into evenly spaced intervals
        along the time axis and processing each block separately. It computes the summed absolute
        difference of the data along consecutive time points over a predefined segment.

        References
        ----------
        See also
        --------
        :see: Line length as a robust method to detect high-activity events:
              Automated burst detection in premature EEG recordings
        (https://www.sciencedirect.com/science/article/pii/S1388245714001114?via%3Dihub).
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
        n_processes: int = 5,
        line_length_freq: int = 50,
        line_length_window: int = 40,
    ) -> Tuple[float, List[str], np.ndarray]:
        # Set optional line length params
        self.line_length_freq = line_length_freq
        self.line_length_window = line_length_window

        # Load the eeg traces from the given file
        data_loader = DataLoader()
        start_timestamp = None
        labels = []
        line_length_list = []

        # Sequential data loading due to memory limitations
        for channel_set in np.array_split(self.dataset_paths, n_processes):
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
    ) -> DetectionFunction:
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
            detection_function_matrix=std_line_length, sfreq=line_length_freq
        )
        threshold = threshold_generator.generate_threshold()
        detected_periods = threshold_generator.find_events(threshold)

        # Create unique id
        filename = self.file_path[self.file_path.rfind("/") + 1 :]
        unique_id = f"{filename[:filename.rfind('.')]}_std_line_length"

        return DetectionFunction(
            label="Std Line Length",
            unique_id=unique_id,
            times=times,
            data_array=std_line_length,
            detected_events_on=detected_periods.get(0)["events_on"],
            detected_events_off=detected_periods.get(0)["events_off"],
            event_threshold=threshold,
        )
