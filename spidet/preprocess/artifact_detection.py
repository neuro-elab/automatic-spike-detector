from datetime import datetime, timezone
from typing import List

import numpy as np
from loguru import logger
from collections import namedtuple

from spidet.domain.Artifacts import Artifacts
from spidet.domain.BadTimesType import BadTimesType
from spidet.domain.Trace import Trace
from spidet.load.data_loading import DataLoader
from spidet.utils.times_utils import compute_rescaled_timeline


class ArtifactDetector:
    @staticmethod
    def __merge_overlapping_bad_times(bad_times: np.ndarray) -> np.ndarray:
        logger.debug("Merging potentially overlapping bad times")
        BadTime = namedtuple("BadTime", "type index")
        bad_times_split = []

        # Split up bad times in ON and OFF parts and gather altogether in one array
        for interval in bad_times:
            bad_times_split.append(BadTime(BadTimesType.BAD_TIMES_ON, interval[0]))
            bad_times_split.append(BadTime(BadTimesType.BAD_TIMES_OFF, interval[1]))

        # Sort according to the index
        bad_times_split.sort(key=lambda bad_time: bad_time.index)

        merged = []
        current_on = None
        current_off = None

        # Merge overlapping bad times
        for time in bad_times_split:
            if BadTimesType.BAD_TIMES_ON == time.type:
                if current_on is None:
                    current_on = time
                elif current_off is not None:
                    merged.append([current_on.index, current_off.index])
                    current_on = time
                    current_off = None
            elif BadTimesType.BAD_TIMES_OFF == time.type:
                if current_off is None or current_off.index < time.index:
                    current_off = time

        return np.array(merged)

    @staticmethod
    def __detect_bad_times(
        data: np.ndarray[np.dtype[float]],
        sfreq: int,
    ) -> np.ndarray[np.dtype[np.int64]]:
        """
        Detects periods within the underlying EEG data that are considered bad, meaning that they represent
        some kind of artifact.

        Parameters
        ----------
        data : numpy.ndarray[numpy.dtype[float]]
            The underlying data containing the bad times (artifacts) periods.

        sfreq : int
            The frequency of the underlying data.

        Returns
        -------
        bad_times : numpy.ndarray[numpy.dtype[numpy.int64]]
            An array containing the periods detected as bad.
        """
        logger.debug("Computing bad times")
        bad_times = None
        times = data.shape[1]

        medians_channels = np.median(np.abs(data), axis=0)

        # Binary array indicating where channel medians pass critical threshold
        sat = medians_channels > 100 * np.median(medians_channels)

        # Binary array indicating where sum of channels is zero
        flat = np.sum(data, axis=0) == 0

        # Calculate start and end points of bad times
        on_bad_times = np.where(
            np.diff(np.concatenate([[False], np.bitwise_or(flat, sat)]).astype(int))
            == 1
        )[0].squeeze()

        off_bad_times = np.where(
            np.diff(np.concatenate([[False], np.bitwise_or(flat, sat)]).astype(int))
            == -1
        )[0].squeeze()

        # Correct for unequal number of elements
        if len(on_bad_times) > len(off_bad_times):
            off_bad_times = np.append(off_bad_times, times)
        elif len(off_bad_times) > len(on_bad_times):
            on_bad_times = np.append(0, on_bad_times)

        if on_bad_times.size != 0:
            # Extract periods between artifacts
            gaps = on_bad_times[1:] - off_bad_times[:-1]

            # Only consider gaps of a certain minimum length
            relevant_gaps = gaps >= 0.1 * sfreq

            on_indices = np.append(1, relevant_gaps).nonzero()[0]
            on_bad_times = on_bad_times[on_indices]

            off_indices = np.append(relevant_gaps, 1).nonzero()[0]
            off_bad_times = off_bad_times[off_indices]

            bad_times = np.vstack((on_bad_times, off_bad_times)).T

        logger.debug(
            f"Identified {0 if bad_times is None else bad_times.shape[0]} periods as bad times"
        )
        return bad_times

    @staticmethod
    def __detect_bad_channels(data: np.ndarray, bad_times: np.ndarray):
        logger.debug("Computing bad channels")

        nr_channels, times = data.shape

        # Binary array indicating which channels are considered empty
        empty_channels = np.sum(data == 0, axis=1) > 0.1 * data.shape[1]

        # Detect white noise
        if bad_times is not None:
            white_noise = np.zeros(times)
            for idx in range(bad_times.shape[0]):
                white_noise[bad_times[idx, 0] : bad_times[idx, 1]] = 1
            relevant_data = data[:, (1 - white_noise).nonzero()[0]]
        else:
            relevant_data = data

        sum_per_channel = np.sum(np.abs(relevant_data), axis=1)

        # Calculate the interquartile range
        q1 = np.percentile(relevant_data, 25)
        q3 = np.percentile(relevant_data, 75)
        iqr = q3 - q1

        # Transform the channel-sums in a z-score manner
        sum_per_channel = (sum_per_channel - np.median(sum_per_channel)) / iqr

        white_noise_channels = sum_per_channel > 3

        bad_channels = np.bitwise_or(
            np.zeros(nr_channels, dtype=bool), empty_channels, white_noise_channels
        )

        logger.debug(f"Identified {np.sum(bad_channels)} channels as potentially bad")
        return bad_channels

    @staticmethod
    def __detect_stimulation_artifacts(
        data: np.ndarray, bad_times: np.ndarray
    ) -> np.ndarray[np.dtype[np.int64]]:
        logger.debug("Detecting stimulation artifacts")

        # A stimulation has the same value along a channel and across channels
        stimulation = np.append(
            0, np.all(np.diff(data, axis=1) == 0, axis=0).astype(int)
        )

        # Find starting and ending points of stimulation
        stim_on = np.nonzero(np.diff(stimulation, 1) == 1)[0]
        stim_off = np.nonzero(np.diff(stimulation, 1) == -1)[0]

        # Correct for unequal number of elements
        if len(stim_on) > len(stim_off):
            stim_on = stim_on[:-1]

        if len(stim_off) > len(stim_on):
            stim_on = np.append(1, stim_on)

        if len(stim_on) == 0:
            return bad_times

        # Calculate gaps (periods between off and the next on)
        gaps = stim_on[1:] - stim_off[:-1]

        # Only consider gaps of a minimum length of 10 data points
        relevant_gaps = gaps >= 10

        on_indices = np.append(1, relevant_gaps).nonzero()[0]
        stim_on = stim_on[on_indices]

        off_indices = np.append(relevant_gaps, 1).nonzero()[0]
        stim_off = stim_off[off_indices]

        # Calculate durations of the periods of stimulation
        durations = stim_off - stim_on

        # Consider only periods of length at least 2 data points
        relevant_periods = np.nonzero(durations >= 2)[0]

        stim_on = stim_on[relevant_periods]
        stim_off = stim_off[relevant_periods]
        durations = durations[relevant_periods]

        max_duration = np.percentile(durations, 90)

        if len(stim_on) > 0:
            stim_periods = np.vstack((stim_on - 10, stim_off + max_duration)).T

            bad_times = (
                stim_periods
                if bad_times is None
                else np.vstack((bad_times, stim_periods))
            )

        return bad_times

    @staticmethod
    def __add_stimulation_trigger_times(
        trigger_times: List[str],
        bad_times: np.ndarray[np.dtype[np.int64]],
        times: np.ndarray[np.dtype[float]],
        sfreq: int,
    ) -> np.ndarray[np.dtype[int]]:
        """
        Creates an artifact window around each stimulation trigger by adding a 100ms window before
        and a 1sec window after the trigger time, and adds them to the existing artifact periods.

                            |
                            |
                            |
        ____________| 100ms |    1 sec   |____________


        Parameters
        ----------
        trigger_times : List[str]
            A list of datetime strings corresponding to the time points of trigger events.

        bad_times : numpy.ndarray[numpy.dtype[numpy.int64]]
            An array containing indices representing the start and end points of intervals
            associated with artifacts.

        times : numpy.ndarray[numpy.dtype[float]]
            Timestamps corresponding to the respective values along the x-axis of the underlying data.

        sfreq : int
            The frequency of the underlying data.

        Returns
        -------
        bad_times : numpy.ndarray[numpy.dtype[numpy.int64]]
            An array of artifact intervals, complemented by intervals encompassing respective
            stimulation events.

        """
        logger.debug(f"Adding {len(trigger_times)} trigger periods to bad times")
        # Map trigger times to timestamps
        trigger_timestamps = list(
            map(
                lambda trig: datetime.strptime(trig, "%Y-%m-%d %H:%M:%S.%f")
                .replace(tzinfo=timezone.utc)
                .timestamp(),
                trigger_times,
            )
        )

        # Map timestamps to indices
        trigger_indices = np.array(
            list(
                map(
                    lambda trig: np.argmin(np.abs(times - trig)),
                    trigger_timestamps,
                )
            )
        )

        # Adding a 100ms window before and a 1sec window after the trigger events
        trigger_periods = np.vstack(
            (
                np.maximum(0, trigger_indices - np.rint(0.1 * sfreq)),
                np.minimum(trigger_indices + np.rint(1 * sfreq), len(times) - 1),
            )
        ).T

        bad_times = (
            trigger_periods
            if bad_times is None
            else np.vstack((bad_times, trigger_periods))
        )

        return bad_times

    def run_on_data(
        self,
        data: np.ndarray,
        sfreq: int,
        times: np.ndarray = None,
        trigger_times: List[str] = None,
        detect_bad_times: bool = True,
        detect_bad_channels: bool = True,
        detect_stimulation_artifacts: bool = False,
    ) -> Artifacts:
        bad_times = None
        bad_channels = None

        logger.debug("Running artifact detection")

        # Calculate bad times, i.e. times corresponding to possible artifact
        if detect_bad_times:
            bad_times = self.__detect_bad_times(data, sfreq)

        # Calculate bad channels, i.e. channels that could be considered an artifact
        if detect_bad_channels:
            bad_channels = self.__detect_bad_channels(data, bad_times)

        # Calculate artifacts that might be induced by stimulation
        if detect_stimulation_artifacts:
            bad_times = self.__detect_stimulation_artifacts(data, bad_times)

        # If stimulations are present and need to exclude triggers
        if trigger_times is not None:
            bad_times = self.__add_stimulation_trigger_times(
                trigger_times, bad_times, times, sfreq
            )

        # Sort and merge potentially overlapping bad time periods
        if bad_times is not None:
            bad_times = self.__merge_overlapping_bad_times(bad_times)

        return Artifacts(bad_times=bad_times, bad_channels=bad_channels)

    def run(
        self,
        file_path: str,
        channel_paths: List[str],
        bipolar_reference: bool = False,
        leads: List[str] = None,
        trigger_times: List[str] = None,
        detect_bad_times: bool = True,
        detect_bad_channels: bool = True,
        detect_stimulation_artifacts: bool = False,
    ) -> Artifacts:
        # Read data from file
        data_loader = DataLoader()
        traces: List[Trace] = data_loader.read_file(
            path=file_path,
            channel_paths=channel_paths,
            bipolar_reference=bipolar_reference,
            leads=leads,
        )

        # Retrieve necessary information from traces
        data = np.array([trace.data for trace in traces])
        sfreq = traces[0].sfreq
        start_timestamp = traces[0].start_timestamp

        # Compute times corresponding to dta points
        times = compute_rescaled_timeline(
            start_timestamp=start_timestamp,
            length=data.shape[1],
            sfreq=sfreq,
        )

        # Perform artifact detection
        artifacts: Artifacts = self.run_on_data(
            data=data,
            sfreq=sfreq,
            times=times,
            trigger_times=trigger_times,
            detect_bad_times=detect_bad_times,
            detect_bad_channels=detect_bad_channels,
            detect_stimulation_artifacts=detect_stimulation_artifacts,
        )

        return artifacts
