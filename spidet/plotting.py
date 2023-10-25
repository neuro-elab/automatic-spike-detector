import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from numpy import genfromtxt

from spidet.load.data_loading import DataLoader
from spidet.utils.variables import (
    LEAD_PREFIXES_008,
    DATASET_PATHS_008,
)
from spidet.utils import plotting_utils

SZ_LABEL = "Sz"


if __name__ == "__main__":
    # Parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", help="folder where the results reside", required=True
    )
    parser.add_argument(
        "--annotations", help="path to annotations file", required=False
    )

    folder: str = parser.parse_args().folder
    annotations: str = parser.parse_args().annotations

    # Set plotting variables
    plot_h: bool = True
    plot_w: bool = False
    plot_line_length: bool = True
    plot_seizures = False
    plot_unique_line_length = False
    plot_metrics: bool = True

    # Define data set
    leads = LEAD_PREFIXES_008
    datasets = DATASET_PATHS_008

    # Set seizure params
    offset_gaps = [
        timedelta(minutes=0),
        timedelta(minutes=20),
        timedelta(minutes=22),
        timedelta(minutes=23),
    ]
    durations = np.array([33, 7, 4, 2], dtype=int) * 60
    seizure_start_offsets = {
        1: [timedelta(hours=1, minutes=51) + offset for offset in offset_gaps],
        2: [timedelta(hours=3, minutes=54) + offset for offset in offset_gaps],
        3: [
            timedelta(hours=5, minutes=12, seconds=30) + offset
            for offset in offset_gaps
        ],
        4: [
            timedelta(hours=7, minutes=14, seconds=40) + offset
            for offset in offset_gaps
        ],
    }

    # Set labels if known
    rank_labels_idx = dict(
        {
            4: dict({3: SZ_LABEL}),
            5: dict({4: SZ_LABEL}),
            6: dict({0: SZ_LABEL}),
            7: dict({3: SZ_LABEL}),
            8: dict({7: SZ_LABEL}),
            9: dict({2: SZ_LABEL}),
            10: dict({9: SZ_LABEL}),
        }
    )

    # Set start time of the recording
    start_time_recording: datetime = datetime(2023, 5, 10, 22, 00, 38)
    # start_time_recording: datetime = datetime(2023, 2, 15, 1, 20, 28)

    # Set params for single plotting periods
    offset = timedelta(hours=0, minutes=0, seconds=0)
    duration = 2 * 60
    display_all = True
    y_lim = 1e-9

    # Get spike annotations if available
    if annotations is not None:
        df_annotations = pd.read_csv(annotations)
        df_annotations["description"] = df_annotations["description"].str.lower()
        spike_times = list(
            (
                df_annotations[df_annotations["description"].str.startswith("x")][
                    "onset"
                ]
            ).values
        )
        spike_times = list(
            map(
                lambda spike: datetime.strptime(spike, "%Y-%m-%d %H:%M:%S.%f"),
                spike_times,
            )
        )
    else:
        spike_times = None

    # Get list of channel names
    anodes, cathodes = DataLoader().get_anodes_and_cathodes(
        leads, DataLoader().extract_channel_names(datasets)
    )
    channel_names = [anode + "-" + cathode for anode, cathode in zip(anodes, cathodes)]

    # Plot W matrices
    if plot_w:
        rank_dirs = plotting_utils.get_rank_dirs_sorted(folder)
        w_matrices = []
        consensus_matrices = []
        for idx in range(len(rank_dirs)):
            logger.debug(
                f"{rank_dirs[idx][rank_dirs[idx].rfind('/') + 1:]}: Loading w and consensus matrices"
            )

            file_path_w = rank_dirs[idx] + "/W_best.csv"
            w_best = genfromtxt(file_path_w, delimiter=",")
            w_matrices.append(w_best)

            file_path_consensus = rank_dirs[idx] + "/consensus_matrix.csv"
            consensus = genfromtxt(file_path_consensus, delimiter=",")
            consensus_matrices.append(consensus)

        plotting_utils.plot_w_and_consensus_matrix(
            w_matrices=w_matrices,
            consensus_matrices=consensus_matrices,
            experiment_dir=folder,
            channel_names=channel_names,
            rank_labels_idx=rank_labels_idx,
        )

    # PLot metrics
    if plot_metrics:
        metrics = pd.read_csv(os.path.join(folder, "metrics.csv"))
        plotting_utils.plot_metrics(metrics, folder)

    # Get line length eeg if necessary
    if plot_line_length:
        logger.debug("Loading line length data")
        file_path_data = os.path.join(folder, "line_length.csv")
        line_length_eeg = genfromtxt(file_path_data, delimiter=",")

    # Plot std line length
    if plot_unique_line_length:
        file_path_data = os.path.join(folder, "std_line_length.csv")
        std_line_length = genfromtxt(file_path_data, delimiter=",")

    # Get H matrices if necessary
    if plot_h:
        rank_dirs = plotting_utils.get_rank_dirs_sorted(folder)
        h_matrices = []
        for idx in range(len(rank_dirs)):
            logger.debug(
                f"{rank_dirs[idx][rank_dirs[idx].rfind('/')+1:]}: Loading h matrices"
            )
            file_path = rank_dirs[idx] + "/H_best.csv"
            h_best = genfromtxt(file_path, delimiter=",")
            h_matrices.append(h_best)

    # If true, produce several plots around seizures with different scales
    if plot_seizures:
        if display_all:
            logger.warning(
                "Seizure mode is active, ignore display_all (currently True)"
            )

        for seizure in seizure_start_offsets.keys():
            for dur, start_offset in zip(durations, seizure_start_offsets.get(seizure)):
                # PLot preprocessed data
                if plot_line_length:
                    plotting_utils.plot_line_length_data(
                        folder,
                        line_length_eeg,
                        channel_names,
                        lead_prefixes=leads,
                        start_time_recording=start_time_recording,
                        offset=start_offset,
                        duration=dur,
                        seizure=seizure,
                    )

                if plot_unique_line_length:
                    plotting_utils.plot_std_line_length(
                        folder,
                        std_line_length,
                        start_time_recording=start_time_recording,
                        offset=start_offset,
                        duration=dur,
                        seizure=seizure,
                    )

                # plot H matrices
                if plot_h:
                    plotting_utils.plot_h_matrix_period(
                        folder,
                        h_matrices,
                        start_time_recording=start_time_recording,
                        offset=start_offset,
                        duration=dur,
                        seizure=seizure,
                        rank_labels_idx=rank_labels_idx,
                    )

    # Plot only a predefined period
    else:
        # PLot preprocessed data
        if plot_line_length:
            plotting_utils.plot_line_length_data(
                folder,
                line_length_eeg,
                channel_names,
                lead_prefixes=leads,
                start_time_recording=start_time_recording,
                display_all=display_all,
                offset=offset,
                duration=duration,
            )

        # plot H matrices
        if plot_h:
            plotting_utils.plot_h_matrix_period(
                folder,
                h_matrices,
                start_time_recording=start_time_recording,
                display_all=display_all,
                offset=offset,
                duration=duration,
                spike_annotations=spike_times,
                # rank_labels_idx=rank_labels_idx,
            )
