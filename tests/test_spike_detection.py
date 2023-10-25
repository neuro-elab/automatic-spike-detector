import argparse
import multiprocessing
import time
from typing import List

import numpy as np
from loguru import logger
import pandas as pd

from spidet.load.data_loading import DataLoader
from spidet.spike_detection.spike_detection_pipeline import SpikeDetectionPipeline
from spidet.utils.variables import (
    DATASET_PATHS_008,
    LEAD_PREFIXES_008,
)
from spidet.utils import logging_utils


def get_bipolar_channel_names(leads: List[str], channel_names: List[str]) -> List[str]:
    anodes, cathodes = DataLoader().get_anodes_and_cathodes(leads, channel_names)

    bipolar_ch_names = []
    for prefix in leads:
        lead_anodes = list(filter(lambda name: name.startswith(prefix), anodes))
        lead_cathodes = list(filter(lambda name: name.startswith(prefix), cathodes))
        for anode, cathode in zip(lead_anodes, lead_cathodes):
            bipolar_ch_names.append(f"{anode}-{cathode}")

    return bipolar_ch_names


if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be processed", required=True
    )
    parser.add_argument(
        "-br",
        action="store_true",
        help="Flag to indicate bipolar reference",
        required=False,
    )
    parser.add_argument("--bt", help="path to bad times file", required=False)
    parser.add_argument("--bc", help="path to bad channels file", required=False)
    parser.add_argument("--labels", help="path labels file", required=False)

    file: str = parser.parse_args().file
    bipolar_reference: bool = parser.parse_args().br
    bad_times_file: str = parser.parse_args().bt
    bad_channels_file: str = parser.parse_args().bc
    labels_file: str = parser.parse_args().labels

    # Configure logger
    logging_utils.add_logger_with_process_name()

    # Channels and leads
    channel_paths = DATASET_PATHS_008
    leads = LEAD_PREFIXES_008

    multiprocessing.freeze_support()

    # Specify range of ranks
    k_min = 2
    k_max = 10

    # How many runs of NMF to perform per rank
    runs_per_rank = 100

    # Define bad times
    if bad_times_file is not None:
        bad_times = np.genfromtxt(bad_times_file, delimiter=",")
    else:
        bad_times = None

    # Define labels to exclude
    if labels_file is not None:
        exclude = pd.read_excel(labels_file)["EDF"].values.tolist()
    else:
        exclude = None

    # extract channel names from channel paths
    channels = DataLoader().extract_channel_names(channel_paths)

    # Define bad channels
    channels_included = None
    if bad_channels_file is not None:
        # Retrieve bad channels indices
        bad_channels = np.genfromtxt(bad_channels_file, delimiter=",")

        # Reverse to get channels to be included and retrieve its indices
        include_channels = np.nonzero((bad_channels + 1) % 2)[0]

        if bipolar_reference:
            bipolar_channels = get_bipolar_channel_names(leads, channels)
            print(bipolar_channels)
            bipolar_channels_included = [
                bipolar_channels[channel] for channel in include_channels
            ]

            # Map to regular channel names
            channels_included = sum(
                list(
                    map(
                        lambda bipolar_channel_name: bipolar_channel_name.split("-"),
                        bipolar_channels_included,
                    )
                ),
                [],
            )
        else:
            channels_included = [channels[channel] for channel in include_channels]
    else:
        channels_included = channels

    # Get unique channels
    channels_included = list(set(channels_included))

    # Initialize spike detection pipeline
    spike_detection_pipeline = SpikeDetectionPipeline(
        file_path=file,
        save_nmf_matrices=True,
        bad_times=bad_times,
        nmf_runs=runs_per_rank,
        rank_range=(k_min, k_max),
    )

    # Run spike detection pipeline
    start = time.time()
    basis_functions, spike_detection_functions = spike_detection_pipeline.run(
        channel_paths=channels_included,
        exclude=exclude,
        bipolar_reference=bipolar_reference,
        leads=leads,
    )
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    logger.debug(
        f"Results:\n Basis Functions: {basis_functions}\n Spike Detection Functions: {spike_detection_functions}"
    )