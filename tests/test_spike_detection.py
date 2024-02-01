import argparse
import multiprocessing
import re
import time
import csv
from typing import List

import numpy as np
from loguru import logger
import pandas as pd

from spidet.load.data_loading import DataLoader
from spidet.spike_detection.spike_detection_pipeline import SpikeDetectionPipeline
from spidet.utils.variables import (
    DATASET_PATHS_006,
    LEAD_PREFIXES_006,
    DATASET_PATHS_007,
    LEAD_PREFIXES_007,
    DATASET_PATHS_008,
    DATASET_PATHS_BIP_008,
    LEAD_PREFIXES_008,
    DATASET_PATHS_SZ2,
    LEAD_PREFIXES_SZ2,
    DATASET_PATHS_EL003,
    LEAD_PREFIXES_EL003,
    DATASET_PATHS_EL003_BIP,
    CHANNELS_EL010_FIF,
    PREFIXES_EL010_FIF,
    CHANNEL_NAMES_005,
    LEAD_PREFIXES_005,
    DATASET_PATHS_BIP_005,
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


def read_csv(file_path: str) -> List[str]:
    with open(file_path, newline="") as csvfile:
        rows = csv.reader(csvfile, delimiter=",", quotechar='"')
        items = []
        for row in rows:
            items.extend(row)

    return items


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
    parser.add_argument("--v", help="nmf version to use", required=True)
    parser.add_argument(
        "--ch_names", help="file containing channel names", required=False
    )
    parser.add_argument(
        "--ch_prefixes", help="file containing channel prefixes", required=False
    )

    file: str = parser.parse_args().file
    bipolar_reference: bool = parser.parse_args().br
    bad_times_file: str = parser.parse_args().bt
    bad_channels_file: str = parser.parse_args().bc
    labels_file: str = parser.parse_args().labels
    nmf_version: str = parser.parse_args().v
    ch_names_file: str = parser.parse_args().ch_names
    ch_prefixes_file: str = parser.parse_args().ch_prefixes

    # Configure logger
    # logging_utils.add_logger_with_process_name()

    # Channels and leads
    channel_paths = (
        read_csv(ch_names_file) if ch_names_file is not None else DATASET_PATHS_BIP_008
    )
    leads = (
        read_csv(ch_prefixes_file)
        if ch_prefixes_file is not None
        else LEAD_PREFIXES_008
    )

    multiprocessing.freeze_support()

    # Specify range of ranks
    k_min = 2
    k_max = 5

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

    # Extract channel names from channel paths
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
            bipolar_channels_included = [
                bipolar_channels[channel] for channel in include_channels
            ]

            # Map to regular channel names
            regular_channel_names = sum(
                list(
                    map(
                        lambda bipolar_channel_name: bipolar_channel_name.split("-"),
                        bipolar_channels_included,
                    )
                ),
                [],
            )

            channels_included = []
            for regular_name in regular_channel_names:
                paths = filter(lambda ch_path: regular_name in ch_path, channel_paths)
                channels_included.extend(paths)

            # Remove duplicates and sort channels
            channels_included = list(set(channels_included))
            sorted_channels = []
            for prefix in leads:
                prefix_channels = list(
                    filter(
                        lambda name: name[name.rfind("/") + 1 :].startswith(prefix),
                        channels_included,
                    )
                )
                prefix_channels_sorted = sorted(
                    prefix_channels, key=lambda s: int(re.search(r"\d+", s).group())
                )
                sorted_channels.extend(prefix_channels_sorted)
            channels_included = sorted_channels
        else:
            channels_included = [channel_paths[channel] for channel in include_channels]
    else:
        channels_included = channel_paths

    for sparseness in [0.0]:
        sparseness = sparseness * 100
        # Initialize spike detection pipeline
        spike_detection_pipeline = SpikeDetectionPipeline(
            file_path=file,
            save_nmf_matrices=True,
            sparseness=sparseness,  # if nmf_version == "nmf" else True,
            bad_times=bad_times,
            nmf_runs=runs_per_rank,
            rank_range=(k_min, k_max),
        )

        # Run spike detection pipeline
        start = time.time()
        logger.debug(
            f"\n\n##########################################################################"
        )
        logger.debug(
            f"File: {file}; Start nmf for sparseness {sparseness}; Start time is {start}"
        )
        basis_functions, spike_activation_functions = spike_detection_pipeline.run(
            channel_paths=channels_included,
            exclude=exclude,
            bipolar_reference=bipolar_reference,
            leads=leads,
        )
        end = time.time()
        logger.debug(f"Finished nmf in {end - start} seconds")

        logger.debug(
            f"Results:\n Basis Functions: {basis_functions}\n Spike Activation Functions: {spike_activation_functions}"
        )
