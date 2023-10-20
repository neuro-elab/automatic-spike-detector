import argparse
import multiprocessing
import time
from typing import List

import numpy as np
from loguru import logger
import pandas as pd

from spidet.load.data_loading import DataLoader
from spidet.spike_detection.spike_detection_pipeline import SpikeDetectionPipeline
from tests.variables import (
    DATASET_PATHS_EL010,
    LEAD_PREFIXES_EL010,
    DATASET_PATHS_008,
    LEAD_PREFIXES_008,
    DATASET_PATHS_007,
    LEAD_PREFIXES_007,
    DATASET_PATHS_006,
    LEAD_PREFIXES_006,
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
    parser.add_argument("--bt", help="path to bad times file", required=False)
    parser.add_argument("--bc", help="path to bad channels file", required=False)
    parser.add_argument("--labels", help="path labels file", required=False)

    file: str = parser.parse_args().file
    bad_times_file: str = parser.parse_args().bt
    bad_channels_file: str = parser.parse_args().bc
    labels_file: str = parser.parse_args().labels

    # Configure logger
    logging_utils.add_logger_with_process_name()

    # Channels and leads
    channels = DATASET_PATHS_008
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

    # Define bad channels
    if bad_channels_file is not None:
        bad_channels = np.genfromtxt(bad_channels_file, delimiter=",")
        include_channels = np.nonzero((bad_channels + 1) % 2)[0]
        channels = DataLoader().extract_channel_names(channels)
        bip_ch_names = get_bipolar_channel_names(leads, channels)
        channels = [bip_ch_names[channel] for channel in include_channels]

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
        channel_paths=channels,
        exclude=exclude,
        bipolar_reference=True,
        leads=leads,
    )
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    logger.debug(
        f"Results:\n Basis Functions: {basis_functions}\n Spike Detection Functions: {spike_detection_functions}"
    )
