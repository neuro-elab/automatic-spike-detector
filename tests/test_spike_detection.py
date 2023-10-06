import argparse
import multiprocessing
import time

import numpy as np
from loguru import logger

from spidet.spike_detection.spike_detection_pipeline import SpikeDetectionPipeline
from tests.variables import (
    DATASET_PATHS_EL010,
    LEAD_PREFIXES_EL010,
)
from spidet.utils import logging_utils

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be processed", required=True
    )
    parser.add_argument("--bt", help="path to bad times file", required=False)

    file: str = parser.parse_args().file
    bad_times_file: str = parser.parse_args().bt

    # Configure logger
    logging_utils.add_logger_with_process_name()

    multiprocessing.freeze_support()

    # Specify range of ranks
    k_min = 3
    k_max = 7

    # How many runs of NMF to perform per rank
    runs_per_rank = 100

    # Define bad times
    if bad_times_file is not None:
        bad_times = np.genfromtxt(bad_times_file, delimiter=",")
    else:
        bad_times = None

    # Initialize spike detection pipeline
    spike_detection_pipeline = SpikeDetectionPipeline(
        file_path=file,
        bad_times=bad_times,
        nmf_runs=runs_per_rank,
        rank_range=(k_min, k_max),
    )

    # Run spike detection pipeline
    start = time.time()
    basis_functions, spike_detection_functions = spike_detection_pipeline.run(
        channel_paths=DATASET_PATHS_EL010,
        bipolar_reference=True,
        leads=LEAD_PREFIXES_EL010,
    )
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    logger.debug(
        f"Results:\n Basis Functions: {basis_functions}\n Spike Detection Functions: {spike_detection_functions}"
    )
