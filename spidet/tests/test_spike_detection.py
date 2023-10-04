import argparse
import multiprocessing
import time

import numpy as np
from loguru import logger

from spidet.spike_detection.spike_detection_pipeline import SpikeDetectionPipeline
from spidet.tests.variables import (
    DATASET_PATHS_SZ2,
    LEAD_PREFIXES_SZ2,
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

    file: str = parser.parse_args().file

    # Configure logger
    logging_utils.add_logger_with_process_name()

    multiprocessing.freeze_support()

    # Specify range of ranks
    k_min = 3
    k_max = 7

    # How many runs of NMF to perform per rank
    runs_per_rank = 100

    # Define bad times
    bad_times = np.array(
        [
            [3080850, 3080854],
            [3545668, 3545670],
            [3551326, 3551327],
            [3563311, 3563313],
            [6233543, 6233545],
            [29343015, 29343019],
            [29368598, 29368606],
            [31829228, 31829233],
            [31961742, 31961746],
            [32573698, 32573872],
            [41377108, 41377115],
            [45858403, 45858417],
            [50101975, 50101980],
            [50501533, 50501541],
            [50502059, 50502061],
            [50521703, 50521712],
            [50723130, 50723140],
            [56445678, 56445684],
            [56564357, 56564378],
        ]
    )

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
