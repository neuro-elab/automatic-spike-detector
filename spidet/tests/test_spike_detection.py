import argparse
import multiprocessing
import time

from loguru import logger

from spidet.spike_detection.spike_detection_pipeline import SpikeDetectionPipeline
from spidet.tests.variables import DATASET_PATHS_SZ2, LEAD_PREFIXES_SZ2
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

    # Initialize spike detection pipeline
    spike_detection_pipeline = SpikeDetectionPipeline(
        file_path=file,
        nmf_runs=runs_per_rank,
        rank_range=(k_min, k_max),
    )

    # Run spike detection pipeline
    start = time.time()
    basis_functions, spike_detection_functions = spike_detection_pipeline.run(
        channel_paths=DATASET_PATHS_SZ2,
        bipolar_reference=True,
        leads=LEAD_PREFIXES_SZ2,
    )
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    logger.debug(
        f"Results:\n Basis Functions: {basis_functions}\n Spike Detection Functions: {spike_detection_functions}"
    )
