import multiprocessing
import argparse
import time

from loguru import logger
from sklearn.preprocessing import normalize
from spike_detection.nmf import parallel_nmf_consensus_clustering
from preprocessing.pipeline import parallel_preprocessing
from loader.loader import read_file
from src.spike_detection import thresholding
from src.utils import logging_utils

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be processed", required=True
    )

    file: str = parser.parse_args().file
    path_to_file = file[: file.rfind("/")]
    filename_for_saving = (
        file[file.rfind("/") + 1 :].replace(".", "_").replace(" ", "_")
    )

    # Configure logger
    logging_utils.add_logger_with_process_name()

    #####################
    #   DATA LOADING    #
    #####################

    start = time.time()
    data = read_file(file)
    end = time.time()
    logger.debug(f"Finished loading data in {end - start} seconds")

    #####################
    #   PREPROCESSING   #
    #####################

    # Preprocessing steps, ran on several partitions of the data concurrently
    # if multiprocessing is available
    start = time.time()
    preprocessed_data = parallel_preprocessing(data)
    end = time.time()
    logger.debug(f"Finished preprocessing in {end - start} seconds")

    multiprocessing.freeze_support()

    #####################
    #   NMF             #
    #####################

    # Normalize for NMF (preprocessed data needs to be non-negative)
    data_matrix = normalize(preprocessed_data)

    # Specify range of ranks
    k_min = 2
    k_max = 10

    # How many runs of NMF to perform per rank
    runs_per_rank = 100

    # Run the NMF consensus clustering
    start = time.time()
    experiment_dir = parallel_nmf_consensus_clustering(
        data_matrix,
        (k_min, k_max),
        runs_per_rank,
        filename_for_saving,
        target_clusters=None,
    )
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    # Print a confirmation that the results have been saved in the appropriate directory
    logger.debug(f"Results saved in directory: {experiment_dir}")

    #####################
    #   THRESHOLDING    #
    #####################

    spike_annotations = thresholding.parallel_thresholding(experiment_dir)

    logger.debug("Spike annotations saved in respecting rank folders")
