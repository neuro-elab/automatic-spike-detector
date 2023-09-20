import argparse
import datetime
import multiprocessing
import os
import time

import numpy as np
from loguru import logger
from numpy import genfromtxt

from loader.loader import read_file
from preprocessing.pipeline import parallel_preprocessing
from src.spike_detection import thresholding
from src.spike_detection.clustering import BasisFunctionClusterer
from src.spike_detection.nmf_pipeline import NmfPipeline
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

    # Create a directory to save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(filename_for_saving + "_" + timestamp)
    os.makedirs(results_dir, exist_ok=True)

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
    #   NMF PIPELINE    #
    #####################

    # Specify range of ranks
    k_min = 2
    k_max = 10

    # How many runs of NMF to perform per rank
    runs_per_rank = 100

    # Initialize NMF pipeline
    nmf_pipeline = NmfPipeline(results_dir, runs_per_rank, (k_min, k_max))

    # Run NMF pipeline
    start = time.time()
    results_dir = nmf_pipeline.parallel_processing(preprocessed_data)
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    # Print a confirmation that the results have been saved in the appropriate directory
    logger.debug(f"Results saved in directory: {results_dir}")

    #####################
    # CLUSTERING BS FCT #
    #####################

    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        results_dir + "/" + k_dir
        for k_dir in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, k_dir)) and "k=" in k_dir
    ]

    filename_data_matrix = "H_best.csv"

    # Initialize kmeans clustering object
    kmeans = BasisFunctionClusterer(n_clusters=2, use_cosine_dist=True)

    for rank_dir in rank_dirs:
        w_matrix = genfromtxt(rank_dir + "/" + filename_data_matrix, delimiter=",")

        cluster_assignments = kmeans.cluster_and_sort(w_matrix)
        cluster_assignments = np.where(cluster_assignments == 1, "BF", "noise")

        assignments_path = os.path.join(rank_dir, "cluster_assignments.csv")
        np.savetxt(assignments_path, cluster_assignments, delimiter=",")

        logger.debug(
            f"Clustering W for rank {rank_dir[rank_dir.rfind('=') + 1:]} "
            f"produced the following assignments for the basis functions: "
            f"{cluster_assignments}"
        )

    #####################
    #   THRESHOLDING    #
    #####################

    spike_annotations = thresholding.parallel_thresholding(rank_dirs)

    logger.debug("Spike annotations saved in respecting rank folders")

    #####################
    #   PROJECTING      #
    #####################
