import argparse
import datetime
import multiprocessing
import os
import time

import numpy as np
from loguru import logger
from numpy import genfromtxt

from preprocess.preprocessing import Preprocessor
from spidet.spike_detection import thresholding
from spidet.spike_detection.clustering import BasisFunctionClusterer
from spidet.spike_detection.nmf_pipeline import NmfPipeline
from spidet.utils import logging_utils


LABELS_EL010 = [
    "Hip21",
    "Hip22",
    "Hip23",
    "Hip24",
    "Hip25",
    "Hip26",
    "Hip27",
    "Hip28",
    "Hip29",
    "Hip210",
    "Temp1",
    "Temp2",
    "Temp3",
    "Temp4",
    "Temp5",
    "Temp6",
    "Temp7",
    "Temp8",
    "Temp9",
    "Temp10",
    "FrOr1",
    "FrOr2",
    "FrOr3",
    "FrOr4",
    "FrOr5",
    "FrOr6",
    "FrOr7",
    "FrOr8",
    "FrOr9",
    "FrOr10",
    "FrOr11",
    "FrOr12",
    "In An1",
    "In An2",
    "In An3",
    "In An4",
    "In An5",
    "In An6",
    "In An7",
    "In An8",
    "In An9",
    "In An10",
    "In An11",
    "In An12",
    "InPo1",
    "InPo2",
    "InPo3",
    "InPo4",
    "InPo5",
    "InPo6",
    "InPo7",
    "InPo8",
    "InPo9",
    "InPo10",
    "InPo11",
    "InPo12",
    "Hip11",
    "Hip12",
    "Hip13",
    "Hip14",
    "Hip15",
    "Hip16",
    "Hip17",
    "Hip18",
    "Hip19",
    "Hip110",
    "Hip111",
    "Hip112",
]

LEAD_PREFIXES_EL010 = ["Hip2", "Temp", "FrOr", "In An", "InPo", "Hip1"]

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
    #   PREPROCESSING   #
    #####################

    # Preprocessing steps, ran on several partitions of the data concurrently
    # if multiprocessing is available
    start = time.time()

    # Instantiate a Preprocessor instance
    preprocessor = Preprocessor(
        file_path=file,
        dataset_paths=LABELS_EL010,
        bipolar_reference=True,
        leads=LEAD_PREFIXES_EL010,
    )
    preprocessed = preprocessor.parallel_preprocessing()
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
    results_dir = nmf_pipeline.parallel_processing(preprocessed.line_length)
    end = time.time()
    logger.debug(f"Finished nmf in {end - start} seconds")

    # Print a confirmation that the results have been saved in the appropriate directory
    logger.debug(f"Results saved in directory: {results_dir}")

    #####################
    #   PROJECTING      #
    #####################
