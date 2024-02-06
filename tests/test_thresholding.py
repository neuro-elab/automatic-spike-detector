import argparse
import os
import time

import numpy as np
from loguru import logger
from numpy import genfromtxt

from spidet.spike_detection.thresholding import ThresholdGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", help="full path to directory of experiment", required=True
    )

    experiment_dir: str = parser.parse_args().dir

    # number of cols to include
    length = 1440125
    usecols = list(range(np.rint(length / 8).astype(int)))

    # Read in preprocessed data
    start = time.time()
    line_length_data = genfromtxt(experiment_dir + "/line_length.csv", delimiter=",")
    end = time.time()

    print(line_length_data.shape)

    logger.debug(f"Loaded preprocessed data in {end - start} seconds")

    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        experiment_dir + "/" + k_dir
        for k_dir in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, k_dir)) and "k=" in k_dir
    ]

    for rank_dir in rank_dirs:
        h_sorted = genfromtxt(rank_dir + "/H_best.csv", delimiter=",")
        w_sorted = genfromtxt(rank_dir + "/W_best.csv", delimiter=",")

        threshold_generator = ThresholdGenerator(h_sorted, line_length_data, sfreq=50)
        threshold = threshold_generator.generate_threshold()
        spike_annotations = threshold_generator.find_events(threshold)

        logger.debug(
            f"Got the following spike annotations for rank "
            f"{rank_dir[rank_dir.rfind('=') + 1:]}: {spike_annotations}"
        )
