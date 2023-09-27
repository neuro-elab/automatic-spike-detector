import argparse
import os
import time

from loguru import logger
from numpy import genfromtxt

from spidet.spike_detection.thresholding import ThresholdGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", help="full path to directory of experiment", required=True
    )

    experiment_dir: str = parser.parse_args().dir

    # Read in preprocessed data
    start = time.time()
    preprocessed_data = genfromtxt(experiment_dir + "/preprocessed.csv", delimiter=",")
    end = time.time()

    logger.debug(f"Loaded preprocessed data in {end - start} seconds")

    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        experiment_dir + "/" + k_dir
        for k_dir in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, k_dir)) and "k=" in k_dir
    ]

    for rank_dir in rank_dirs:
        h_sorted = genfromtxt(rank_dir + "/H_best_sorted.csv", delimiter=",")
        w_sorted = genfromtxt(rank_dir + "/W_best_sorted.csv", delimiter=",")

        threshold_generator = ThresholdGenerator(preprocessed_data, h_sorted, sfreq=50)
        threshold = threshold_generator.generate_threshold()
        spike_annotations = threshold_generator.find_spikes(threshold)

        logger.debug(
            f"Got the following spike annotations for rank "
            f"{rank_dir[rank_dir.rfind('=') + 1:]}: {spike_annotations}"
        )
