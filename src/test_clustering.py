import argparse
import os

import numpy as np
from loguru import logger
from numpy import genfromtxt

from src.spike_detection.clustering import BasisFunctionClusterer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", help="full path to directory of experiment", required=True
    )

    experiment_dir: str = parser.parse_args().dir

    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        experiment_dir + "/" + k_dir
        for k_dir in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, k_dir)) and "k=" in k_dir
    ]

    filename_data_matrix = "H_best.csv"

    # Initialize kmeans clustering object
    kmeans = BasisFunctionClusterer(n_clusters=2, use_cosine_dist=True)

    for rank_dir in rank_dirs:
        w_matrix = genfromtxt(rank_dir + "/" + filename_data_matrix, delimiter=",")

        cluster_assignments = kmeans.cluster(w_matrix)
        cluster_assignments = np.where(cluster_assignments == 1, "BF", "noise")

        logger.debug(
            f"Clustering W for rank {rank_dir[rank_dir.rfind('=') + 1:]} "
            f"produced the following assignments for the basis functions: "
            f"{cluster_assignments}"
        )
