import argparse
import os

from loguru import logger
from numpy import genfromtxt

from spidet.spike_detection.clustering import BasisFunctionClusterer

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
        h_matrix = genfromtxt(rank_dir + "/H_best.csv", delimiter=",")
        w_matrix = genfromtxt(rank_dir + "/W_best.csv", delimiter=",")

        sorted_w, sorted_h, sorted_assignments = kmeans.cluster_and_sort(
            h_matrix, w_matrix
        )

        h_best_sorted_path = os.path.join(rank_dir, "H_best_sorted.csv")
        # np.savetxt(h_best_sorted_path, sorted_h, delimiter=",")

        w_best_sorted_path = os.path.join(rank_dir, "W_best_sorted.csv")
        # np.savetxt(w_best_sorted_path, sorted_w, delimiter=",")

        logger.debug(
            f"Clustering W for rank {rank_dir[rank_dir.rfind('=') + 1:]} "
            f"produced the following sorted w: "
            f"{sorted_w}"
        )
