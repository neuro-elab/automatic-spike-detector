import logging
import multiprocessing
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import rel_entr
from sklearn.preprocessing import normalize

from spidet.spike_detection.clustering import BasisFunctionClusterer
from spidet.spike_detection.nmf import Nmf
from spidet.spike_detection.projecting import Projector
from spidet.spike_detection.thresholding import ThresholdGenerator


class NmfPipeline:
    def __init__(
        self,
        result_dir: str,
        nmf_runs: int = 100,
        rank_range: Tuple[int, int] = (2, 10),
    ):
        self.result_dir: str = result_dir
        self.nmf_runs: int = nmf_runs
        self.rank_range: Tuple[int, int] = rank_range

    @staticmethod
    def __compute_cdf(matrix: np.ndarray, bins: np.ndarray):
        N = matrix.shape[0]
        values = matrix[np.triu_indices(N)]
        counts, _ = np.histogram(values, bins=bins, density=True)
        cdf_vals = np.cumsum(counts) / (N * (N - 1) / 2)
        return cdf_vals + 1e-10  # add a small offset to avoid div0!

    def __compute_cdf_area(self, cdf_vals, bin_width):
        return np.sum(cdf_vals[:-1]) * bin_width

    def __compute_delta_k(self, areas, cdfs):
        delta_k = np.zeros(len(areas))
        delta_y = np.zeros(len(areas))
        delta_k[0] = areas[0]
        for i in range(1, len(areas)):
            delta_k[i] = (areas[i] - areas[i - 1]) / areas[i - 1]
            delta_y[i] = sum(rel_entr(cdfs[:, i], cdfs[:, i - 1]))
        return delta_k, delta_y

    def __calculate_statistics(self, consensus_matrices: List[np.ndarray]):
        k_min, k_max = self.rank_range
        bins = np.linspace(0, 1, 101)
        bin_width = bins[1] - bins[0]

        num_bins = len(bins) - 1
        cdfs = np.zeros((num_bins, k_max - k_min + 1))
        areas = np.zeros(k_max - k_min + 1)

        for idx, consensus in enumerate(consensus_matrices):
            cdf_vals = self.__compute_cdf(consensus, bins)
            areas[idx] = self.__compute_cdf_area(cdf_vals, bin_width)
            cdfs[:, idx] = cdf_vals

        delta_k, delta_y = self.__compute_delta_k(areas, cdfs)
        k_opt = np.argmax(delta_k) + k_min if delta_k.size > 0 else k_min

        return areas, delta_k, delta_y, k_opt

    def run_nmf_pipeline(
        self,
        preprocessed_data: np.ndarray,
        rank: int,
        n_runs: int,
        results_dir: str,
        execute: bool = False,
    ):
        logging.debug(f"Starting NMF pipeline for rank {rank}")

        # Create results directory for specific rank
        rank_dir = os.path.join(results_dir, f"k={rank}")
        os.makedirs(rank_dir, exist_ok=True)

        #####################
        #   NMF             #
        #####################

        # Instantiate nmf classifier
        nmf_classifier = Nmf(rank_dir, rank)

        # Run NMF consensus clustering for specified rank and number of runs (default = 100)
        metrics, consensus, h_best, w_best = nmf_classifier.nmf_run(
            preprocessed_data, n_runs
        )

        if execute:
            #####################
            # CLUSTERING BS FCT #
            #####################

            # Initialize kmeans classifier
            kmeans = BasisFunctionClusterer(n_clusters=2, use_cosine_dist=True)

            # Cluster into noise / basis function and sort according to cluster assignment
            sorted_w, sorted_h = kmeans.cluster_and_sort(
                h_matrix=h_best, w_matrix=w_best
            )
            # TODO check if necessary: cluster_assignments = np.where(cluster_assignments == 1, "BF", "noise")

            #####################
            #   THRESHOLDING    #
            #####################

            threshold_generator = ThresholdGenerator(
                preprocessed_data, sorted_h, sfreq=50
            )

            threshold = threshold_generator.generate_threshold()
            spike_annotations = threshold_generator.find_spikes(threshold)

            #####################
            #   PROJECTING      #
            #####################

            projector = Projector(h_matrix=sorted_h, w_matrix=sorted_w)
            w_projection, data_projections = projector.find_and_project_peaks(
                preprocessed_data
            )

        return metrics, consensus

    def parallel_processing(self, preprocessed_data: np.ndarray):
        """
        Parallel NMF spike detection.

        Parameters:
        - preprocessed_data : numpy.ndarray
            Data matrix.

        Returns:
        - results_dir : str
            Directory where results are saved.
        - M : numpy.ndarray
            3D array where each slice along the first axis is a consensus matrix for a specific rank.
        """
        # List of ranks to run NMF for
        rank_list = list(range(self.rank_range[0], self.rank_range[1] + 1))

        logger.debug(
            f"Running NMF on {len(rank_list)} cores for ranks {rank_list} and {self.nmf_runs} runs each"
        )

        # Normalize for NMF (preprocessed data needs to be non-negative)
        data_matrix = normalize(preprocessed_data)

        # Using all available cores for process pool, currently only one process per rank is used
        n_cores = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self.run_nmf_pipeline,
                [
                    (data_matrix, rank, self.nmf_runs, self.result_dir)
                    for rank in range(self.rank_range[0], self.rank_range[1] + 1)
                ],
            )

        # Extract and store consensus matrices and metrics from results
        consensus_matrices = [consensus for _, consensus in results]
        metrics = [metrics for metrics, _ in results]

        # Calculate final statistics
        C, delta_k, delta_y, k_opt = self.__calculate_statistics(consensus_matrices)

        # Generate metrics data frame
        metrics_df = pd.DataFrame(metrics)
        metrics_df["C"] = C
        metrics_df["delta_k (AUC)"] = delta_k
        metrics_df["delta_y (KL-div)"] = delta_y

        logger.debug(
            f"Calculated statistics:\n "
            f"C = {C}\n "
            f"delta_k (AUC) = {delta_k}\n "
            f"delta_y (KL-div) = {delta_y}\n "
            f"optimal k = {k_opt}"
        )

        # Saving metrics as CSV
        metrics_path = os.path.join(self.result_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        return (
            self.result_dir  # return the directory where results are saved and the M matrix
        )
