import logging
import multiprocessing
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import rel_entr
from sklearn.preprocessing import normalize
from pathlib import Path

from spidet.domain.BasisFunction import BasisFunction
from spidet.domain.SpikeDetectionFunction import SpikeDetectionFunction
from spidet.domain.CoefficentsFunction import CoefficientsFunction
from spidet.spike_detection.clustering import BasisFunctionClusterer
from spidet.spike_detection.line_length import LineLength
from spidet.spike_detection.nmf import Nmf
from spidet.spike_detection.projecting import Projector
from spidet.spike_detection.thresholding import ThresholdGenerator
from spidet.utils.times_utils import compute_rescaled_timeline
from spidet.utils.plotting_utils import plot_w_and_consensus_matrix


class SpikeDetectionPipeline:
    def __init__(
        self,
        file_path: str,
        results_dir: str = None,
        save_nmf_matrices: bool = False,
        bad_times: np.ndarray = None,
        nmf_runs: int = 100,
        rank_range: Tuple[int, int] = (2, 10),
        line_length_freq: int = 50,
    ):
        self.file_path = file_path
        self.results_dir: str = self.__create_results_dir(results_dir)
        self.save_nmf_matrices = save_nmf_matrices
        self.bad_times = bad_times
        self.nmf_runs: int = nmf_runs
        self.rank_range: Tuple[int, int] = rank_range
        self.line_length_freq = line_length_freq

    def __create_results_dir(self, results_dir: str):
        if results_dir is None:
            # Create default directory if none is given
            file_path = self.file_path
            filename_for_saving = (
                file_path[file_path.rfind("/") + 1 :]
                .replace(".", "_")
                .replace(" ", "_")
            )

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_dir = os.path.join(
                Path.home(), filename_for_saving + "_" + timestamp
            )
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    @staticmethod
    def __compute_cdf(matrix: np.ndarray, bins: np.ndarray):
        N = matrix.shape[0]
        values = matrix[np.triu_indices(N)]
        counts, _ = np.histogram(values, bins=bins, density=True)
        cdf_vals = np.cumsum(counts) / (N * (N - 1) / 2)
        return cdf_vals + 1e-10  # add a small offset to avoid div0!

    @staticmethod
    def __compute_cdf_area(cdf_vals, bin_width):
        return np.sum(cdf_vals[:-1]) * bin_width

    @staticmethod
    def __compute_delta_k(areas, cdfs):
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

    @staticmethod
    def perform_nmf_steps_for_rank(
        preprocessed_data: np.ndarray,
        rank: int,
        n_runs: int,
        execute: bool = False,
    ) -> Tuple[
        Dict,
        np.ndarray[Any, np.dtype[float]],
        np.ndarray[Any, np.dtype[float]],
        np.ndarray[Any, np.dtype[float]],
        Dict[int, np.ndarray[Any, np.dtype[int]]],
        float,
    ]:
        logging.debug(f"Starting Spike detection pipeline for rank {rank}")

        #####################
        #   NMF             #
        #####################

        # Instantiate nmf classifier
        nmf_classifier = Nmf(rank=rank)

        # Run NMF consensus clustering for specified rank and number of runs (default = 100)
        metrics, consensus, h_best, w_best = nmf_classifier.nmf_run(
            preprocessed_data, n_runs
        )

        #####################
        # CLUSTERING BS FCT #
        #####################

        # Initialize kmeans classifier
        kmeans = BasisFunctionClusterer(n_clusters=2, use_cosine_dist=True)

        # Cluster into noise / basis function and sort according to cluster assignment
        sorted_w, sorted_h = kmeans.cluster_and_sort(h_matrix=h_best, w_matrix=w_best)
        # TODO check if necessary: cluster_assignments = np.where(cluster_assignments == 1, "BF", "noise")

        #####################
        #   THRESHOLDING    #
        #####################

        threshold_generator = ThresholdGenerator(preprocessed_data, sorted_h, sfreq=50)

        threshold = threshold_generator.generate_threshold()
        spike_annotations = threshold_generator.find_spikes(threshold)

        if execute:
            #####################
            #   PROJECTING      #
            #####################

            projector = Projector(h_matrix=sorted_h, w_matrix=sorted_w)
            w_projection, data_projections = projector.find_and_project_peaks(
                preprocessed_data
            )

        return metrics, consensus, h_best, w_best, spike_annotations, threshold

    def parallel_processing(
        self,
        preprocessed_data: np.ndarray[Any, np.dtype[float]],
        channel_names: List[str],
    ) -> Tuple[
        np.ndarray[Any, np.dtype[float]],
        np.ndarray[Any, np.dtype[float]],
        Dict[int, np.ndarray[Any, np.dtype[int]]],
        float,
    ]:
        # List of ranks to run NMF for
        rank_list = list(range(self.rank_range[0], self.rank_range[1] + 1))
        nr_ranks = len(rank_list)

        # Normalize for NMF (preprocessed data needs to be non-negative)
        data_matrix = normalize(preprocessed_data)

        # Using all cores except 2 if necessary
        n_cores = multiprocessing.cpu_count() - 2

        logger.debug(
            f"Running NMF on {n_cores if nr_ranks > n_cores else nr_ranks} cores "
            f"for ranks {rank_list} and {self.nmf_runs} runs each"
        )

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self.perform_nmf_steps_for_rank,
                [
                    (data_matrix, rank, self.nmf_runs)
                    for rank in range(self.rank_range[0], self.rank_range[1] + 1)
                ],
            )

        # Extract return objects from results
        consensus_matrices = [consensus for _, consensus, _, _, _, _ in results]
        h_matrices = [h_best for _, _, h_best, _, _, _ in results]
        w_matrices = [w_best for _, _, _, w_best, _, _ in results]
        metrics = [metrics for metrics, _, _, _, _, _ in results]
        spike_annotations = [
            spike_annotations for _, _, _, _, spike_annotations, _ in results
        ]
        thresholds = [threshold for _, _, _, _, _, threshold in results]

        # Calculate final statistics
        C, delta_k, delta_y, k_opt = self.__calculate_statistics(consensus_matrices)

        # Get the H and W matrices with spike annotations for the optimal rank
        idx_opt = k_opt - self.rank_range[0]
        h_opt = h_matrices[idx_opt]
        w_opt = w_matrices[idx_opt]
        spikes_opt = spike_annotations[idx_opt]
        threshold_opt = thresholds[idx_opt]

        # Generate metrics data frame
        metrics_df = pd.DataFrame(metrics)
        metrics_df["AUC"] = C
        metrics_df["delta_k (CDF)"] = delta_k
        metrics_df["delta_y (KL-div)"] = delta_y

        logger.debug(f"Optimal rank: k = {k_opt}")

        # Plot and save W and consensus matrices
        plot_w_and_consensus_matrix(
            w_matrices=w_matrices,
            consensus_matrices=consensus_matrices,
            experiment_dir=self.results_dir,
            channel_names=channel_names,
        )

        # Saving metrics as CSV
        logger.debug("Saving metrics")
        metrics_path = os.path.join(self.results_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Saving H and W matrices and spike annotations
        if self.save_nmf_matrices:
            logger.debug(
                f"Saving Consensus, W, H matrices and corresponding spike annotations for ranks {rank_list}"
            )
            for idx in range(nr_ranks):
                # Saving Consensus, W and H matrices
                h_matrix = h_matrices[idx]
                w_matrix = w_matrices[idx]
                consensus_matrix = consensus_matrices[idx]

                saving_path = os.path.join(self.results_dir, f"k={rank_list[idx]}")
                os.makedirs(saving_path, exist_ok=True)

                np.savetxt(f"{saving_path}/H_best.csv", h_matrix, delimiter=",")
                np.savetxt(f"{saving_path}/W_best.csv", w_matrix, delimiter=",")
                np.savetxt(
                    f"{saving_path}/consensus_matrix.csv",
                    consensus_matrix,
                    delimiter=",",
                )

                # Saving spike annotations
                spikes = spike_annotations[idx]
                headers = []
                spike_times = []
                max_length = 0
                for h_idx in spikes.keys():
                    spike_times_on = spikes.get(h_idx).get("spikes_on")
                    spike_times_off = spikes.get(h_idx).get("spikes_off")

                    if len(spike_times_on) > max_length:
                        max_length = len(spike_times_on)

                    spike_times.append(spike_times_on)
                    spike_times.append(spike_times_off)
                    headers.extend(
                        [f"h{h_idx + 1}_spikes_on", f"h{h_idx + 1}_spikes_off"]
                    )

                df_spike_times = pd.DataFrame(spike_times)
                df_spike_times = df_spike_times.transpose()
                df_spike_times.columns = headers

                df_spike_times.to_csv(f"{saving_path}/spike_annotations.csv")

        return h_opt, w_opt, spikes_opt, threshold_opt

    def run(
        self,
        channel_paths: List[str],
        bipolar_reference: bool = False,
        exclude: List[str] = None,
        leads: List[str] = None,
    ) -> Tuple[List[BasisFunction], List[SpikeDetectionFunction]]:
        # Instantiate a LineLength instance
        line_length = LineLength(
            file_path=self.file_path,
            dataset_paths=channel_paths,
            exclude=exclude,
            bipolar_reference=bipolar_reference,
            leads=leads,
            bad_times=self.bad_times,
        )

        # Perform line length steps to compute line length
        (
            start_timestamp,
            channel_names,
            line_length_matrix,
        ) = line_length.apply_parallel_line_length_pipeline()

        # Run parallelized NMF
        h_opt, w_opt, spikes_opt, threshold_opt = self.parallel_processing(
            preprocessed_data=line_length_matrix, channel_names=channel_names
        )

        # Create unique id prefix
        filename = self.file_path[self.file_path.rfind("/") + 1 :]
        unique_id_prefix = filename[: filename.rfind(".")]

        # Compute times for H x-axis
        times = compute_rescaled_timeline(
            start_timestamp=start_timestamp,
            length=h_opt.shape[1],
            sfreq=self.line_length_freq,
        )

        # Create return objects
        basis_functions: List[BasisFunction] = []
        coefficient_functions: List[CoefficientsFunction] = []

        for idx, (bf, sdf, spikes_idx) in enumerate(zip(w_opt.T, h_opt, spikes_opt)):
            # Create BasisFunction
            label_bf = f"W{idx + 1}"
            unique_id_bf = f"{unique_id_prefix}_{label_bf}"
            basis_fct = BasisFunction(
                label=label_bf,
                unique_id=unique_id_bf,
                channel_names=channel_names,
                data_array=bf,
            )

            # Create SpikeDetectionFunction
            label_sdf = f"H{idx + 1}"
            unique_id_sdf = f"{unique_id_prefix}_{label_sdf}"
            coefficient_fct = CoefficientsFunction(
                label=label_sdf,
                unique_id=unique_id_sdf,
                times=times,
                data_array=sdf,
                spikes_on_indices=spikes_opt.get(spikes_idx)["spikes_on"],
                spikes_off_indices=spikes_opt.get(spikes_idx)["spikes_off"],
                spike_threshold=threshold_opt,
            )

            basis_functions.append(basis_fct)
            coefficient_functions.append(coefficient_fct)

        return basis_functions, coefficient_functions
