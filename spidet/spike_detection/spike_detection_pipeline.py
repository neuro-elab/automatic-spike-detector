import logging
import multiprocessing
import os
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import rel_entr
from sklearn.preprocessing import normalize
from pathlib import Path

from spidet.utils import logging_utils

from spidet.domain.BasisFunction import BasisFunction
from spidet.domain.ActivationFunction import ActivationFunction
from spidet.spike_detection.clustering import BasisFunctionClusterer
from spidet.spike_detection.line_length import LineLength
from spidet.spike_detection.nmf import Nmf
from spidet.spike_detection.thresholding import ThresholdGenerator
from spidet.utils.times_utils import compute_rescaled_timeline
from spidet.utils.plotting_utils import plot_w_and_consensus_matrix


class SpikeDetectionPipeline:
    """
    This class builds the heart of the automatic-spike-detection library. It provides an end-to-end
    pipeline that takes in a path to a file containing an iEEG recording and returns periods of
    abnormal activity. The pipeline is a multistep process that includes

        1.  reading the data from the provided file (supported file formats are .h5, .edf, .fif) and
            transforming the data into a list of :py:mod:`~spidet.domain.Trace` objects,
        2.  performing the necessary preprocessing steps by means of the :py:mod:`~spidet.preprocess.preprocessing` module,
        3.  applying the line-length transformation using the :py:mod:`~spidet.spike_detection.line_length` module,
        4.  performing Nonnegative Matrix Factorization to extract the most discriminating metappatterns,
            done by the :py:mod:`~spidet.spike_detection.nmf` module and
        5.  computing periods of abnormal activity by means of the :py:mod:`~spidet.spike_detection.thresholding` module.

    Parameters
    ----------

    file_path: str
        Path to the file containing the iEEG data.

    results_dir: str
        Path to the directory where the folder containing the results of the spike detection run should be saved.
        If None, the results folder will be saved in the user's home directory. By default, the results contain
        the metrics representing the computation of the optimal rank and two plots for both the basis functions
        and the consensus matrices of the different ranks.

    save_nmf_matrices: bool
        If True, in addition to the results saved by default, the W matrix containing the basis functions and
        the H matrix containing the activation functions for each rank, the line-length transformed data and
        the standard deviation of the line length are saved.

    sparseness: float
        A floating point number :math:`\in [0, 1]`.
        If this parameter is non-zero, nonnegative matrix factorization is run with sparseness constraints.

    bad_times: numpy.ndarray[numpy.dtype[float]]
        An optional N x 2 numpy array containing periods that must be excluded before applying
        the line-length transformation. Each of th N rows in the array represents a period to be excluded,
        defined by the start and end indices of the period in the original iEEG data.
        The defined periods will be set to zero with the transitions being smoothed by applying a hanning window
        to prevent spurious patterns.

    nmf_runs: int
        The number of nonnegative matrix factorization runs performed for each rank, default is 100.

    rank_range: Tuple[int, int]
        A tuple defining the range of ranks for which to perform the nonnegative matrix factorization,
        default is (2, 5).

    line_length_freq: int
        The sampling frequency of the line-length transformed data, default is 50 hz.
    """

    def __init__(
        self,
        file_path: str,
        results_dir: str = None,
        save_nmf_matrices: bool = False,
        sparseness: float = 0.0,
        bad_times: np.ndarray[np.dtype[float]] = None,
        nmf_runs: int = 100,
        rank_range: Tuple[int, int] = (2, 5),
        line_length_freq: int = 50,
    ):
        self.sparseness = sparseness
        self.file_path = file_path
        self.results_dir: str = self.__create_results_dir(results_dir)
        self.save_nmf_matrices: bool = save_nmf_matrices
        self.bad_times = bad_times
        self.nmf_runs: int = nmf_runs
        self.rank_range: Tuple[int, int] = rank_range
        self.line_length_freq = line_length_freq

        # Configure logger
        logging_utils.add_logger_with_process_name(self.results_dir)

    def __create_results_dir(self, results_dir: str):
        # Create folder to save results
        file_path = self.file_path
        filename_for_saving = (
            file_path[file_path.rfind("/") + 1 :].replace(".", "_").replace(" ", "_")
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nmf_version = "NMFSC" if self.sparseness != 0.0 else "NMF"
        folder_name = "_".join([nmf_version, filename_for_saving, timestamp])

        if results_dir is None:
            results_dir = os.path.join(Path.home(), folder_name)
        else:
            results_dir = os.path.join(results_dir, folder_name)

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

    def perform_nmf_steps_for_rank(
        self,
        preprocessed_data: np.ndarray,
        rank: int,
        n_runs: int,
    ) -> Tuple[
        Dict,
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
        Dict[int, np.ndarray[np.dtype[int]]],
        float,
        Dict[int, int],
    ]:
        logging.debug(f"Starting Spike detection pipeline for rank {rank}")

        #####################
        #   NMF             #
        #####################

        # Instantiate nmf classifier
        nmf_classifier = Nmf(rank=rank, sparseness=self.sparseness)

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
        sorted_w, sorted_h, cluster_assignments = kmeans.cluster_and_sort(
            h_matrix=h_best, w_matrix=w_best
        )
        # TODO check if necessary: cluster_assignments = np.where(cluster_assignments == 1, "BF", "noise")

        #####################
        #   THRESHOLDING    #
        #####################

        threshold_generator = ThresholdGenerator(sorted_h, preprocessed_data, sfreq=50)

        threshold_generator.generate_individual_thresholds()
        spike_annotations = threshold_generator.find_events()

        return (
            metrics,
            consensus,
            sorted_h,
            sorted_w,
            spike_annotations,
            threshold_generator.thresholds,
            cluster_assignments,
        )

    def parallel_processing(
        self,
        preprocessed_data: np.ndarray[np.dtype[float]],
        channel_names: List[str],
    ) -> Tuple[
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
        Dict[int, np.ndarray[np.dtype[int]]],
        Dict[int, float],
        Dict[int, int],
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
        consensus_matrices = [consensus for _, consensus, _, _, _, _, _ in results]
        h_matrices = [h_best for _, _, h_best, _, _, _, _ in results]
        w_matrices = [w_best for _, _, _, w_best, _, _, _ in results]
        metrics = [metrics for metrics, _, _, _, _, _, _ in results]
        event_annotations = [
            spike_annotations for _, _, _, _, spike_annotations, _, _ in results
        ]
        thresholds = [threshold for _, _, _, _, _, threshold, _ in results]
        cluster_assignments = [assignments for _, _, _, _, _, _, assignments in results]

        # Calculate final statistics
        C, delta_k, delta_y, k_opt = self.__calculate_statistics(consensus_matrices)

        # Get objects for the optimal rank
        idx_opt = k_opt - self.rank_range[0]
        h_opt = h_matrices[idx_opt]
        w_opt = w_matrices[idx_opt]
        events_opt = event_annotations[idx_opt]
        thresholds_opt = thresholds[idx_opt]
        assignments_opt = cluster_assignments[idx_opt]

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

        # Saving H and W matrices, event annotations and line length matrix
        if self.save_nmf_matrices:
            logger.debug(
                f"Saving LineLength and Consensus, W, H matrices and corresponding event annotations for ranks {rank_list}"
            )

            # Saving line length and std line length
            np.savetxt(
                f"{self.results_dir}/line_length.csv", data_matrix, delimiter=","
            )

            np.savetxt(
                f"{self.results_dir}/std_line_length.csv",
                np.std(data_matrix, axis=0),
                delimiter=",",
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

                # Saving event annotations
                spikes = event_annotations[idx]
                headers = []
                event_times = []
                max_length = 0
                for h_idx in spikes.keys():
                    event_times_on = spikes.get(h_idx).get("events_on")
                    event_times_off = spikes.get(h_idx).get("events_off")

                    if len(event_times_on) > max_length:
                        max_length = len(event_times_on)

                    event_times.append(event_times_on)
                    event_times.append(event_times_off)
                    headers.extend(
                        [f"h{h_idx + 1}_events_on", f"h{h_idx + 1}_events_off"]
                    )

                df_spike_times = pd.DataFrame(event_times)
                df_spike_times = df_spike_times.transpose()
                df_spike_times.columns = headers

                df_spike_times.to_csv(f"{saving_path}/event_annotations.csv")

        return h_opt, w_opt, events_opt, thresholds_opt, assignments_opt

    def run(
        self,
        channel_paths: List[str] = None,
        bipolar_reference: bool = False,
        exclude: List[str] = None,
        leads: List[str] = None,
        notch_freq: int = 50,
        resampling_freq: int = 500,
        bandpass_cutoff_low: int = 0.1,
        bandpass_cutoff_high: int = 200,
        line_length_freq: int = 50,
        line_length_window: int = 40,
    ) -> Tuple[List[BasisFunction], List[ActivationFunction]]:
        """
        This method triggers a complete run of the spike detection pipline with the arguments passed
        to the :py:class:`SpikeDetectionPipeline` at initialization.

        Parameters
        ----------
        channel_paths: List[str]
            A list of paths to the traces to be included within an h5 file. This is only necessary in the case
            of h5 files.

        bipolar_reference: bool
            If True, the bipolar references of the included channels will be computed. If channels already are
            in bipolar form this needs to be False.

        exclude: List[str]
            A list of channel names that need to be excluded. This only applies in the case of .edf and .fif files.

        leads: List[str]
            A list of the leads included. Only necessary if bipolar_reference is True, otherwise can be None.

        notch_freq: int, optional, default = 50
            The frequency of the notch filter; data will be notch-filtered at this frequency
            and at the corresponding harmonics,
            e.g. notch_freq = 50 Hz -> harmonics = [50, 100, 150, etc.]

        resampling_freq: int, optional, default = 500
            The frequency to resample the data after filtering and rescaling

        bandpass_cutoff_low: int, optional, default = 0.1
            Cut-off frequency at the lower end of the passband of the bandpass filter.

        bandpass_cutoff_high: int, optional, default = 200
            Cut-off frequency at the higher end of the passband of the bandpass filter.

        line_length_freq: int, optional, default = 50
            Sampling frequency of the line-length transformed data

        line_length_window: int, optional, default = 40
            Window length used to for the line-length operation (in milliseconds).

        Returns
        -------
        Tuple[List[BasisFunction], List[ActivationFunction]]
            Two lists containing the :py:mod:`~spidet.domain.BasisFunction`
            and :py:mod:`~spidet.domain.ActivationFunction`, where each activation function contains
            the corresponding detected events.
        """
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
        ) = line_length.apply_parallel_line_length_pipeline(
            notch_freq=notch_freq,
            resampling_freq=resampling_freq,
            bandpass_cutoff_low=bandpass_cutoff_low,
            bandpass_cutoff_high=bandpass_cutoff_high,
            line_length_freq=line_length_freq,
            line_length_window=line_length_window,
        )

        # Run parallelized NMF
        (
            h_opt,
            w_opt,
            spikes_opt,
            thresholds_opt,
            assignments_opt,
        ) = self.parallel_processing(
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
        activation_functions: List[ActivationFunction] = []

        for idx, (bf, sdf, spikes_idx, threshold_idx, assignments_idx) in enumerate(
            zip(w_opt.T, h_opt, spikes_opt, thresholds_opt, assignments_opt)
        ):
            # Create BasisFunction
            label_bf = f"W{idx + 1}"
            unique_id_bf = f"{unique_id_prefix}_{label_bf}"
            basis_fct = BasisFunction(
                label=label_bf,
                unique_id=unique_id_bf,
                channel_names=channel_names,
                data_array=bf,
            )

            # Create ActivationFunction
            label_af = f"H{idx + 1}"
            unique_id_af = f"{unique_id_prefix}_{label_af}"
            activation_fct = ActivationFunction(
                label=label_af,
                unique_id=unique_id_af,
                times=times,
                data_array=sdf,
                detected_events_on=spikes_opt.get(spikes_idx)["events_on"],
                detected_events_off=spikes_opt.get(spikes_idx)["events_off"],
                event_threshold=thresholds_opt.get(threshold_idx),
            )

            basis_functions.append(basis_fct)
            activation_functions.append(activation_fct)

        return basis_functions, activation_functions
