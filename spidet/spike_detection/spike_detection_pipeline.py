import logging
import multiprocessing
import os
import re
import datetime as dt
from typing import Tuple, List, Dict

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import rel_entr
from sklearn.preprocessing import normalize

from spidet.save.nmf_data import (
    FeatureMatrixGroup,
    NMFRoot,
)
from spidet.utils import logging_utils

from spidet.domain.BasisFunction import BasisFunction
from spidet.domain.ActivationFunction import ActivationFunction
from spidet.spike_detection.line_length import LineLength
from spidet.spike_detection.nmf import Nmf
import spidet.utils.h5_utils as h5_utils


class SpikeDetectionPipeline:
    r"""
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
        6.  Storing the results in an .h5 file using the :py:mod:`~spidet.spike_detection.save.nmf_data` module.

    Parameters
    ----------

    file_path: str
        Path to the file containing the iEEG data.

    result_path: str, default: "."
        The results will be stored in .h5 format at the specified path. It should be ensured that the path is
        writable. If None, the current directory will be used.

    sparseness: float, default: 0.0
        A floating point number :math:`\in [0, 1]`.
        If this parameter is non-zero, nonnegative matrix factorization is run with sparseness constraints.

    version: str, optional, default: "l"
        If version = 'l', sparseness will be imposed on the columns of :math:`W`, if version = 'r',
        sparseness will be imposed on the rows of :math:`H`.

    bad_times: numpy.ndarray[numpy.dtype[float]], optional
        N x 2 numpy array, designating periods to be zeroed before applying the line-length transformation.
        Each row represents a time period with [start, end]. Values for start and end correspond to the time in seconds
        since the start of the recording. A hanning window is applied to smooth transitions around the periods.

    nmf_runs: int, default: 100
        The number of nonnegative matrix factorization runs performed for each rank.

    ranks: List[int], default: [2, 3, 4, 5]
        A tuple defining the range of ranks for which to perform the nonnegative matrix factorization.

    line_length_freq: int, default: 50
        The sampling frequency of the line-length transformed data in hz.

    H: np.ndarray | None, default: None
        If given, the algorithm will set it as the initial value for H.

    W: np.ndarray | None, default: None
        If given, the algorithm will set it as the initial value for W.

    subject_id: str | None, default: None
        The id of the subject from which the data originates.

    dataset_id: str, default: "id0"
        The id of the dataset under which the results will be stored in the h5 file.

    """

    def __init__(
        self,
        file_path: str,
        result_path: str = "nmf.h5",
        sparseness: float = 0.0,
        version: str = "l",
        bad_times: np.ndarray[np.dtype[float]] = None,
        nmf_runs: int = 100,
        ranks: List[int] = [2, 3, 4, 5],
        line_length_freq: int = 50,
        H: np.ndarray | None = None,
        W: np.ndarray | None = None,
        subject_id: str | None = None,
        dataset_id: str = "id0",
    ):
        self.sparseness = sparseness
        self.version = version
        self.file_path = file_path
        self.results_path: str = result_path
        self.bad_times = bad_times
        self.nmf_runs = nmf_runs
        self.ranks = ranks
        self.line_length_freq = line_length_freq
        # Set results data
        if subject_id is None:
            filename = os.path.basename(file_path)
            self.subject_id = re.match(r"[a-zA-Z]+\d+", filename)[0]
        else:
            self.subject_id = subject_id

        self.nmf_dataset = NMFRoot(filepath=self.results_path).dataset(dataset_id)
        self.meta = self.nmf_dataset.meta()
        self.meta.creation_date = (
            dt.datetime.now().replace(tzinfo=dt.timezone.utc).timestamp()
        )
        self.meta.subject_id = self.subject_id
        self.meta.species = "human"
        with h5py.File(file_path) as file:
            self.meta.start_timestamp = h5_utils.read_start_timestamp(file)
            self.meta.duration = h5_utils.read_recording_duration(file)
            self.meta.utility_freq = h5_utils.read_utility_freq(file)
        self.H = H
        self.W = W

        # Configure logger
        logging_utils.add_logger_with_process_name(os.path.dirname(self.results_path))

        # Initialize bad times to correct indices

    def feature_matrix_name(self, line_length_window, n_channels):
        if line_length_window > 100:
            return f"V_LL_{line_length_window/100:1.1f}s"
        return f"V_LL_{line_length_window}ms_c{n_channels}"

    def model_name(self, h_init: bool, w_init: bool):
        name = "nmf_"
        if self.sparseness > 0:
            version_name = "W" if self.version == "l" else "H"
            name += f"s{version_name}_{self.sparseness:1.2f}"

        if h_init:
            name += "initH_"
        if w_init:
            name += "initW_"

        return name[:-1]

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
        num_bins = 100
        bins = np.linspace(0, 1, num_bins + 1)
        bin_width = bins[1] - bins[0]

        cdfs = np.zeros((num_bins, len(self.ranks)))
        areas = np.zeros(len(self.ranks))

        for idx, consensus in enumerate(consensus_matrices):
            cdf_vals = self.__compute_cdf(consensus, bins)
            areas[idx] = self.__compute_cdf_area(cdf_vals, bin_width)
            cdfs[:, idx] = cdf_vals

        delta_k, delta_y = self.__compute_delta_k(areas, cdfs)
        k_opt = self.ranks[np.argmax(delta_k)]

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
        nmf_classifier = Nmf(
            rank=rank, sparseness=self.sparseness, version=self.version
        )

        # Run NMF consensus clustering for specified rank and number of runs (default = 100)
        metrics, consensus, h, w = nmf_classifier.nmf_run(
            V=preprocessed_data,
            n_runs=n_runs,
            H=self.H,
            W=self.W,
        )

        return (
            metrics,
            consensus,
            h,
            w,
        )

    def parallel_processing(
        self,
        preprocessed_data: np.ndarray[np.dtype[float]],
        n_cores: int = 1,
    ) -> Tuple[
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
        Dict[int, np.ndarray[np.dtype[int]]],
        Dict[int, float],
        Dict[int, int],
    ]:
        # List of ranks to run NMF for
        nr_ranks = len(self.ranks)
        data_matrix = preprocessed_data.copy()

        # Using all cores except 2 if necessary
        n_cores = min(n_cores, nr_ranks)

        logger.info(
            f"Running NMF on {n_cores if nr_ranks > n_cores else nr_ranks} cores "
            f"for ranks {self.ranks}, sparsity {self.sparseness} with model {self.version}, {self.nmf_runs} runs each"
        )

        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.starmap(
                self.perform_nmf_steps_for_rank,
                [(data_matrix, rank, self.nmf_runs) for rank in self.ranks],
            )

        # Extract return objects from results
        metrics = [metrics for metrics, _, _, _ in results]
        consensus_matrices = [consensus for _, consensus, _, _ in results]
        h_matrices = [h_best for _, _, h_best, _ in results]
        w_matrices = [w_best for _, _, _, w_best in results]

        # Calculate final statistics
        C, delta_k, delta_y, _idx_opt = self.__calculate_statistics(consensus_matrices)

        # Generate metrics data frame
        metrics_df = pd.DataFrame(metrics)
        metrics_df["AUC"] = C
        metrics_df["delta_k (CDF)"] = delta_k
        metrics_df["delta_y (KL-div)"] = delta_y

        return h_matrices, w_matrices, consensus_matrices, metrics_df

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
        n_cores: int = 1,
        load_line_length_if_available: bool = True,
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
            Window length used for the line-length operation (in milliseconds).

        n_cores, default = 1
            Number of cores to use for computation.

        load_line_length_if_available: bool, default: True
            If true, checks whether there is a line length feature matrix available with the
            current configuration and if available, loads it instead of computing.

        Returns
        -------
        Tuple[List[BasisFunction], List[ActivationFunction]]
            Two lists containing the :py:mod:`~spidet.domain.BasisFunction`
            and :py:mod:`~spidet.domain.ActivationFunction`, where each activation function contains
            the corresponding detected events.
        """
        fm_name = self.feature_matrix_name(line_length_window, len(channel_paths))
        fm_group = self.nmf_dataset.feature_matrix(fm_name)
        if load_line_length_if_available and fm_group.has_dset(
            fm_group._feature_matrix_label
        ):
            line_length_matrix = self._load_line_length(fm_group=fm_group)
        else:
            channel_names, line_length_matrix = self._compute_line_length(
                channel_paths=channel_paths,
                exclude=exclude,
                bipolar_reference=bipolar_reference,
                leads=leads,
                notch_freq=notch_freq,
                resampling_freq=resampling_freq,
                bandpass_cutoff_low=bandpass_cutoff_low,
                bandpass_cutoff_high=bandpass_cutoff_high,
                line_length_freq=line_length_freq,
                line_length_window=line_length_window,
                n_cores=n_cores,
            )
            self._save_line_length(
                fm_group=fm_group,
                channel_names=channel_names,
                line_length_matrix=line_length_matrix,
            )

        # Run parallelized NMF
        (
            h_matrices,
            w_matrices,
            consensus_matrices,
            metrics,
        ) = self.parallel_processing(
            preprocessed_data=line_length_matrix,
            n_cores=n_cores,
        )

        # free some memory
        del line_length_matrix

        # Save NMFs
        h_init = self.H != None
        w_init = self.W != None

        for rank, w, h, cm in zip(
            self.ranks, w_matrices, h_matrices, consensus_matrices
        ):
            model = fm_group.by_value(rank).model(self.model_name(h_init, w_init))
            model.w = w
            model.h = h
            metrics_for_rank = metrics[metrics["Rank"] == rank]

            metrics_folder = model.metrics()
            metrics_folder.write_metric("consensus_matrix", cm)
            for name in metrics_for_rank:
                metrics_folder.write_metric(name, metrics_for_rank[name].values[0])

    def _compute_line_length(
        self,
        channel_paths,
        exclude,
        bipolar_reference,
        leads,
        notch_freq,
        resampling_freq,
        bandpass_cutoff_low,
        bandpass_cutoff_high,
        line_length_freq,
        line_length_window,
        n_cores,
    ):
        # Instantiate a LineLength instance
        line_length = LineLength(
            file_path=self.file_path,
            dataset_paths=channel_paths,
            exclude=exclude,
            bipolar_reference=bipolar_reference,
            leads=leads,
            bad_times=self.bad_times,
        )

        logger.info("Computing line length")
        # Perform line length steps to compute line length
        (
            _,
            channel_names,
            line_length_matrix,
        ) = line_length.apply_parallel_line_length_pipeline(
            notch_freq=notch_freq,
            resampling_freq=resampling_freq,
            bandpass_cutoff_low=bandpass_cutoff_low,
            bandpass_cutoff_high=bandpass_cutoff_high,
            line_length_freq=line_length_freq,
            line_length_window=line_length_window,
            n_cores=n_cores,
        )
        # Normalize line length data
        line_length_matrix = normalize(line_length_matrix)

        return channel_names, line_length_matrix

    def _load_line_length(self, fm_group: FeatureMatrixGroup) -> (list, np.ndarray):
        logger.info("Loading precomputed line length")
        return fm_group.feature_matrix

    def _save_line_length(
        self,
        fm_group: FeatureMatrixGroup,
        channel_names: list,
        line_length_matrix: np.ndarray,
    ) -> None:
        # Save feature matrix (line length)
        fm_group.feature_matrix = line_length_matrix
        fm_group.feature_names = channel_names
        fm_group.feature_units = ["uV" for _ in channel_names]
        fm_group.sfreq = self.line_length_freq
        fm_group.processing = "Butterworth [forward, backward, 0.1 - 200Hz], Notch [line noise and harmonics], rescaled median [20 uV], resampled [500 Hz]"
