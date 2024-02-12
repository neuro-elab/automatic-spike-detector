from typing import Tuple, Dict

import nimfa
import numpy as np
from loguru import logger
from scipy.cluster.hierarchy import linkage, cophenet

from spidet.domain.Nmfsc import Nmfsc


class Nmf:
    """
    This class hosts operations concerned with nonnegative matrix factorization (NMF).

    Parameters
    ----------

    rank: int
        The rank used for the nonnegative matrix factorization

    sparseness: float, optional, default = 0.0
        The sparseness parameter used in case NMF is run with sparseness constraints.
    """

    def __init__(self, rank: int, sparseness: float = 0.0):
        self.rank = rank
        self.sparseness = float(sparseness)

    @staticmethod
    def __calculate_cophenetic_corr(consensus_matrix: np.ndarray) -> np.ndarray:
        # Extract the values from the lower triangle of A
        avec = np.array(
            [
                consensus_matrix[i, j]
                for i in range(consensus_matrix.shape[0] - 1)
                for j in range(i + 1, consensus_matrix.shape[1])
            ]
        )

        # Consensus entries are similarities, conversion to distances
        Y = 1 - avec

        # Hierarchical clustering
        Z = linkage(Y, method="average")

        # Cophenetic correlation coefficient of a hierarchical clustering
        coph = cophenet(Z, Y)[0]

        return coph

    def nmf_run(
        self,
        preprocessed_data: np.ndarray[np.dtype[float]],
        n_runs: int,
    ) -> Tuple[
        Dict,
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
    ]:
        """
        This method performs a number of nonegative matrix factorization runs for the rank
        defined at initialization of the :py:class:`Nmf` object. At the beginning of each run the :math:`W`
        and :math:`H` matrices are randomly initialized. The algorithm runs until convergence
        (i.e., the reconstruction error < 1e-5) or for a maximum of 10 iterations.

        Parameters
        ----------
        preprocessed_data: numpy.ndarray[numpy.dtype[float]]
            The data used as input for NMF algorithm.

        n_runs: int
            The number of NMF runs.

        Returns
        -------
        Tuple[Dict,
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]],
        np.ndarray[np.dtype[float]]]
            A dictionary containing the rank, the minimum reconstruction error, the cophenetic correlation
            and the instability index, together with the consensus, :math:`W` and :math:`H` matrices.
        """
        data_matrix = preprocessed_data
        consensus = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
        obj = np.zeros(n_runs)
        lowest_obj = float("inf")
        h_best = None
        w_best = None

        if self.sparseness == 0.0:
            nmf = nimfa.Nmf(
                data_matrix.T, rank=self.rank, seed="random_vcol", max_iter=10
            )
        else:
            nmf = Nmfsc(data_matrix, rank=self.rank, max_iter=10, sW=self.sparseness)

        for n in range(n_runs):
            logger.debug(
                f"Rank {self.rank}, Run {n + 1}/{n_runs}: Perform matrix factorization"
            )
            if self.sparseness != 0.0:
                fit = nmf()
                consensus += fit.connectivity()
                obj[n] = fit.final_obj
                if obj[n] < lowest_obj:
                    logger.debug(
                        f"Rank {self.rank}, Run {n + 1}/{n_runs}: Update COEFFICIENTS and BASIS FCTs"
                    )
                    lowest_obj = obj[n]
                    w_best = np.array(fit.basis())
                    h_best = np.array(fit.coef())
            else:
                fit = nmf()
                consensus += fit.fit.connectivity()
                obj[n] = fit.fit.final_obj
                if obj[n] < lowest_obj:
                    logger.debug(
                        f"Rank {self.rank}, Run {n + 1}/{n_runs}: Update COEFFICIENTS and BASIS FCTs"
                    )
                    lowest_obj = obj[n]
                    w_best = np.array(fit.fit.coef().T)
                    h_best = np.array(fit.fit.basis().T)

        consensus /= n_runs
        coph = self.__calculate_cophenetic_corr(consensus)
        instability = 1 - coph

        # Storing metrics
        metrics = {
            "Rank": self.rank,
            "Min Final Obj": lowest_obj,
            "Cophenetic Correlation": coph,
            "Instability index": instability,
        }

        logger.debug(f"Rank {self.rank}: Finished {n_runs} iterations of NMF")

        return metrics, consensus, h_best, w_best
