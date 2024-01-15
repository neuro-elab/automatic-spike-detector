from typing import Tuple, Dict

import nimfa
from nimfa.utils.linalg import *
import numpy as np
from loguru import logger
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from spidet.domain.Nmfsc import Nmfsc


class Nmf:
    def __init__(self, rank: int, use_sparsness_constraint: bool = False):
        self.rank = rank
        self.use_sparsness_constraint = use_sparsness_constraint

    def __calculate_cophenetic_corr(self, A: np.ndarray) -> np.ndarray:
        """
        Compute the cophenetic correlation coefficient for matrix A.

        Parameters:
        - A : numpy.ndarray
            Input matrix.

        Returns:
        - float
            Cophenetic correlation coefficient.
        """
        # Extract the values from the lower triangle of A
        avec = np.array(
            [A[i, j] for i in range(A.shape[0] - 1) for j in range(i + 1, A.shape[1])]
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
        preprocessed_data: np.ndarray,
        n_runs: int,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        data_matrix = preprocessed_data
        consensus = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
        obj = np.zeros(n_runs)
        lowest_obj = float("inf")
        h_best = None
        w_best = None

        if not self.use_sparsness_constraint:
            nmf = nimfa.Nmf(
                data_matrix.T, rank=self.rank, seed="random_vcol", max_iter=10
            )
        else:
            nmf = Nmfsc(data_matrix, rank=self.rank, max_iter=10)

        for n in range(n_runs):
            logger.debug(
                f"Rank {self.rank}, Run {n + 1}/{n_runs}: Perform matrix factorization"
            )
            if self.use_sparsness_constraint:
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
