from typing import Tuple, Dict

import numpy as np
from sklearn.cluster import KMeans


class BasisFunctionClusterer(KMeans):
    def __init__(self, n_clusters: int = 2, use_cosine_dist: bool = False):
        super().__init__(n_clusters=n_clusters, n_init=30)
        self.use_cosine_dist = use_cosine_dist

    def cluster_and_sort(
        self, h_matrix: np.ndarray, w_matrix: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        rank: int = h_matrix.shape[0]

        # Get empirical H PDF
        unique_vals: np.ndarray = np.unique(
            np.append(0, np.rint(h_matrix.flatten() * 10000) / 10000)
        )
        n_obs: int = unique_vals.size
        pdf: np.ndarray = np.zeros((n_obs, rank))

        for idx in range(n_obs - 1):
            pdf[idx, :] = (
                np.sum(
                    np.bitwise_and(
                        h_matrix > unique_vals[idx], h_matrix < unique_vals[idx + 1]
                    ),
                    1,
                )
                / h_matrix.shape[1]
            )

        if self.use_cosine_dist:
            # Normalize data to get cosine distance
            length = np.sqrt((pdf**2).sum(axis=1))[:, None]
            pdf = np.divide(pdf, length, out=np.zeros_like(pdf), where=length != 0)

        # Clustering
        cluster_indices = self.fit_predict(pdf.T)

        # Correct for arbitrary attribution of indices
        if np.median(np.median(h_matrix[cluster_indices == 0, :])) < np.median(
            np.median(h_matrix[cluster_indices == 1, :])
        ):
            cluster_indices = cluster_indices % 2 + 1
            cluster_indices = np.array(
                [1 if idx == 1 else 0 for idx in cluster_indices]
            )

        # Assign basis functions / samples to clusters
        cluster_assignments = np.vstack((cluster_indices, np.arange(rank))).T
        sorted_assignments = cluster_assignments[cluster_assignments[:, 0].argsort()]

        # Create dict indicating whether index codes for basis fct (1) or noise (0)
        assignments = {
            idx: assignment[0] for idx, assignment in enumerate(sorted_assignments)
        }

        # Sort W and H by cluster assignment
        if w_matrix is not None:
            w_matrix = w_matrix[:, sorted_assignments[:, 1]]
        h_matrix = h_matrix[sorted_assignments[:, 1], :]

        return w_matrix, h_matrix, assignments
