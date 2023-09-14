from typing import Any

import numpy as np
from sklearn.cluster import KMeans


class BasisFunctionClusterer(KMeans):
    def __init__(self, n_clusters: int = 2, use_cosine_dist: bool = False):
        super().__init__(n_clusters=n_clusters, n_init=30)
        self.use_cosine_dist = use_cosine_dist

    def cluster(self, h_matrix: np.ndarray) -> np.ndarray:
        # Get empirical H PDF
        pdf = np.unique(np.append(0, np.round(h_matrix.flatten() * 1000) / 1000))
        n_obs = pdf.size
        y = np.zeros((n_obs, h_matrix.shape[0]))

        for idx in range(n_obs - 1):
            y[idx, :] = (
                np.sum(np.bitwise_and(h_matrix > pdf[idx], h_matrix < pdf[idx + 1]), 1)
                / h_matrix.shape[1]
            )

        if self.use_cosine_dist:
            # Normalize data to get cosine distance
            length = np.sqrt((y**2).sum(axis=1))[:, None]
            y = np.divide(y, length, out=np.zeros_like(y), where=length != 0)

        # Clustering
        return self.fit_predict(y.T)
