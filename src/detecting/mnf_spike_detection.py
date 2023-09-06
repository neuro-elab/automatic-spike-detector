import multiprocessing
import os
import datetime
import numpy as np
import pandas as pd
import nimfa
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.special import rel_entr


def compute_cdf(matrix, bins):
    N = matrix.shape[0]
    values = np.array([matrix[i, j] for i in range(N - 1) for j in range(i + 1, N)])
    counts, _ = np.histogram(values, bins=bins, density=True)
    cdf_vals = np.cumsum(counts) / (N * (N - 1) / 2)
    return cdf_vals + 1e-10  # we have to add a small offset to avoid div0!


def compute_cdf_area(cdf_vals, bin_width):
    return np.sum(cdf_vals[:-1]) * bin_width


def compute_delta_k(areas, cdfs):
    delta_k = np.zeros(len(areas))
    delta_y = np.zeros(len(areas))
    delta_k[0] = areas[0]
    for i in range(1, len(areas)):
        delta_k[i] = (areas[i] - areas[i - 1]) / areas[i - 1]
        delta_y[i] = sum(rel_entr(cdfs[:, i], cdfs[:, i - 1]))
    return delta_k, delta_y


def calculate_statistics(M, rank_range, bins):
    k_min, k_max = rank_range
    bin_width = bins[1] - bins[0]

    num_bins = len(bins) - 1
    cdfs = np.zeros((num_bins, k_max - k_min + 1))
    areas = np.zeros(k_max - k_min + 1)

    for i, m in enumerate(M):
        cdf_vals = compute_cdf(m, bins)
        areas[i] = compute_cdf_area(cdf_vals, bin_width)
        cdfs[:, i] = cdf_vals

    delta_k, delta_y = compute_delta_k(areas, cdfs)
    k_opt = np.argmax(delta_k) + k_min if delta_k.size > 0 else k_min

    return areas, delta_k, delta_y, k_opt


def calculate_cophenetic_corr(A):
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
    avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                     for j in range(i + 1, A.shape[1])])

    # Consensus entries are similarities, conversion to distances
    Y = 1 - avec

    # Hierarchical clustering
    Z = linkage(Y, method='average')

    # Cophenetic correlation coefficient of a hierarchical clustering
    coph = cophenet(Z, Y)[0]

    return coph


def nmf_run(args):
    data_matrix, rank, n_runs, target_clusters = args
    consensus = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))
    obj = np.zeros(n_runs)
    connectivity_matrices = []
    lowest_obj = float('inf')
    best_H = None
    best_W = None

    for n in range(n_runs):
        nmf = nimfa.Nmf(data_matrix.T, rank=rank, seed="random_vcol", max_iter=10)
        fit = nmf()
        connectivity = fit.fit.connectivity()
        connectivity_matrices.append(connectivity)
        consensus += connectivity
        obj[n] = fit.fit.final_obj
        if obj[n] < lowest_obj:
            lowest_obj = obj[n]
            best_H = fit.coef()
            best_W = fit.basis()

    consensus /= n_runs
    coph = calculate_cophenetic_corr(consensus)
    instability = 1 - coph

    # Computing ARI if target_clusters is provided
    ari = None
    if target_clusters is not None:
        clusters = np.array([np.argmax(best_H[:, i]) for i in range(best_H.shape[1])])
        ari = adjusted_rand_score(target_clusters, clusters)

    # Storing metrics
    metrics = {
        "Rank": rank,
        "Min Final Obj": lowest_obj,
        "Adjusted Rand Index": ari,
        "Cophenetic Correlation": coph,
        "Instability index": instability
    }

    return metrics, consensus, connectivity_matrices, best_H, best_W


def parallel_nmf_consensus_clustering(data_matrix, rank_range, n_runs, target_clusters=None):
    """
    Parallel NMF consensus clustering.

    Parameters:
    - data_matrix : numpy.ndarray
        Data matrix.
    - rank_range : tuple of int
        Range of ranks, e.g., (k_min, k_max)
    - n_runs : int
        Number of runs for each rank.
    - target_clusters : numpy.ndarray or None
        Target clusters, if available.

    Returns:
    - experiment_dir : str
        Directory where results are saved.
    - M : numpy.ndarray
        3D array where each slice along the first axis is a consensus matrix for a specific rank.
    """
    # Create a directory for the experiment
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join("Experiment_" + timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # Using all available cores
    n_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(nmf_run, [(data_matrix, rank, n_runs, target_clusters) for rank in
                                     range(rank_range[0], rank_range[1] + 1)])

    # Create the M matrix
    n = data_matrix.shape[0]
    k_range = rank_range[1] - rank_range[0] + 1
    M = np.zeros((k_range, n, n))

    for idx, (metrics, consensus, connectivity_matrices, best_H, best_W) in enumerate(results):
        # Saving consensus matrix to M
        M[idx] = consensus

        # Saving other results
        rank = rank_range[0] + idx
        rank_dir = os.path.join(experiment_dir, f"k={rank}")
        os.makedirs(rank_dir, exist_ok=True)

        connectivity_dir = os.path.join(rank_dir, "connectivity_matrices")
        os.makedirs(connectivity_dir, exist_ok=True)

        # Saving connectivity matrices
        for idx, matrix in enumerate(connectivity_matrices):
            connectivity_path = os.path.join(connectivity_dir, f"connectivity_{idx + 1}.csv")
            np.savetxt(connectivity_path, matrix, delimiter=",")

        # Saving consensus matrix
        consensus_path = os.path.join(rank_dir, "consensus_matrix.csv")
        np.savetxt(consensus_path, consensus, delimiter=",")

        # Saving H_best and W_best matrices
        h_best_path = os.path.join(rank_dir, "H_best.csv")
        np.savetxt(h_best_path, best_H, delimiter=",")

        w_best_path = os.path.join(rank_dir, "W_best.csv")
        np.savetxt(w_best_path, best_W, delimiter=",")

    bins = np.linspace(0, 1, 101)
    C, delta_k, delta_y, k_opt = calculate_statistics(M, rank_range, bins)

    # Extend the metrics data frame
    metrics_df = pd.DataFrame([res[0] for res in results])
    metrics_df['C'] = C
    metrics_df['delta_k (AUC)'] = delta_k
    metrics_df['delta_y (KL-div)'] = delta_y
    print(f"Optimal k = {k_opt}")

    # Saving metrics as CSV
    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    return experiment_dir  # return the directory where results are saved and the M matrix


# Return the modified functions for use
nmf_run, parallel_nmf_consensus_clustering