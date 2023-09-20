import os
import re
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import genfromtxt


def plot_w_and_consensus_matrix(experiment_dir: str, labels: List[str]) -> None:
    rank_dirs = get_rank_dirs_sorted(experiment_dir)
    first_rank = int(rank_dirs[0][-1:])

    nr_ranks = len(rank_dirs)

    nr_cols = 3 if nr_ranks >= 9 else 2
    nr_rows = int(
        nr_ranks / nr_cols
        if nr_ranks % nr_cols == 0
        else (nr_ranks + nr_ranks % nr_cols) / nr_cols
    )

    fig_w, ax_w = plt.subplots(nr_rows, nr_cols, figsize=(10, 10))
    fig_consensus, ax_consensus = plt.subplots(nr_rows, nr_cols, figsize=(10, 10))

    nr_ranks_plotted = 0
    for row in range(nr_rows):
        for col in range(nr_cols):
            if nr_ranks_plotted >= nr_ranks:
                break
            # PLot W matrix
            file_path_w = rank_dirs[nr_ranks_plotted] + "/W_best.csv"
            w_best = genfromtxt(file_path_w, delimiter=",")
            ax_w[row, col].imshow(w_best, cmap=mpl.colormaps["YlOrRd"], aspect="auto")
            ax_w[row, col].set_title(f"k = {nr_ranks_plotted + first_rank}")

            xticks_labels = [
                "W" + str(rank) for rank in range(nr_ranks_plotted + first_rank)
            ]
            ax_w[row, col].set_xticks(range(nr_ranks_plotted + first_rank))
            ax_w[row, col].set_xticklabels(xticks_labels, fontsize=6)
            ax_w[row, col].set_yticks(range(len(labels)))
            ax_w[row, col].set_yticklabels(labels, fontsize=3)
            ax_w[row, col].tick_params(bottom=False, top=False, left=False)

            # Plot consensus matrix
            file_path_consensus = rank_dirs[nr_ranks_plotted] + "/consensus_matrix.csv"
            consensus_matrix = genfromtxt(file_path_consensus, delimiter=",")
            ax_consensus[row, col].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
            ax_consensus[row, col].set_title(f"k = {nr_ranks_plotted + first_rank}")

            ax_consensus[row, col].set_xticks(range(len(labels)))
            ax_consensus[row, col].set_xticklabels(labels, fontsize=3, rotation=90)
            ax_consensus[row, col].set_yticks(range(len(labels)))
            ax_consensus[row, col].set_yticklabels(labels, fontsize=3)
            ax_consensus[row, col].tick_params(bottom=False, top=False, left=False)

            nr_ranks_plotted += 1

    fig_w.suptitle("W matrix - " + experiment_dir[experiment_dir.rfind("/") + 1 :])
    fig_w.subplots_adjust(hspace=0.3)
    fig_w.colorbar(
        mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlOrRd"]), ax=ax_w, shrink=0.5
    )
    fig_w.savefig(experiment_dir + "/W_best.pdf", bbox_inches="tight")

    fig_consensus.suptitle(
        "CONSENSUS matrix - " + experiment_dir[experiment_dir.rfind("/") + 1 :]
    )
    fig_consensus.subplots_adjust(hspace=0.3)
    fig_consensus.colorbar(
        mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlGn"]), ax=ax_w, shrink=0.5
    )
    fig_consensus.savefig(experiment_dir + "/consensus_matrix.pdf", bbox_inches="tight")


def plot_h_matrix_spike_period(experiment_dir: str, labels: List[str] = None) -> None:
    rank_dirs = get_rank_dirs_sorted(experiment_dir)

    fig, ax = plt.subplots(len(rank_dirs), 1, figsize=(20, 20))

    for idx in range(len(rank_dirs)):
        file_path = rank_dirs[idx] + "/H_best.csv"
        h_best = genfromtxt(file_path, delimiter=",")
        ax[idx].plot(h_best[:, 9750:9900].T)
        ax[idx].set_title("k = " + str(idx + 2))

    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("H matrix - " + experiment_dir[experiment_dir.rfind("/") + 1 :])
    plt.savefig(experiment_dir + "/H_best.pdf")


def get_rank_dirs_sorted(experiment_dir: str) -> List[str]:
    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        experiment_dir + "/" + k_dir
        for k_dir in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, k_dir)) and "k=" in k_dir
    ]

    return sorted(rank_dirs, key=lambda x: int(re.search(r"\d+$", x).group()))
