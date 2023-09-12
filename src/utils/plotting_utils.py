from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import genfromtxt


def plot_consensus_matrices(folder: str, labels: List[str]):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Rank 2
    file_path = folder + "/k=2/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[0, 0].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[0, 0].set_title("k = 2")

    # Rank 3
    file_path = folder + "/k=3/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[0, 1].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[0, 1].set_title("k = 3")

    # Rank 4
    file_path = folder + "/k=4/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[1, 0].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[1, 0].set_title("k = 4")

    # Rank 5
    file_path = folder + "/k=5/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[1, 1].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[1, 1].set_title("k = 5")

    # Add labels
    ax[0, 0].set_xticks(range(len(labels)))
    ax[0, 0].set_xticklabels(labels, fontsize=3, rotation=90)
    ax[0, 0].set_yticks(range(len(labels)))
    ax[0, 0].set_yticklabels(labels, fontsize=3)
    ax[0, 0].tick_params(bottom=False, top=False, left=False)

    ax[0, 1].set_xticks(range(len(labels)))
    ax[0, 1].set_xticklabels(labels, fontsize=3, rotation=90)
    ax[0, 1].set_yticks(range(len(labels)))
    ax[0, 1].set_yticklabels(labels, fontsize=3)
    ax[0, 1].tick_params(bottom=False, top=False, left=False)

    ax[1, 0].set_xticks(range(len(labels)))
    ax[1, 0].set_xticklabels(labels, fontsize=3, rotation=90)
    ax[1, 0].set_yticks(range(len(labels)))
    ax[1, 0].set_yticklabels(labels, fontsize=3)
    ax[1, 0].tick_params(bottom=False, top=False, left=False)

    ax[1, 1].set_xticks(range(len(labels)))
    ax[1, 1].set_xticklabels(labels, fontsize=3, rotation=90)
    ax[1, 1].set_yticks(range(len(labels)))
    ax[1, 1].set_yticklabels(labels, fontsize=3)
    ax[1, 1].tick_params(bottom=False, top=False, left=False)

    fig.suptitle(folder[folder.rfind("/") + 1 :])
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlGn"]), ax=ax, shrink=0.5)

    plt.savefig(folder + "/consensus_matrices.pdf", bbox_inches="tight")


def plot_W_matrix(folder: str, labels: List[str]):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Rank 2
    file_path = folder + "/k=2/W_best.csv"
    W_best = genfromtxt(file_path, delimiter=",")
    ax[0, 0].imshow(W_best, cmap=mpl.colormaps["YlOrRd"], aspect="auto")
    ax[0, 0].set_title("k = 2")

    # Rank 3
    file_path = folder + "/k=3/W_best.csv"
    W_best = genfromtxt(file_path, delimiter=",")
    ax[0, 1].imshow(W_best, cmap=mpl.colormaps["YlOrRd"], aspect="auto")
    ax[0, 1].set_title("k = 3")

    # Rank 4
    file_path = folder + "/k=4/W_best.csv"
    W_best = genfromtxt(file_path, delimiter=",")
    ax[1, 0].imshow(W_best, cmap=mpl.colormaps["YlOrRd"], aspect="auto")
    ax[1, 0].set_title("k = 4")

    # Rank 5
    file_path = folder + "/k=5/W_best.csv"
    W_best = genfromtxt(file_path, delimiter=",")
    ax[1, 1].imshow(W_best, cmap=mpl.colormaps["YlOrRd"], aspect="auto")
    ax[1, 1].set_title("k = 5")

    # Add labels
    ax[0, 0].set_xticks(range(2))
    ax[0, 0].set_xticklabels(["W0", "W1"], fontsize=10)
    ax[0, 0].set_yticks(range(len(labels)))
    ax[0, 0].set_yticklabels(labels, fontsize=3)
    ax[0, 0].tick_params(bottom=False, top=False, left=False)

    ax[0, 1].set_xticks(range(3))
    ax[0, 1].set_xticklabels(["W0", "W1", "W3"], fontsize=10)
    ax[0, 1].set_yticks(range(len(labels)))
    ax[0, 1].set_yticklabels(labels, fontsize=3)
    ax[0, 1].tick_params(bottom=False, top=False, left=False)

    ax[1, 0].set_xticks(range(4))
    ax[1, 0].set_xticklabels(["W0", "W1", "W3", "W4"], fontsize=10)
    ax[1, 0].set_yticks(range(len(labels)))
    ax[1, 0].set_yticklabels(labels, fontsize=3)
    ax[1, 0].tick_params(bottom=False, top=False, left=False)

    ax[1, 1].set_xticks(range(5))
    ax[1, 1].set_xticklabels(["W0", "W1", "W3", "W4", "W5"], fontsize=10)
    ax[1, 1].set_yticks(range(len(labels)))
    ax[1, 1].set_yticklabels(labels, fontsize=3)
    ax[1, 1].tick_params(bottom=False, top=False, left=False)

    fig.suptitle(folder[folder.rfind("/") + 1 :])
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlOrRd"]), ax=ax, shrink=0.5)

    plt.savefig(folder + "/W_best.pdf", bbox_inches="tight")
