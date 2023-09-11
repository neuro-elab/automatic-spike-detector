import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import genfromtxt


def plot_consensus_matrices(folder: str):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # rank 2
    file_path = folder + "/k=2/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[0, 0].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[0, 0].set_title("k = 2")

    # rank 3
    file_path = folder + "/k=3/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[0, 1].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[0, 1].set_title("k = 3")

    # rank 4
    file_path = folder + "/k=4/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[1, 0].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[1, 0].set_title("k = 4")

    # rank 5
    file_path = folder + "/k=5/consensus_matrix.csv"
    consensus_matrix = genfromtxt(file_path, delimiter=",")
    ax[1, 1].matshow(consensus_matrix, cmap=mpl.colormaps["YlGn"])
    ax[1, 1].set_title("k = 5")

    fig.suptitle(folder[folder.rfind("/") + 1 :])
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlGn"]), ax=ax, shrink=0.6)

    plt.savefig(folder + "/consensus_matrices.pdf", bbox_inches="tight")
