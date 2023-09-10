import argparse

from src.utils import plotting_utils

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", help="folder where the results reside", required=True
    )

    folder: str = parser.parse_args().folder

    # Plot consensus matrices
    plotting_utils.plot_consensus_matrices(folder)
