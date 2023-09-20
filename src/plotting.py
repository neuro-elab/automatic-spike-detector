import argparse

from src.utils import plotting_utils

LABELS_EL010 = [
    "Hip21",
    "Hip22",
    "Hip23",
    "Hip28",
    "Hip29",
    "Hip210",
    "Temp1",
    "Temp2",
    "Temp4",
    "Temp5",
    "Temp6",
    "Temp7",
    "Temp8",
    "FrOr1",
    "FrOr9",
    "FrOr10",
    "FrOr11",
    "FrOr12",
    "In An1",
    "In An2",
    "In An3",
    "In An4",
    "In An5",
    "In An6",
    "In An7",
    "In An12",
    "InPo3",
    "InPo4",
    "InPo5",
    "InPo6",
    "InPo7",
    "InPo8",
    "InPo9",
    "InPo11",
    "InPo12",
    "Hip11",
    "Hip12",
    "Hip13",
    "Hip14",
    "Hip15",
    "Hip19",
    "Hip110",
    "Hip111",
]

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", help="folder where the results reside", required=True
    )

    folder: str = parser.parse_args().folder

    # Plot W matrices
    plotting_utils.plot_w_and_consensus_matrix(folder, LABELS_EL010)

    # plot H
    plotting_utils.plot_h_matrix_spike_period(folder)
