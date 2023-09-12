import argparse

from src.utils import plotting_utils

LABELS_EL010 = [
    "Hip21",
    "Hip22",
    "Hip23",
    "Hip24",
    "Hip25",
    "Hip26",
    "Hip27",
    "Hip28",
    "Hip29",
    "Hip210",
    "Temp1",
    "Temp2",
    "Temp3",
    "Temp4",
    "Temp5",
    "Temp6",
    "Temp7",
    "Temp8",
    "Temp9",
    "Temp10",
    "FrOr1",
    "FrOr2",
    "FrOr3",
    "FrOr4",
    "FrOr5",
    "FrOr6",
    "FrOr7",
    "FrOr8",
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
    "In An8",
    "In An9",
    "In An10",
    "In An11",
    "In An12",
    "InPo1",
    "InPo2",
    "InPo3",
    "InPo4",
    "InPo5",
    "InPo6",
    "InPo7",
    "InPo8",
    "InPo9",
    "InPo10",
    "InPo11",
    "InPo12",
    "Hip11",
    "Hip12",
    "Hip13",
    "Hip14",
    "Hip15",
    "Hip16",
    "Hip17",
    "Hip18",
    "Hip19",
    "Hip110",
    "Hip111",
    "Hip112",
]

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", help="folder where the results reside", required=True
    )

    folder: str = parser.parse_args().folder

    # Plot consensus matrices
    plotting_utils.plot_consensus_matrices(folder, LABELS_EL010)

    # Plot W matrices
    plotting_utils.plot_W_matrix(folder, LABELS_EL010)
