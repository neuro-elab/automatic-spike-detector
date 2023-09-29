import argparse
import multiprocessing
import os

import numpy as np

from spidet.spike_detection.line_length import LineLength
from spidet.utils import logging_utils

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

LEAD_PREFIXES_EL010 = ["Hip2", "Temp", "FrOr", "In An", "InPo", "Hip1"]

DATASET_PATHS_SZ2 = [
    "/traces/raw/Hip1",
    "/traces/raw/Hip2",
    "/traces/raw/Hip3",
    "/traces/raw/Hip4",
    "/traces/raw/Hip5",
    "/traces/raw/Hip6",
    "/traces/raw/Hip7",
    "/traces/raw/Hip8",
    "/traces/raw/Hip9",
    "/traces/raw/Hip10",
    "/traces/raw/Hip11",
    "/traces/raw/Hip12",
    "/traces/raw/aHip1",
    "/traces/raw/aHip2",
    "/traces/raw/aHip3",
    "/traces/raw/aHip4",
    "/traces/raw/aHip5",
    "/traces/raw/aHip6",
    "/traces/raw/aHip7",
    "/traces/raw/aHip8",
    "/traces/raw/aHip9",
    "/traces/raw/aHip10",
    "/traces/raw/aHip11",
    "/traces/raw/aHip12",
]

LEAD_PREFIXES_SZ2 = ["Hip", "aHip"]

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be processed", required=True
    )

    file: str = parser.parse_args().file
    path_to_file = file[: file.rfind("/")]
    filename_for_saving = (
        file[file.rfind("/") + 1 :].replace(".", "_").replace(" ", "_")
    )

    # configure logger
    logging_utils.add_logger_with_process_name()

    # Instantiate a LineLength instance
    line_length = LineLength(
        file_path=file,
        dataset_paths=DATASET_PATHS_SZ2,
        bipolar_reference=True,
        leads=LEAD_PREFIXES_EL010,
    )

    # Perform line length steps to compute unique line length
    spike_detection_function = line_length.compute_unique_line_length()

    # Perform line length steps to compute line length
    (
        start_timestamp,
        channel_names,
        line_length_matrix,
    ) = line_length.apply_parallel_line_length_pipeline()

    multiprocessing.freeze_support()

    os.makedirs(filename_for_saving, exist_ok=True)

    data_path = os.path.join(filename_for_saving, "line_length.csv")
    np.savetxt(data_path, line_length_matrix, delimiter=",")

    data_path = os.path.join(filename_for_saving, "std_line_length.csv")
    np.savetxt(data_path, spike_detection_function.data_array, delimiter=",")

    print("DONE preprocess")
