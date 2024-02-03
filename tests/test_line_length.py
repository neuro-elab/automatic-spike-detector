import argparse
import multiprocessing
import os

import numpy as np

from spidet.domain.ActivationFunction import ActivationFunction
from spidet.spike_detection.line_length import LineLength
from spidet.utils import logging_utils
from spidet.utils.variables import (
    DATASET_PATHS_008,
    LEAD_PREFIXES_008,
    DATASET_PATHS_SZ2,
    LEAD_PREFIXES_SZ2,
    DATASET_PATHS_EL003,
    LEAD_PREFIXES_EL003,
    DATASET_PATHS_EL003_BIP,
    CHANNEL_NAMES_005,
    LEAD_PREFIXES_005,
    CHANNELS_EL010_FIF,
    PREFIXES_EL010_FIF,
)

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be processed", required=True
    )
    parser.add_argument("--bt", help="path to bad times file", required=False)

    file: str = parser.parse_args().file
    bad_times_file: str = parser.parse_args().bt
    path_to_file = file[: file.rfind("/")]
    filename_for_saving = (
        file[file.rfind("/") + 1 :].replace(".", "_").replace(" ", "_")
    )

    # configure logger
    logging_utils.add_logger_with_process_name()

    # Define bad times
    if bad_times_file is not None:
        bad_times = np.genfromtxt(bad_times_file, delimiter=",")
    else:
        bad_times = None

    # Instantiate a LineLength instance
    line_length = LineLength(
        file_path=file,
        dataset_paths=CHANNEL_NAMES_005,
        bipolar_reference=False,
        leads=LEAD_PREFIXES_005,
        bad_times=bad_times,
    )

    # Perform line length steps to compute unique line length
    activation_function: ActivationFunction = line_length.compute_unique_line_length()

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
    np.savetxt(data_path, activation_function.data_array, delimiter=",")

    print("DONE preprocess")
