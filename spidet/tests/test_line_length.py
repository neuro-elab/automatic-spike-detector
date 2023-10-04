import argparse
import multiprocessing
import os

import numpy as np

from spidet.domain.SpikeDetectionFunction import SpikeDetectionFunction
from spidet.spike_detection.line_length import LineLength
from spidet.utils import logging_utils
from spidet.tests.variables import (
    DATASET_PATHS_SZ2,
    LEAD_PREFIXES_SZ2,
    DATASET_PATHS_EL010,
    LEAD_PREFIXES_EL010,
)

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
        dataset_paths=DATASET_PATHS_EL010,
        bipolar_reference=True,
        leads=LEAD_PREFIXES_EL010,
    )

    # Perform line length steps to compute unique line length
    spike_detection_function: SpikeDetectionFunction = (
        line_length.compute_unique_line_length()
    )

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
