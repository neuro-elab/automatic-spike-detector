import argparse
import multiprocessing
import os

import numpy as np

from preprocessing.pipeline import parallel_preprocessing
from loader.loader import read_file
from spidet.utils import logging_utils

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

    data = read_file(file)
    preprocessed = parallel_preprocessing(data)
    multiprocessing.freeze_support()

    os.makedirs(filename_for_saving, exist_ok=True)

    data_path = os.path.join(filename_for_saving, "preprocessed.csv")
    np.savetxt(data_path, preprocessed, delimiter=",")

    print("DONE preprocessing")
