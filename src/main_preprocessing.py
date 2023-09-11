import argparse
import multiprocessing

from preprocessing.pipeline import parallel_preprocessing
from loader.loader import read_file
from src.utils import logging_utils

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

    print("DONE preprocessing")
