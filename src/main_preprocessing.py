import multiprocessing

from preprocessing.pipeline import parallel_preprocessing
from loader.loader import read_file
from src.utils import logging_utils

FILEPATH = "/home/dominik/Uni/HS2023/Thesis/Data/"
FILENAME = "Sz2.h5"

if __name__ == "__main__":
    # configure logger
    logging_utils.add_logger_with_process_name()

    data = read_file(FILEPATH + FILENAME)
    preprocessed = parallel_preprocessing(data)
    multiprocessing.freeze_support()

    print("DONE preprocessing")
