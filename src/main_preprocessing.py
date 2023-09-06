import multiprocessing

from preprocessing.pipeline import parallel_preprocessing
from loader.loader import read_h5_file

FILEPATH = "../../../Data/"
FILENAME = "Sz2.h5"

if __name__ == "__main__":
    data = read_h5_file(FILEPATH + FILENAME)
    preprocessed = parallel_preprocessing(data)
    multiprocessing.freeze_support()

    print("DONE preprocessing")
