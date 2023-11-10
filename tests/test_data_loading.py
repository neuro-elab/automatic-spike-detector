import argparse

from datetime import datetime

from spidet.load.data_loading import DataLoader
from spidet.utils import logging_utils
from spidet.utils.variables import DATASET_PATHS_EL003_BIP
from loguru import logger

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="full path to file to be loaded", required=True)

    file: str = parser.parse_args().file

    # configure logger
    logging_utils.add_logger_with_process_name()

    start_datetime = datetime(2021, 11, 11, 16, 1, 20)

    # Initialize data loader
    data_loader = DataLoader()

    traces = data_loader.read_file(path=file, channel_paths=DATASET_PATHS_EL003_BIP)

    logger.debug(f"Channels: {[trace.label for trace in traces]}")
