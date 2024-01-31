import argparse
from typing import List
from datetime import datetime

from loguru import logger

from spidet.domain.ActivationFunction import ActivationFunction
from spidet.load.data_loading import DataLoader
from spidet.utils import logging_utils

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

    # Load spike activation functions
    activation_functions: List[
        ActivationFunction
    ] = data_loader.load_activation_functions(
        file_path=file, start_timestamp=start_datetime.timestamp()
    )

    logger.debug(
        f"Loaded the following spike activation functions:\n {activation_functions}"
    )
