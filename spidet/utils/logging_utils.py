import os.path
import sys

from loguru import logger


def add_logger_with_process_name(directory: str = None):
    logger.remove(0)
    if directory is not None:
        logger.add(
            os.path.join(directory, "file_{time}.log"),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <red>|</red> "
            "<level>{level: <8}</level> <red>|</red> "
            "<level>{process.name}</level> <red>|</red> "
            "<cyan>{name}</cyan><red>:</red><cyan>{function}</cyan><red>:</red><cyan>{line}</cyan> <red>-</red> "
            "<level>{message}</level>",
        )
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <red>|</red> "
        "<level>{level: <8}</level> <red>|</red> "
        "<level>{process.name}</level> <red>|</red> "
        "<cyan>{name}</cyan><red>:</red><cyan>{function}</cyan><red>:</red><cyan>{line}</cyan> <red>-</red> "
        "<level>{message}</level>",
    )
