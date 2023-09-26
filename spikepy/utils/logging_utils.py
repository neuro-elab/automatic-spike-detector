import sys

from loguru import logger


def add_logger_with_process_name():
    logger.remove(0)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <red>|</red> "
        "<level>{level: <8}</level> <red>|</red> "
        "<level>{process.name}</level> <red>|</red> "
        "<cyan>{name}</cyan><red>:</red><cyan>{function}</cyan><red>:</red><cyan>{line}</cyan> <red>-</red> "
        "<level>{message}</level>",
    )
