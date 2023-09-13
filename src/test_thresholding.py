import argparse

from loguru import logger

from src.spike_detection import thresholding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", help="full path to directory of experiment", required=True
    )

    experiment_dir: str = parser.parse_args().dir

    spike_annotations = thresholding.annotate_spikes(experiment_dir + "/k=3")

    logger.debug(
        f"Got the following spike annotations for ranks in range (2, 10): {spike_annotations}"
    )
