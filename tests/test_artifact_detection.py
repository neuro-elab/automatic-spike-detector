import argparse

import numpy as np

from spidet.domain.Artifacts import Artifacts
from spidet.preprocess.artifact_detection import ArtifactDetector
from tests.variables import DATASET_PATHS_008, LEAD_PREFIXES_008

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be processed", required=True
    )

    file: str = parser.parse_args().file

    # Initialize artifact detector
    artifact_detector = ArtifactDetector()

    # Run artifact detection
    artifacts: Artifacts = artifact_detector.run(
        file_path=file,
        channel_paths=DATASET_PATHS_008,
        detect_stimulation_artifacts=True,
    )

    if artifacts.bad_times is not None:
        np.savetxt("bad_times.csv", artifacts.bad_times, delimiter=",")

    print(f"Bad channels: {artifacts.bad_channels}")
    print(f"Bad times: {artifacts.bad_times}")
