import argparse

from spidet.utils import file_utils
from spidet.utils.variables import LEAD_PREFIXES_EL010, DATASET_PATHS_EL010

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be converted", required=True
    )

    file: str = parser.parse_args().file

    file_utils.filter_leads_and_re_reference(
        file_path=file, channel_paths=DATASET_PATHS_EL010, leads=LEAD_PREFIXES_EL010
    )
