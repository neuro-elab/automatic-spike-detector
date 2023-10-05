import argparse

from spidet.utils import file_utils

if __name__ == "__main__":
    # parse cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="full path to file to be converted", required=True
    )
    parser.add_argument("--save", help="path to where to save the file", required=True)

    file: str = parser.parse_args().file
    save: str = parser.parse_args().save

    file_utils.convert_fif_to_edf(file_path=file, save_path=save)
