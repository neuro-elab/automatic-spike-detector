import os.path

from mne.io import Raw, read_raw_fif
from mne.export import export_raw

FIF = "fif"


def convert_fif_to_edf(file_path: str, save_path: str) -> None:
    filename = file_path[file_path.rfind("/") + 1 :]
    filename = filename[: filename.rfind(".")]
    file_format = file_path[file_path.rfind(".") + 1 :].lower()

    if file_format != FIF:
        raise Exception(f"File format must be of type fif, but was {file_format}")

    raw_object: Raw = read_raw_fif(file_path, preload=True, verbose=False)

    filename = os.path.join(save_path, f"{filename}.edf")
    export_raw(fname=filename, raw=raw_object, fmt="edf")
