import numpy as np
import pandas as pd
import os

from h5py import File, Dataset, Group

LINE_LENGTH_DATASET = "line_lengths"
CHANNELS_ATTR = "channels"
W_DATASET = "W"
H_DATASET = "H"
NMF_GROUP = "NMF"

ANNO_TIME = "annotations/time"
ANNO_TRIG = "annotations/text"
TRIGGER = "TRIG"


def find_bad_times(filepath: str) -> list:
    """Searches for bad times within the h5 data file at filepath

    Parameters
    ----------
    filepath : str
        Path to the h5 data file

    Returns
    -------
    list
        Intervals [start, stop] of bad times, in seconds since recording start.

    """
    with File(filepath, "r") as recording:
        if (
            recording["/time_grades/text"]
            and recording["/time_grades/time"]
            and recording["/time_grades/duration"]
        ):
            description = recording["/time_grades/text"]
            onset = recording["/time_grades/time"]
            duration = recording["/time_grades/duration"]
            expert_df = pd.DataFrame(
                {"Description": description, "Onset": onset, "Duration": duration}
            )
            expert_df["Description"] = expert_df["Description"].str.decode("utf8")
            expert_df["Offset"] = events.loc[:, "Onset"] + events.loc[:, "Duration"]
            mask = expert_df["Description"] == "NOISY"
            return expert_df.loc[:, ["Onset", "Offset"]][mask].to_numpy()


def change_interval(t, a: float, b: float, A: float = 0, B: float = 1):
    """
    Map t from interval [A, B] to interval [a, b]
    """
    t = (t - A) / (B - A)  # Normalize t to [0, 1]
    return b * t + a * (1 - t)


def detect_triggers(
    filepath: str,
    trig_path: str = ANNO_TRIG,
    times_path: str = ANNO_TIME,
    prefix=TRIGGER,
    pad_left=0.1,
    pad_right=1.0,
) -> np.ndarray:
    """Looks for triggers indicating stimulation events within a file describing an
    EEG Data recording and generates bad times based on it.

    Parameters
    ----------
    filepath : str
        H5 data file.
    trig_path : str
        Path to dataset storing event kind
    times_path : str
        Path to dataset storing event time
        Times are expected to be in seconds and relative to recording start.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2).
        Each row corresponds to an intervals: [trigger - pad_left, trigger + pad_right].
        for each trigger found.

    """
    trigs = find_values(filepath, trig_path, times_path, prefix)
    dur = duration(filepath)

    start = np.maximum(0, trigs - pad_left)  # subtract 0.1s before trig
    end = np.minimum(trigs + pad_right, dur)  # Add one second after trig

    return np.vstack((start, end)).T


def find_values(
    filepath: str, description_path: str, value_path: str, prefix: str
) -> np.ndarray:
    """Collect all values in the given h5 file from the given value_path where corresponding (index-wise)
    values of the description path are prefixed with the given prefix.
    description_path and value_path are expected to point to Datasets of the same length.

    Parameters
    ----------
    filepath : str
        Path to the h5 file
    description_path : str
        Path within the h5 file to a dataset describing the dataset at value_path
    value_path : str
        Path within the h5 file to a dataset
    prefix : str
        Prefix indicating the description in description_path

    Returns
    -------
    np.ndarray
        All the values from the dataset at value_path with the same index as all
        descriptions from the dataset at description_path where the description is starting
        with prefix.

    """
    with File(filepath, "r") as recording:
        if description_path in recording and value_path in recording:
            df = pd.DataFrame(
                {"descr": recording[description_path], "value": recording[value_path]}
            )
            df["descr"] = df["descr"].str.decode("utf8")
            return df[df["descr"].str.startswith(prefix)]["value"].values
        return np.array([])


def list_datasets(recording: File) -> list:
    dset_paths = []

    def visitor(name, node):
        if isinstance(node, Dataset):
            dset_paths.append(name)

    recording.visititems(visitor)

    return dset_paths


def find_dataset_paths(recording: File, path=""):
    if len(path) == 0:
        paths = []
        for key in recording.keys():
            paths += find_dataset_paths(recording, "/" + key)
        return paths

    if not path_valid(recording, path):
        return []

    elif isinstance(recording[path], Group):
        paths = []
        for key in recording[path].keys():
            paths += find_dataset_paths(recording, "/".join([path, key]))
        return paths
    elif isinstance(recording[path], Dataset) and not dataset_corrupt(recording, path):
        return [path]
    return []


def dataset_corrupt(recording, path):
    try:
        np.array(recording[path])
        return False
    except Exception as e:
        print(f"dataset corrupt: {path}: \t{e}")
        return True


def path_valid(recording, path):
    try:
        return len(recording[path])
    except Exception as e:
        print(f"path invalid: {path}: \t {e}")
        return 0


def find_channel_paths(recording: File):
    paths = find_dataset_paths(recording)
    return filter_list_for("bipolar/lead", paths)


def read_recording_duration(recording: File):
    return recording["meta"].attrs["duration"]


def duration(filepath: str):
    with File(filepath, "r") as file:
        return read_recording_duration(file)


def read_start_timestamp(recording: File):
    return int(recording["meta"].attrs["start_timestamp"])


def read_utility_freq(recording: File):
    return recording["meta"].attrs["utility_freq"]


def get_n_samples(recording: File):
    return recording["traces/raw"].attrs["n_samples"]


def frequency(filepath: str):
    with File(filepath, "r") as file:
        get_frequency(file)


def get_frequency(recording: File):
    n = get_n_samples(recording)
    t = read_recording_duration(recording)
    return int(n / t)


def load_data(recording, paths=None):
    if paths is None:
        paths = find_channel_paths(recording)

    data = []
    for path in paths:
        try:
            data.append(np.array(recording[path]))
        except Exception as e:
            print(f"data loading failed for path {path}:\t{e}")
    return np.vstack(data).astype(float)


def print_summary(recording: File):
    start_timestamp = read_start_timestamp(recording)
    n_samples = get_n_samples(recording)
    duration = read_recording_duration(recording)
    frequency = get_frequency(recording)
    channels = find_channel_paths(recording)
    print(
        f"start_timestamp:\t{start_timestamp}\nsamples:\t\t{n_samples}\nduration:\t\t{duration}\nfrequency:\t\t{frequency}\nchannels:\t\t{len(channels)}"
    )


def get_timestamp(recording: File, index: int):
    start_timestamp = read_start_timestamp(recording)
    duration = read_recording_duration(recording)
    n_samples = get_n_samples(recording)

    return start_timestamp + index * duration / n_samples


def get_timestamps(recording: File, indices):
    return [get_timestamp(recording, index) for index in indices]


def get_index(recording: File, time):
    """
    time: Time since start of recording measured in seconds.
    """
    duration = read_recording_duration(recording)
    n_samples = get_n_samples(recording)

    return int(n_samples * time / duration)


def filter_list_for(s: str, paths: list[str]):
    return list(filter(lambda path: s in path, paths))


def save_line_lengths_to_h5(
    directory,
    filename,
    channels,
    line_length_matrix,
    dset_path=LINE_LENGTH_DATASET,
    channel_attr_path=CHANNELS_ATTR,
):
    path = os.path.join(directory, filename)

    assert (
        len(channels) == line_length_matrix.shape[0]
    ), "Number of channels must match number of rows in line length matrix"
    assert os.access(directory, os.W_OK), "Directory has to be writable"

    with File(path, "a") as file:
        file[dset_path] = line_length_matrix
        file[dset_path].attrs[channel_attr_path] = channels


def read_line_lengths_from_h5(
    directory, filename, dset_path=LINE_LENGTH_DATASET, channel_attr_path=CHANNELS_ATTR
):
    path = os.path.join(directory, filename)

    assert os.access(path, os.R_OK), "Path has to be readable"

    with File(path, "r") as file:
        line_length_matrix = file[dset_path][()]
        channels = file[dset_path].attrs[channel_attr_path]

    return channels, line_length_matrix


def save_nmf_to_h5(directory, filename, W, H, wset_path=W_DATASET, hset_path=H_DATASET):
    path = os.path.join(directory, filename)

    assert os.access(directory, os.W_OK), "Directory has to be writeable"

    with File(path, "a") as file:
        file[wset_path] = W
        file[hset_path] = H


def read_nmf_from_h5(directory, filename, wset_path=W_DATASET, hset_path=H_DATASET):
    path = os.path.join(directory, filename)

    assert os.access(path, os.R_OK), "Path has to be readable"

    with File(path, "r") as file:
        W = file[wset_path][()]
        H = file[hset_path][()]
    return W, H
