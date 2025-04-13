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


def find_triggers(filepath: str) -> list:
    with File(filepath) as file:
        if ANNO_TRIG in file and ANNO_TIME in file:
            df = pd.DataFrame({"annotations": file[ANNO_TRIG], "time": file[time]})
            df["annotations"] = df["annotations"].str.decode("utf8")
            return df[df["annotations"].str.startswith(TRIGGER)]["time"].values
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


def read_start_timestamp(recording: File):
    return int(recording["meta"].attrs["start_timestamp"])


def read_utility_freq(recording: File):
    return recording["meta"].attrs["utility_freq"]


def get_n_samples(recording: File):
    return len(recording[find_channel_paths(recording)[0]])


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
