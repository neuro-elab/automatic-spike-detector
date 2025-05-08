from __future__ import annotations

import h5py as h5
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime

from spidet.utils.h5_utils import (
    read_recording_duration,
    read_start_timestamp,
    read_utility_freq,
)

META_GROUP = "meta"
NMF_GROUP = "nmf"

CREATION_DATE_LABEL = "creation_date"
SUBJECT_ID_LABEL = "subject_id"
SPECIES_LABEL = "species"
START_TIMESTAMP_LABEL = "start_timestamp"
DURATION_LABEL = "duration"
UTILITY_FREQ_LABEL = "utility_freq"

FEATURE_MATRIX_LABEL = "feature_matrix"
FEATURE_NAMES_LABEL = "feature_names"
FEATURE_UNITS_LABEL = "feature_units"
SFREQ_LABEL = "sfreq"
PROCESSING_LABEL = "processing"

W_LABEL = "w"
H_LABEL = "h"
PARAMETERS_LABEL = "parameters"
CONSENSUS_MATRIX_LABEL = "consensus_matrix"


class NMFData:
    def __init__(
        self,
        filepath: str,
        dataset_id="0",
        subject_id="",
        species="",
        start_timestamp="",
        duration="",
        utility_freq=0,
    ) -> None:
        assert os.path.exists(filepath) or os.access(os.path.dirname(filepath), os.W_OK)

        self._file_path = filepath

        if not os.path.exists(filepath):
            self._create_file()
            self.subject_id = subject_id
            self.species = species
            self.start_timestamp = start_timestamp
            self.duration = duration
            self.utility_freq = utility_freq

    @staticmethod
    def from_recording(
        recording_path, filepath, subject_id, species="human"
    ) -> NMFData:
        with h5.File(recording_path) as file:
            return NMFData(
                filepath=filepath,
                subject_id=subject_id,
                species=species,
                start_timestamp=read_start_timestamp(file),
                duration=read_recording_duration(file),
                utility_freq=read_utility_freq(file),
            )

    @staticmethod
    def subject_id_from_filepath(filepath):
        filename = os.path.basename(filepath)
        return re.match(r"[a-zA-Z]+\d+", filename)[0]

    @property
    def subject_id(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[SUBJECT_ID_LABEL]

    @subject_id.setter
    def subject_id(self, value):
        with h5.File(self._file_path, "r+") as file:
            file[META_GROUP].attrs[SUBJECT_ID_LABEL] = value

    @property
    def species(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[SPECIES_LABEL]

    @species.setter
    def species(self, value):
        with h5.File(self._file_path, "r+") as file:
            file[META_GROUP].attrs[SPECIES_LABEL] = value

    @property
    def start_timestamp(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[START_TIMESTAMP_LABEL]

    @start_timestamp.setter
    def start_timestamp(self, value):
        with h5.File(self._file_path, "r+") as file:
            file[META_GROUP].attrs[START_TIMESTAMP_LABEL] = value

    @property
    def duration(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[DURATION_LABEL]

    @duration.setter
    def duration(self, value):
        with h5.File(self._file_path, "r+") as file:
            file[META_GROUP].attrs[DURATION_LABEL] = value

    @property
    def utility_freq(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[UTILITY_FREQ_LABEL]

    @utility_freq.setter
    def utility_freq(self, value):
        with h5.File(self._file_path, "r+") as file:
            file[META_GROUP].attrs[UTILITY_FREQ_LABEL] = value

    def _create_file(self):
        with h5.File(self._file_path, "x") as file:
            file.create_group(NMF_GROUP)
            meta = file.create_group(META_GROUP)
            meta.attrs[CREATION_DATE_LABEL] = datetime.now().strftime("%Y-%m-%d")

    def _update_dset(self, path: str, data: np.ndarray):
        dtype = data.dtype
        # check if dtype is unicode; if so, use dtype object to conform to h5py
        if "U" in str(dtype):
            dtype = h5.string_dtype()
        with h5.File(self._file_path, "r+") as file:
            dset = file.require_dataset(
                name=path, shape=data.shape, dtype=dtype, exact=True
            )
            dset[()] = data[()]

    def list_feature_matrices(self):
        with h5.File(self._file_path, "r") as file:
            return list(file[NMF_GROUP].keys())

    def set_feature_matrix(
        self,
        feature_matrix_name: str,
        feature_matrix: np.ndarray,
        feature_names: list,
        feature_units: list,
        sfreq: int,
        processing: str = "",
    ):
        grp_path = os.path.join(NMF_GROUP, feature_matrix_name)
        self._update_dset(os.path.join(grp_path, FEATURE_MATRIX_LABEL), feature_matrix)
        self._update_dset(
            os.path.join(grp_path, FEATURE_NAMES_LABEL), np.array(feature_names)
        )
        self._update_dset(
            os.path.join(grp_path, FEATURE_UNITS_LABEL), np.array(feature_units)
        )
        with h5.File(self._file_path, "r+") as file:
            grp = file[grp_path]
            grp.attrs[SFREQ_LABEL] = sfreq
            grp.attrs[PROCESSING_LABEL] = processing

    def set_nmf(
        self,
        w,
        h,
        feature_matrix_name: str,
        model: str,
        rank: int,
        consensus_matrix: np.ndarray | None = None,
        metrics: pd.DataFrame | None = None,
        parameters: str | None = None,
    ):
        path = os.path.join(NMF_GROUP, feature_matrix_name, self.rank_str(rank), model)
        self._update_dset(os.path.join(path, W_LABEL), w)
        self._update_dset(os.path.join(path, H_LABEL), h)
        with h5.File(self._file_path, "r+") as file:
            if parameters:
                file[path].attrs[PARAMETERS_LABEL] = parameters

    def set_consesus_matrix(
        self,
        feature_matrix_name: str,
        model: str,
        rank: int,
        consensus_matrix: np.array,
    ):
        path = os.path.join(
            NMF_GROUP,
            feature_matrix_name,
            self.rank_str(rank),
            model,
            CONSENSUS_MATRIX_LABEL,
        )
        self._update_dset(path, consensus_matrix)

    def list_ranks(self, feature_matrix_name: str) -> list:
        with h5.File(self._file_path, "r") as file:
            keys = file[os.path.join(NMF_GROUP, feature_matrix_name)].keys()
            ranks = [rank for rank in keys if "rank" in rank]
            return ranks

    def list_models(self, feature_matrix_name: str, rank: str | int) -> list:
        if isinstance(rank, int):
            rank = rank_str(rank)
        path = os.path.join(NMF_GROUP, feature_matrix_name, rank)
        with h5.File(self._file_path, "r") as file:
            return list(file[path].keys())

    def channel_names(self, feature_matrix_name: str) -> list:
        fnames_path = os.path.join(NMF_GROUP, feature_matrix_name, FEATURE_NAMES_LABEL)
        with h5.File(self._file_path, "r") as file:
            return [bytes.decode(name) for name in file[fnames_path][()]]

    def feature_matrix(self, feature_matrix_name: str) -> np.ndarray:
        fm_path = os.path.join(NMF_GROUP, feature_matrix_name)
        with h5.File(self._file_path, "r") as file:
            return file[os.path.join(fm_path, FEATURE_MATRIX_LABEL)][()]

    def sfreq(self, feature_matrix_name: str) -> int:
        fm_path = os.path.join(NMF_GROUP, feature_matrix_name)
        with h5.File(self._file_path, "r") as file:
            return file[fm_path].attrs[SFREQ_LABEL]

    def nmf(
        self, feature_matrix_name: str, rank: str, model: str
    ) -> tuple[np.ndarray, np.np.ndarray]:
        model_path = os.path.join(NMF_GROUP, feature_matrix_name, rank, model)
        with h5.File(self._file_path, "r") as file:
            w = file[os.path.join(model_path, W_LABEL)][()]
            h = file[os.path.join(model_path, H_LABEL)][()]
            return w, h

    def rank_str(self, rank: int) -> str:
        return f"rank_{rank:02}"
