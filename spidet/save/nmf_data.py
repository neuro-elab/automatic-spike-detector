from __future__ import annotations

import h5py as h5
import numpy as np
import os
import re
import datetime

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


class NMFData:
    def __init__(
        self,
        filepath,
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
    def from_recording(recording_path, filepath, species="human") -> NMFData:
        with h5.File(recording_path) as file:
            return NMFData(
                filepath=filepath,
                subject_id=NMFData.subject_id_from_filepath(recording_path),
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
        with h5.File(self.path, "r+") as file:
            file[META_GROUP].attrs[SUBJECT_ID_LABEL] = value

    @property
    def species(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[SPECIES_LABEL]

    @species.setter
    def species(self, value):
        with h5.File(self.path, "r+") as file:
            file[META_GROUP].attrs[SPECIES_LABEL] = value

    @property
    def start_timestamp(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[START_TIMESTAMP_LABEL]

    @start_timestamp.setter
    def start_timestamp(self, value):
        with h5.File(self.path, "r+") as file:
            file[META_GROUP].attrs[START_TIMESTAMP_LABEL] = value

    @property
    def duration(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[DURATION_LABEL]

    @duration.setter
    def duration(self, value):
        with h5.File(self.path, "r+") as file:
            file[META_GROUP].attrs[DURATION_LABEL] = value

    @property
    def utility_freq(self):
        with h5.File(self._file_path, "r") as file:
            return file[META_GROUP].attrs[UTILITY_FREQ_LABEL]

    @utility_freq.setter
    def utility_freq(self, value):
        with h5.File(self.path, "r+") as file:
            file[META_GROUP].attrs[UTILITY_FREQ_LABEL] = value

    def _create_file(self):
        with h5.File(self._file_path, "x") as file:
            meta = file.create_group(META_GROUP)
            meta.attrs[CREATION_DATE_LABEL] = datetime.now().strftime("%Y-%m-%d")
            meta.attrs[SUBJECT_ID_LABEL] = self.subject_id
            meta.attrs[SPECIES_LABEL] = self.species
            meta.attrs[START_TIMESTAMP_LABEL] = self.start_timestamp
            meta.attrs[DURATION_LABEL] = self.duration
            meta.attrs[UTILITY_FREQ_LABEL] = self.utility_freq

            file.create_group(NMF_GROUP)

    def list_feature_matrices(self):
        with h5.File(self.path, "r") as file:
            return file[NMF_GROUP].keys()

    def set_feature_matrix(
        self,
        feature_matrix_name: str,
        feature_matrix: np.ndarray,
        feature_names: list,
        feature_units: list,
        sfreq: int,
        processing: str = "",
    ):
        with h5.File(self._file_path, "r+") as file:
            grp = file[os.path.join(NMF_GROUP, feature_matrix_name)]
            grp[FEATURE_MATRIX_LABEL] = feature_matrix
            grp[FEATURE_NAMES_LABEL] = feature_names
            grp[FEATURE_UNITS_LABEL] = feature_units

            grp.attrs[SFREQ_LABEL] = sfreq
            grp.attrs[PROCESSING_LABEL] = processing

    def set_nmf(
        self,
        w,
        h,
        feature_matrix_name: str,
        model: str,
        rank: int,
        parameters: str | None,
    ):
        with h5.File(self._file_path, "r+") as file:
            path = os.path.join(
                NMF_GROUP, feature_matrix_name, self.rank_str(rank), model
            )
            grp = file[path]
            grp[W_LABEL] = w
            grp[H_LABEL] = h
            if parameters:
                grp[PARAMETERS_LABEL] = parameters

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
            return file[path].keys()

    def rank_str(rank: int) -> str:
        return f"rank_{rank:02}"
