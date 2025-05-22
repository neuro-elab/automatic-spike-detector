from __future__ import annotations

import h5py as h5
import numpy as np
import os


class H5Directory:
    def __init__(self, name: str, parent: H5FileEntity | None) -> None:
        self.name = name
        self._parent = parent

        if self._parent:
            self._filepath = self._parent._filepath

        with h5.File(self._filepath, "r+") as file:
            file.require_group(self.path())

    name: str
    _parent: H5FileEntity | None
    _filepath: str

    def path(self):
        if self._parent:
            return os.path.join(self._parent.path(), self.name)
        return self.name

    def attributes(self):
        with h5.File(self._filepath, "r") as file:
            return list(file[self.path()].attrs.keys())

    def children(self):
        with h5.File(self._filepath, "r") as file:
            return list(file[self.path()].keys())

    def write_attr(self, name: str, value):
        with h5.File(self._filepath, "r+") as file:
            file[self.path()].attrs[name] = value

    def load_attr(self, name: str):
        with h5.File(self._filepath, "r") as file:
            return file[self.path()].attrs[name]

    def write_dset(self, name: str, values: np.ndarray):
        dtype = values.dtype
        if "U" in str(dtype):
            dtype = h5.string_dtype()

        with h5.File(self._filepath, "r+") as file:
            path = os.path.join(self.path(), name)
            dset = file.require_dataset(name=path, shape=values.shape, dtype=dtype)
            dset[()] = values[()]

    def load_dset(self, name, data_range: Tuple | None = None):
        with h5.File(self._filepath, "r") as file:
            path = os.path.join(self.path(), name)
            if data_range:
                start, end = data_range
                return file[path][start:end]
            return file[path][()]

    def has_dset(self, name: str):
        if name in self.children():
            with h5.File(self._filepath, "r") as file:
                return isinstance(file[self.path()][name], h5.Dataset)
        return False


class NMFMetrics(H5Directory):
    def write_metric(self, name, contents) -> None:
        if not hasattr(contents, "__len__"):
            self.write_attr(name, contents)
            return

        if not isinstance(contents, np.ndarray):
            contents = np.array(contents)

        self.write_dset(name, contents)

    def list_metrics(self):
        children = self.children()
        attrs = self.attributes()
        return children + attrs

    def load_metric(self, name):
        if name in self.children():
            return self.load_dset(name)
        if name in self.attributes():
            return self.load_attr(name)
        else:
            print(f"No metric found with name {name}")
            return -1


class NMFModel(H5Directory):
    _w_label = "w"
    _h_label = "h"
    _parameters_label = "parameters"
    _metrics_label = "metrics"

    @property
    def parameters(self):
        return self.load_attr(self._parameters_label)

    @parameters.setter
    def parameters(self, parameters: str) -> None:
        self.write_attr(self._parameters_label, parameters)

    @property
    def h(self) -> np.ndarray:
        return self.load_dset(self._h_label)

    @h.setter
    def h(self, h: np.ndarray) -> None:
        self.write_dset(self._h_label, h)

    @property
    def w(self) -> np.ndarray:
        return self.load_dset(self._w_label)

    @w.setter
    def w(self, w: np.ndarray) -> None:
        self.write_dset(self._w_label, w)

    def metrics(self) -> NMFMetrics:
        return NMFMetrics(self._metrics_label, self)


class RankGroup(H5Directory):
    def models(self) -> list:
        return [self.model(child) for child in self.children()]

    def model(self, name) -> NMFModel:
        return NMFModel(name, self)

    @classmethod
    def rank_from_value(cls, value: int):
        return f"rank_{value:02}"


class FeatureMatrixGroup(H5Directory):
    _feature_matrix_label = "feature_matrix"
    _feature_names_label = "feature_names"
    _feature_units_label = "feature_units"
    _sfreq_label = "sfreq"
    _processing_label = "processing"

    @property
    def feature_matrix(self) -> np.ndarray:
        return self.load_dset(self._feature_matrix_label)

    @feature_matrix.setter
    def feature_matrix(self, feature_matrix: np.ndarray) -> None:
        self.write_dset(self._feature_matrix_label, feature_matrix)

    @property
    def feature_names(self) -> np.ndarray:
        return [
            bytes.decode(name) for name in self.load_dset(self._feature_names_label)
        ]

    @feature_names.setter
    def feature_names(self, feature_names: list) -> None:
        self.write_dset(self._feature_names_label, np.array(feature_names))

    @property
    def feature_units(self) -> np.ndarray:
        return [
            bytes.decode(name) for name in self.load_dset(self._feature_units_label)
        ]

    @feature_units.setter
    def feature_units(self, feature_units: list) -> None:
        self.write_dset(self._feature_units_label, np.array(feature_units))

    @property
    def sfreq(self) -> int:
        return self.load_attr(self._sfreq_label)

    @sfreq.setter
    def sfreq(self, sfreq: int) -> None:
        self.write_attr(self._sfreq_label, sfreq)

    @property
    def processing(self) -> int:
        return self.load_attr(self._processing_label)

    @processing.setter
    def processing(self, processing: int) -> None:
        self.write_attr(self._processing_label, processing)

    def models_by_rank(self, rank: int):
        return RankGroup(RankGroup.rank_from_value(rank), self).list_models()

    def by_rank(self, rank: str):
        return RankGroup(rank, self)

    def by_value(self, value: int):
        return RankGroup(RankGroup.rank_from_value(value), self)

    def ranks(self):
        return [RankGroup(child, self) for child in self.children() if "rank" in child]


class MetaGroup(H5Directory):
    _creation_date_label = "creation_date"
    _subject_id_label = "subject_id"
    _species_label = "species"
    _start_timestamp_label = "start_timestamp"
    _duration_label = "duration"
    _utility_freq_label = "utility_freq"

    @property
    def creation_date(self) -> str:
        return self.load_attr(self._creation_date_label)

    @creation_date.setter
    def creation_date(self, creation_date: str) -> None:
        self.write_attr(self._creation_date_label, creation_date)

    @property
    def subject_id(self) -> str:
        return self.load_attr(self._subject_id_label)

    @subject_id.setter
    def subject_id(self, subject_id: str) -> None:
        self.write_attr(self._subject_id_label, subject_id)

    @property
    def species(self) -> str:
        return self.load_attr(self._species_label)

    @species.setter
    def species(self, species: str) -> None:
        self.write_attr(self._species_label, species)

    @property
    def start_timestamp(self) -> int:
        return self.load_attr(self._start_timestamp_label)

    @start_timestamp.setter
    def start_timestamp(self, start_timestamp: int) -> None:
        self.write_attr(self._start_timestamp_label, start_timestamp)

    @property
    def duration(self) -> int:
        return self.load_attr(self._duration_label)

    @duration.setter
    def duration(self, duration: int) -> None:
        self.write_attr(self._duration_label, duration)

    @property
    def utility_freq(self) -> int:
        return self.load_attr(self._utility_freq_label)

    @utility_freq.setter
    def utility_freq(self, utility_freq: int) -> None:
        self.write_attr(self._utility_freq_label, utility_freq)


class NMFDataset(H5Directory):
    _meta_label = "meta"

    def meta(self) -> MetaGroup:
        return MetaGroup(self._meta_label, self)

    def feature_matrices(self) -> list:
        return [
            self.feature_matrix(child)
            for child in self.children()
            if "meta" not in child
        ]

    def feature_matrix(self, name) -> FeatureMatrixGroup:
        return FeatureMatrixGroup(name, self)


class NMFRoot(H5Directory):
    _root = "nmf"

    def __init__(self, filepath) -> None:
        self._filepath = filepath
        with h5.File(self._filepath, "a") as file:
            file.require_group(self._root)

        super().__init__(self._root, None)

    def datasets(self) -> list:
        return [self.dataset(child) for child in self.children()]

    def dataset(self, name: str):
        return NMFDataset(name, self)
