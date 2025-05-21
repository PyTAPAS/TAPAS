# -*- coding: utf-8 -*-
"""
Copyright © 2025, Philipp Frech

This file is part of TAPAS.

    TAPAS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TAPAS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TAPAS.  If not, see <http://www.gnu.org/licenses/>.
"""

from PyQt6.QtCore import QObject, pyqtSignal
import logging
from numpy.typing import NDArray
from pathlib import Path


logger = logging.getLogger(__name__)


class Model(QObject):
    '''
    Base MVC model providing common data properties and change signals.
    '''

    def __init__(self, path=None, rawdata_before=None, rawdata_after=None):
        super().__init__()
        self._path = path
        self._data_before = rawdata_before
        self._data_after = rawdata_after

    path_changed = pyqtSignal(str)
    rawdata_changed = pyqtSignal(str)

    data_changed = pyqtSignal()
    metadata_changed = pyqtSignal()
    status_signal = pyqtSignal(str, str)
    local_fit_changed = pyqtSignal()
    global_fit_changed = pyqtSignal()


class AbsModel(Model):
    '''
    MVC model for handling absorption data files and raw spectra.

    This model stores file paths and raw absorption data arrays taken
    before and after a processing step, and emits signals when these
    properties change so that views and controllers can update accordingly.

    Signals
    -------
    path_changed(str)
        Emitted when the `path` property is set, with argument "abs".
    rawdata_changed(str)
        Emitted when either `rawdata_before` or `rawdata_after` is set,
        with argument "abs".

    Properties
    ----------
    path : Path or list[Path]
        Filesystem path(s) to the absorption data. Setting this property
        updates `_path` and emits `path_changed("abs")`.
    rawdata_before : NDArray
        The raw absorption spectrum before processing. Setting this
        property updates `_data_before` and emits `rawdata_changed("abs")`.
    rawdata_after : NDArray
        The raw absorption spectrum after processing. Setting this
        property updates `_data_after` and emits `rawdata_changed("abs")`.
    '''

    def __init__(self, ):
        super().__init__()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: Path | list[Path]):
        self._path = value
        self.path_changed.emit("abs")

    @property
    def rawdata_before(self):
        return self._data_before

    @rawdata_before.setter
    def rawdata_before(self, value: NDArray):
        self._data_before = value

        self.rawdata_changed.emit("abs")

    @property
    def rawdata_after(self):
        return self._data_after

    @rawdata_after.setter
    def rawdata_after(self, value: NDArray):
        self._data_after = value


class EmModel(Model):
    '''
    MVC model for handling emission data files and raw spectra.

    This model stores file paths and raw emission data arrays taken
    before and after a processing step, and emits signals when these
    properties change so that views and controllers can update accordingly.

    Signals
    -------
    path_changed(str)
        Emitted when the `path` property is set, with argument "em".
    rawdata_changed(str)
        Emitted when either `rawdata_before` or `rawdata_after` is set,
        with argument "em".

    Properties
    ----------
    path : Path or list[Path]
        Filesystem path(s) to the emission data. Setting this property
        updates `_path` and emits `path_changed("em")`.
    rawdata_before : NDArray
        The raw emission spectrum before processing. Setting this
        property updates `_data_before` and emits `rawdata_changed("em")`.
    rawdata_after : NDArray
        The raw emission spectrum after processing. Setting this
        property updates `_data_after` and emits `rawdata_changed("em")`.
    '''

    def __init__(self, ):
        super().__init__()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: Path | list[Path]):
        self._path = value
        self.path_changed.emit("em")

    @property
    def rawdata_before(self):
        return self._data_before

    @rawdata_before.setter
    def rawdata_before(self, value: NDArray):
        self._data_before = value

        self.rawdata_changed.emit("em")

    @property
    def rawdata_after(self):
        return self._data_after

    @rawdata_after.setter
    def rawdata_after(self, value: NDArray):
        self._data_after = value


class TAModel(Model):
    '''
    MVC model for transient‐absorption (TA) experiments, including raw data, solvent data, and metadata.

    This model holds:
      - File paths for TA and solvent datasets.
      - Raw TA and solvent data dictionaries.
      - Experiment metadata (sample, excitation settings, etc.).
    It emits signals when these properties change so that views/controllers can react.

    Signals
    -------
    path_changed(str)
        Emitted when `path` (TA data) or `solvent_path` is set.
        - Argument "ta" for TA data path updates.
        - Argument "solvent" for solvent data path updates.
    rawdata_changed(str)
        Emitted when `rawdata` (TA) or `solvent` data is set.
        - Argument "ta" for TA raw data updates.
        - Argument "solvent" for solvent raw data updates.
    metadata_changed()
        Emitted when `metadata` is set or updated.

    Parameters
    ----------
    rawdata : dict, optional
        Initial raw TA data. Defaults to empty dict.
    solvent : dict, optional
        Initial solvent data. Defaults to empty dict.
    metadata : dict, optional
        Initial metadata fields. Expected keys include:
          'experiment', 'sample', 'excitation wavelength', 'excitation intensity',
          'solvent', 'notes'
        Defaults to all-empty strings.

    Properties
    ----------
    path : Path or list[Path]
        Filesystem path(s) to TA data. Setting emits `path_changed("ta")`.
    solvent_path : Path or list[Path]
        Filesystem path(s) to solvent data. Setting emits `path_changed("solvent")`.
    metadata : dict
        Experiment metadata. Can set entire dict or a (key, value) pair to update one entry;
        setting emits `metadata_changed()`.
    rawdata : dict
        Raw TA data dictionary. Setting emits `rawdata_changed("ta")`.
    solvent : dict
        Raw solvent data dictionary. Setting emits `rawdata_changed("solvent")`.
    '''

    def __init__(self, rawdata: dict = None, solvent: dict = None,
                 metadata: dict = {'experiment': '', 'sample': '', 'excitation wavelength': '',
                                   'excitation intensity': '', 'solvent': '', 'notes': '', }):
        super().__init__()
        self._rawdata = rawdata if rawdata is not None else {}
        self._metadata = metadata
        self._solvent = solvent if solvent is not None else {}

# --------------------------------------

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: Path | list[Path]):
        self._path = value  # list[Pathlib]
        self.path_changed.emit("ta")

    @property
    def solvent_path(self):
        return self._solvent_path

    @solvent_path.setter
    def solvent_path(self, value: Path | list[Path]):
        self._solvent_path = value  # list[Pathlib]
        self.path_changed.emit("solvent")
# ----------------------------------------

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, val: dict | tuple):

        if isinstance(val, dict):
            self._metadata = val

        else:
            try:
                key, dictionary = val
                self._metadata[key] = dictionary

            except ValueError:
                self._metadata = {}
        self.metadata_changed.emit()

# ----------------------------------------

    @property
    def rawdata(self):
        return self._rawdata

    @rawdata.setter
    def rawdata(self, val: dict):
        if isinstance(val, dict):
            self._rawdata = val
            self.rawdata_changed.emit("ta")
        elif val is None:
            self._rawdata = {}
            self.rawdata_changed.emit("ta")

    @property
    def solvent(self):
        return self._solvent

    @solvent.setter
    def solvent(self, val: dict):
        if isinstance(val, dict):
            self._solvent = val
            self.rawdata_changed.emit("solvent")
        elif val is None:
            self._solvent = {}
            self.rawdata_changed.emit("solvent")


class TAModel_ds(Model):
    '''
    Model for processed transient-absorption dataset, including data grids, fit results, and metadata.

    This MVC model holds:
      - dsX, dsY: 1D coordinate arrays for the two dimensions of the dataset (x: wavelength, y: delay time).
      - dsZ:    2D data array of shape (len(dsY), len(dsX)).
      - local_fit:  dict mapping labels to per-interval (local) fit results.
      - global_fit: dict mapping labels to global fit results across the dataset.
      - metadata:   dict of auxiliary information (e.g., preprocessing steps, chirp coefficients).

    Signals
    -------
    data_changed()
        Emitted whenever `dsZ` is updated, and clears existing fit results.
    local_fit_changed()
        Emitted whenever `local_fit` is set or modified.
    global_fit_changed()
        Emitted whenever `global_fit` is set or modified.

    Parameters
    ----------
    dsX : NDArray, optional
        X-coordinate grid (wavelengths).
    dsY : NDArray, optional
        Y-coordinate grid (delay times).
    dsZ : NDArray, optional
        2D data array.
    local_fit : dict, optional
        Initial dictionary of local fit results.
    global_fit : dict, optional
        Initial dictionary of global fit results.
    metadata : dict, optional
        Initial metadata entries (e.g., experiment parameters or preprocessing history).
    '''

    def __init__(self, dsX: NDArray = None, dsY: NDArray = None, dsZ: NDArray = None,
                 local_fit: dict = None, global_fit: dict = None, metadata: dict = None):
        super().__init__()

        self._dsX = dsX
        self._dsY = dsY
        self._dsZ = dsZ
        self._local_fit = local_fit if local_fit is not None else {}
        self._global_fit = global_fit if global_fit is not None else {}
        self._metadata = metadata if metadata is not None else {}

# ---------------------------------------------------------

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        self._metadata = value

    def set_chirp(self, array: NDArray):
        """Update the metadata dict with chrip_coeffs """
        self._metadata['chirp_coeffs'] = array

    def set_background_surf(self, array: NDArray):
        """Update the metadata dict with background surf """
        self._metadata['substracted surface'] = array

    def update_metadata(self, value: str | NDArray | list):
        """Update the metadata dict with a new key 'Preprocessing N' where N is an increasing number."""
        max_index = 0
        for key in self._metadata:
            if key.startswith("Preprocessing"):
                try:
                    # Extract the number from the key, e.g., "Preprocessing 2" -> 2
                    index = int(key.split(" ")[1])
                    if index > max_index:
                        max_index = index
                except (IndexError, ValueError):
                    continue
        new_key = f"Preprocessing {max_index + 1}"
        self._metadata[new_key] = value

    def clear_metadata(self):
        """Reset metadata to an empty dictionary."""
        self._metadata = {}

    @property
    def dsX(self):
        return self._dsX

    @dsX.setter
    def dsX(self, value):
        self._dsX = value

    @property
    def dsY(self):
        return self._dsY

    @dsY.setter
    def dsY(self, value):
        self._dsY = value

    @property
    def dsZ(self):
        return self._dsZ

    @dsZ.setter
    def dsZ(self, value: NDArray):
        self._dsZ = value
        self.data_changed.emit()
        self._local_fit = {}
        self._global_fit = {}

    @property
    def local_fit(self):
        return self._local_fit

    @local_fit.setter
    def local_fit(self, val: dict):
        if isinstance(val, dict):
            self._local_fit = val
            self.local_fit_changed.emit()

    def update_local_fit(self, key: str, value: dict):
        """Add or replace a local fit entry and emit `local_fit_changed()`."""
        self._local_fit[key] = value
        self.local_fit_changed.emit()

    def update_local_fit_emcee(self, key: str, value: dict):
        """Update the 'emcee' sub-dictionary of a local fit and emit `local_fit_changed()`."""
        self._local_fit[key]['emcee'] = value
        self.local_fit_changed.emit()

    def del_local_fit_key(self, key: str):
        """Delete a local fit entry by key and emit `local_fit_changed()`."""
        if key in self._local_fit:
            del self._local_fit[key]
            self.local_fit_changed.emit()

    @property
    def global_fit(self):
        return self._global_fit

    @global_fit.setter
    def global_fit(self, val: dict):
        if isinstance(val, dict):
            self._global_fit = val
            self.global_fit_changed.emit()

    def update_global_fit(self, key: str, value: dict):
        """Add or replace a global fit entry and emit `global_fit_changed()`."""
        self._global_fit[key] = value
        self.global_fit_changed.emit()

    def update_global_fit_emcee(self, key: str, value: dict):
        """Update the 'emcee' sub-dictionary of a global fit and emit `global_fit_changed()`."""
        self._global_fit[key]['emcee'] = value
        self.global_fit_changed.emit()

    def del_global_fit_key(self, key: str):
        """Delete a global fit entry by key and emit `global_fit_changed()`."""
        if key in self._global_fit:
            del self._global_fit[key]
            self.global_fit_changed.emit()
