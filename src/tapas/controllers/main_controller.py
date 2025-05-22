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

import numpy as np
import logging
import logging.config
import h5py
from ..configurations import messages as msg, exceptions as exc
from pathlib import Path

import json
from lmfit import Parameters
from PyQt6.QtCore import QSignalBlocker


logger = logging.getLogger(__name__)


class MainController(QObject):

    def __init__(self, abs_model, em_model, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()

        self.abs_model = abs_model
        self.em_model = em_model
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3

        # -------- create metadata dict for HDF attributes -----------------------------------------
        self.meta_dict = {
            'wavelength': {'units': 'm', 'description': 'spectral wavelength'},
            'delay':  {'units': 's', 'description': 'time interval between excitation and measurement'},
            'delA': {'units': 'mOD', 'description': 'differential absorbance (A_exc - A_groundstate)',
                     'dimensions (row x column)': 'delay x wavelength'},
            'after':  {'units': '1st column: m, 2nd column: OD / a.u', 'description':
                       'steady-state measurement after the TA experiment', 'dimensions': '1st column: wavelength, 2nd column: absorbance / PL intensity'},
            'before':  {'units': '1st column: m, 2nd column: OD / a.u', 'description':
                        'steady-state measurement before the TA experiment', 'dimensions': '1st column: wavelength, 2nd column: absorbance / PL intensity'},
            'theta': {'units': 's', 'description': 'lifetimes of the fit', 'dimensions': 't0, IRF, t1, t2, ..., t_components'},
            'residuals': {'units': 'mOD', 'description': 'weighted residuals ( = (delA - delA_cal) * sqrt(weights))', 'dimensions (row x column)': 'delay x wavelength'},
            'Amp':  {'units': 'mOD', 'description': '1D representation of the EAS/DAS. Amp * conc = delA_calc',
                     'dimensions': 'EAS1/DAS1, EAS2/DAS2 ... see \'components\' information in the folder attribute '},
            'conc':  {'units': '-', 'description': 'fitted time-dependent concentration of each component',
                      'dimensions': 'row: concentration at each time point. column: EAS1/DAS1, EAS1/DAS2 ... see \'components\' information in the folder attribute '},
            'delA_calc': {'units': 'mOD', 'description': 'calculated differential absorbance (A_exc - A_groundstate) from the fit parameters',
                          'dimensions (row x column)': 'delay x wavelength'},
            'opt_params':  {'description': 'json representative of the lmfit Parameters object. Holds the optimized parameters. See theta'},
            'unweighted_opt_params':  {'description': 'json representative of the lmfit Parameters object. Holds the optimized parameters and the unweighted CIs. See theta'},
            'params':  {'description': 'json representative of the lmfit Parameters object. Holds the parameters used for emcee'},
            'output':  {'description': 'summary of the fit results'},
            'corr_matrix':  {'description': 'Pearson correlation matrix retreived from the covariance matrix',
                             'dimensions': 'parameters x parameters: t0, IRF, t1, t2, ..., t_components'},
            'unweighted_corr_matrix':  {'description': 'Pearson correlation matrix retreived from the unweighted_covariance matrix',
                                        'dimensions': 'parameters x parameters: t0, IRF, t1, t2, ..., t_components'},
            'covariance_matrix':  {'description': 'parameter covariance matrix from the nonlinear least-squares model (JT@J)-1',
                                   'dimensions': 'parameters x parameters: t0, IRF, t1, t2, ..., t_components'},
            'unweighted_covariance_matrix':  {'description': 'parameter covariance matrix from the nonlinear least-squares model (JT@J)-1 of the unweighhted residuals',
                                              'dimensions': 'parameters x parameters: t0, IRF, t1, t2, ..., t_components'},
            'EAS':  {'units': 'mOD', 'description': 'Evolution-associated (difference) spectra resulting from the sequential fit model',
                     'dimensions': 'row: delA at each wavelength point. column: EAS1, EAS2 ... see \'components\' information in the folder attribute'},

            'DAS':  {'units': 'mOD', 'description': 'Decay-associated (difference) spectra resulting from the parallel fit model',
                     'dimensions': 'row: delA at each wavelength point. column: DAS1, DAS2 ... see \'components\' information in the folder attribute'},

            'SAS':  {'units': 'mOD', 'description': 'Species-associated (difference) spectra resulting from the target fit model',
                     'dimensions': 'row: delA at each wavelength point. column: SAS1, SAS2 ... see \'components\' information in the folder attribute'},
            'weight_vector':  {'units': 'relative weights', 'description':
                               'array of per‐wavelength‐point scaling factors by which each residual is multiplied',
                               'dimensions': 'wavelength vector'},
            'flatchain':  {'units': 's', 'description': 'full set of posterior samples. each entry is a sampled value of one model parameter',
                           'dimensions': 'row: num steps x num walkers, column: t0, IRF, t1, t2, ..., ln(sigma)'},

        }

    status_signal = pyqtSignal(str, str)

    def write_dict_to_hdf(self, group: 'hdf group', data_dict: dict) -> None:
        '''
        Recursively save a Python dictionary into an HDF5 group.

        Parameters
        ----------
        group : h5py.Group or h5py.File
            The HDF5 group (or file) into which data will be written.
        data_dict : dict[str, Any]
            The dictionary of data to save. Values may be nested dicts,
            lists/tuples, strings, numeric types, or custom objects
            with a `.dumps()` method.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        str_dt = h5py.string_dtype(encoding='utf-8')
        for key, value in data_dict.items():
            if type(value).__name__ == 'Parameters':
                # Serialize using the custom dumps() method.
                ds = group.create_dataset(key, data=value.dumps())
            elif isinstance(value, dict):
                # If key is marked to be saved as JSON, dump it as a string.
                if key == 'meta':
                    for meta_key, meta_value in value.items():

                        group.attrs[meta_key] = meta_value
                else:
                    # Otherwise, create a subgroup and recursively save.
                    subgroup = group.create_group(key)
                    self.write_dict_to_hdf(subgroup, value)
            elif isinstance(value, (list, tuple)):
                try:
                    # Try converting to a numpy array
                    ds = group.create_dataset(key, data=np.array(value))

                except Exception:
                    # If that fails, store as a JSON string
                    group.create_dataset(key, data=json.dumps(value))
            elif isinstance(value, str):
                # explicitly use our UTF-8 dtype
                ds = group.create_dataset(key,
                                          shape=(),
                                          dtype=str_dt,
                                          data=value)
            else:
                # Try to store the value directly; if it fails, fallback to JSON
                try:

                    ds = group.create_dataset(key, data=value)
                except TypeError:
                    ds = group.create_dataset(key, data=json.dumps(value))

            if key in self.meta_dict:
                ds.attrs.update(self.meta_dict[key])

    def read_dict_from_hdf(self, group: 'hdf group') -> dict:
        '''
        Recursively load an HDF5 group (and its subgroups) into a nested Python dict.

        Each subgroup becomes a nested dict, and each dataset becomes a value.
         attributes are collected under the `'meta'` key.

        Parameters
        ----------
        group : h5py.Group or h5py.File
            The HDF5 group (or file) to read from.

        Returns
        -------
        dict
            A dictionary where:
          - Keys matching subgroup names map to nested dicts.
          - Dataset names map to their array or scalar values.
          - Any group attributes are gathered in `result['meta']`

        '''
        result = {}

        # if there are any attributes, add them as a 'meta' dictionary.
        if group.attrs:
            result['meta'] = {attr: group.attrs[attr] for attr in group.attrs}

        # Then iterate over keys in the group.
        for key, item in group.items():
            if isinstance(item, type(group)):  # if item is a subgroup
                result[key] = self.read_dict_from_hdf(item)
            else:
                # Retrieve the dataset's value.
                value = item[()]

                if key == 'opt_params' or key == 'params':
                    # Assuming Parameters().loads expects a decoded string.
                    value = Parameters().loads(value.decode())
                elif isinstance(value, (bytes, bytearray, np.bytes_)):
                    value = value.decode('utf-8')
                result[key] = value

        return result

    def save_project(self, project_path: Path, gui_config: dict) -> None:
        '''
        Serialize the current application state to an HDF5 “project” file.

        Creates or overwrites `project_path` and writes:
          - Top-level attribute `gui_standard_values` containing JSON of GUI settings.
          - An “Absorption Data” group (if available) with datasets “before” and/or “after”.
          - An “Emission Data” group (if available) with datasets “before” and/or “after”.
          - A “TA Data” group containing:
              • TA metadata as group attributes.
              • A “raw Data” subgroup saved via `write_dict_to_hdf`.
              • A “solvent” subgroup saved via `write_dict_to_hdf`.
              • Subgroups “ds 1”, “ds 2”, “ds 3” for each processed transient‐absorption dataset,
                each with:
                  – Attributes from that dataset’s metadata.
                  – Datasets “wavelength”, “delay”, and “delA” (with metadata attrs).
                  – Optional “local fit” and “global fit” subgroups, each populated
                    recursively via `write_dict_to_hdf`.

        Parameters
        ----------
        project_path : Path
            Filesystem path where the HDF5 project file will be created.
        gui_config : dict
            Dictionary of GUI settings to store as JSON under `gui_standard_values`.

        Returns
        -------
        None
            DESCRIPTION.

        '''

        try:
            with h5py.File(project_path, "w", track_order=True,) as p:
                p.attrs['gui_standard_values'] = json.dumps(gui_config)
                t = p.create_group('TA Data', track_order=True)

                dataset_dict = {'ds 1': self.ta_model_ds1,
                                'ds 2': self.ta_model_ds2, 'ds 3': self.ta_model_ds3}

                if self.abs_model.rawdata_before is not None:
                    a = p.create_group('Absorption Data')
                    ds = a.create_dataset(
                        "before", data=self.abs_model.rawdata_before, track_order=True)
                    ds.attrs.update(self.meta_dict["before"])
                if self.abs_model.rawdata_after is not None:
                    ds = a.create_dataset(
                        "after", data=self.abs_model.rawdata_after, track_order=True)
                    ds.attrs.update(self.meta_dict["after"])
                if self.em_model.rawdata_before is not None:
                    e = p.create_group('Emission Data')
                    ds = e.create_dataset(
                        "before", data=self.em_model.rawdata_before, track_order=True)
                    ds.attrs.update(self.meta_dict["before"])
                if self.em_model.rawdata_after is not None:
                    ds = e.create_dataset(
                        "after", data=self.em_model.rawdata_after, track_order=True)
                    ds.attrs.update(self.meta_dict["after"])
                if self.ta_model.metadata is not None:
                    for k, v in self.ta_model.metadata.items():
                        t.attrs[k] = v

                if self.ta_model.rawdata is not None:
                    t_raw = t.create_group('raw Data', track_order=True)
                    self.write_dict_to_hdf(
                        group=t_raw, data_dict=self.ta_model.rawdata)
                if self.ta_model.solvent is not None:
                    solvent_group = t.create_group('solvent', track_order=True)
                    self.write_dict_to_hdf(
                        group=solvent_group, data_dict=self.ta_model.solvent)

                for ds_name, model in dataset_dict.items():

                    if model.dsX is not None:

                        ds_group = t.create_group(ds_name, track_order=True)
                        for key, value in model.metadata.items():

                            ds_group.attrs[key] = value
                        ds = ds_group.create_dataset(
                            "wavelength", data=model.dsX, track_order=True)
                        ds.attrs.update(self.meta_dict["wavelength"])
                        ds = ds_group.create_dataset(
                            "delay", data=model.dsY, track_order=True)
                        ds.attrs.update(self.meta_dict["delay"])
                        ds = ds_group.create_dataset(
                            "delA", data=model.dsZ, track_order=True)
                        ds.attrs.update(self.meta_dict["delA"])

                        if model.local_fit:
                            lf_group = ds_group.create_group(
                                'local fit', track_order=True)
                            for ukey, data_dict in model.local_fit.items():
                                lf_key_group = lf_group.create_group(ukey)
                                self.write_dict_to_hdf(lf_key_group, data_dict)

                        if model.global_fit:
                            gf_group = ds_group.create_group(
                                'global fit', track_order=True)
                            for ukey, data_dict in model.global_fit.items():
                                gf_key_group = gf_group.create_group(ukey)
                                self.write_dict_to_hdf(gf_key_group, data_dict)

            self.call_statusbar("info", msg.Status.s06)

        except BlockingIOError:
            self.call_statusbar("error", msg.Error.e04)

        except Exception:
            logger.exception("unknown exception occurred")
            self.call_statusbar("error", msg.Error.e01)

    def open_project(self, project_path: Path) -> dict | None:
        '''
        Load an HDF5 “project” file and restore application state into models.

        This method opens `project_path` for reading, temporarily blocks
        TA model signals during bulk updates, and then:

          1. Reads the top‐level `gui_standard_values` JSON attribute (if present)
             and returns it for reconfiguring the GUI.
          2. Loads “Absorption Data” and “Emission Data” groups (datasets “before”/“after”)
             into `self.abs_model.rawdata_before/after` and
             `self.em_model.rawdata_before/after`, clearing them if absent.
          3. Restores TA metadata from `TA Data` group attributes.
          4. Recursively reads `TA Data/solvent` and `TA Data/raw Data` via
             `read_dict_from_hdf` into `self.ta_model.solvent` and `.rawdata`,
             clearing them if absent.
          5. For each sub‐dataset “ds 1”, “ds 2”, “ds 3” under `TA Data`, loads:
             - `wavelength` -> `model.dsX`
             - `delay`      -> `model.dsY`
             - `delA`       -> `model.dsZ`
             - Optional `local fit` and `global fit` subgroups via `read_dict_from_hdf`
               into `model.local_fit` and `model.global_fit`.
          6. On any exception, logs the traceback and emits an error status.

        After loading, emits `data_changed` on `self.ta_model_ds1` to trigger UI refresh.

        Parameters
        ----------
        project_path : Path
            Path to the HDF5 project file to open and read.

        Returns
        -------
        dict or None
            The GUI configuration dictionary loaded from the file’s
            `gui_standard_values` attribute, or None if not present.

        '''

        new_config = None
        # block model emission during batch updates
        with QSignalBlocker(self.ta_model_ds1), QSignalBlocker(self.ta_model_ds2), QSignalBlocker(self.ta_model_ds3):
            try:
                with h5py.File(project_path, "r", ) as p:
                    if 'gui_standard_values' in p.attrs.keys():
                        new_config = json.loads(p.attrs['gui_standard_values'])
                    else:
                        new_config = None

                    if 'Absorption Data/before' in p:
                        self.abs_model.rawdata_before = p['Absorption Data/before'][()]
                        if 'Absorption Data/after' in p:
                            self.abs_model.rawdata_after = p['Absorption Data/after'][()]
                    else:  # ensures that averything is cleared even there is no raw data to override
                        self.abs_model.rawdata_before = None
                        self.abs_model.rawdata_after = None

                    if 'Emission Data/before' in p:
                        self.em_model.rawdata_before = p['Emission Data/before'][()]
                        if 'Emission Data/after' in p:
                            self.em_model.rawdata_after = p['Emission Data/after'][()]
                    else:  # ensures that averything is cleared even there is no raw data to override
                        self.em_model.rawdata_before = None
                        self.em_model.rawdata_after = None

                    for k in p['TA Data'].attrs.keys():
                        self.ta_model.metadata[k] = p['TA Data'].attrs[k]

                    if 'TA Data/solvent' in p:
                        solvent_group = p['TA Data/solvent']
                        self.ta_model.solvent = self.read_dict_from_hdf(
                            group=solvent_group)
                    if 'TA Data/raw Data' in p:
                        rawdata_group = p['TA Data/raw Data']
                        self.ta_model.rawdata = self.read_dict_from_hdf(
                            group=rawdata_group)
                    else:  # ensures that averything is cleared even there is no raw data to override
                        self.ta_model.rawdata = None

                    dataset_dict = {'ds 1': self.ta_model_ds1,
                                    'ds 2': self.ta_model_ds2, 'ds 3': self.ta_model_ds3}

                    for ds_name, model in dataset_dict.items():

                        if 'TA Data/' + ds_name + '/wavelength' in p:
                            ds_group = p['TA Data/' + ds_name]
                            metadata = {
                                key: ds_group.attrs[key] for key in ds_group.attrs}

                            model.metadata = metadata

                            model.dsX = p['TA Data/' +
                                          ds_name + '/wavelength'][()]
                        if 'TA Data/' + ds_name + '/delay' in p:
                            model.dsY = p['TA Data/' + ds_name + '/delay'][()]
                        if 'TA Data/' + ds_name + '/delA' in p:
                            model.dsZ = p['TA Data/' +
                                          ds_name + '/delA'][()]

                            if 'local fit' in ds_group:
                                local_fit_group = ds_group['local fit']
                                # list of h5py group: contains wavelength ukeys or []
                                if local_fit_group.keys():
                                    data_dict = {}
                                    for ukey in local_fit_group.keys():
                                        data_dict[ukey] = self.read_dict_from_hdf(
                                            local_fit_group[ukey])

                                    model.local_fit = data_dict

                            if 'global fit' in ds_group:
                                global_fit_group = ds_group['global fit']
                                if global_fit_group.keys():
                                    # This reads the entire structure recursively.
                                    data_dict = {}
                                    for ukey in global_fit_group.keys():
                                        # For each key (representing a unique global fit),
                                        # call the recursive function on that subgroup.
                                        data_dict[ukey] = self.read_dict_from_hdf(
                                            global_fit_group[ukey])
                                    model.global_fit = data_dict


            except Exception:
                logger.exception("unknown exception occurred")
                self.call_statusbar("error", msg.Error.e01)
        # emit changes so that the tabwidgets can prepare refresh
        self.ta_model_ds1.data_changed.emit()
        return new_config

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)
