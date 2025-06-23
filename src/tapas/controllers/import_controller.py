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
from numpy.typing import NDArray
import logging
import logging.config
from ..configurations import messages as msg
from pathlib import Path
logger = logging.getLogger(__name__)


class ImportController(QObject):

    def __init__(self, abs_model, em_model, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()
        self.abs_model = abs_model
        self.em_model = em_model
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
    status_signal = pyqtSignal(str, str)

    def get_metadata(self) -> dict:
        ''' helper that returns the metadata of the TA model '''
        return self.ta_model.metadata

    def save_metadata(self, key: str, value: str) -> None:
        ''' helper that saves dict entries to the TA model metadata dict '''
        self.ta_model.metadata[key] = value

    def get_rawdata_model(self, raw_ds: str) -> 'object':
        ''' helper that returns the model to the corresponding identifyer string '''
        if raw_ds == 'abs':
            return self.abs_model
        elif raw_ds == 'em':
            return self.em_model
        else:
            return self.ta_model

    def get_ss_data(self, raw_ds: str) -> tuple[NDArray, NDArray]:
        ''' helper that returns the steady-state abs or emission data arrays '''
        model = self.get_rawdata_model(raw_ds)
        return model.rawdata_before, model.rawdata_after

    def get_ta_data(self) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
        ''' helper that returns the TA data arrays or None, if nonexistent'''
        model = self.get_rawdata_model('ta')
        if model.rawdata:
            return model.rawdata['wavelength'], model.rawdata['delay'], model.rawdata['delA']
        else:
            return None, None, None

    def get_solvent_data(self) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
        ''' helper that returns the solvent data arrays or None, if nonexistent '''
        model = self.get_rawdata_model('ta')

        if model.solvent:
            return model.solvent['wavelength'], model.solvent['delay'], model.solvent['delA']
        else:
            return None, None, None

    def get_model_path(self, raw_ds: str) -> Path:
        ''' helper that returns the stored Path object of a given model identifyer raw_ds '''
        model = self.get_rawdata_model(raw_ds)
        return model.path

    def change_path(self, raw_ds: str, paths: list[Path] | str) -> None:
        ''' updates the model path with the given (list of) Path objects '''
        model = self.get_rawdata_model(raw_ds)

        def normalize(p):
            # ensure it’s a str, then strip whitespace *and* any enclosing quotes:
            s = str(p).strip().strip('\'"')
            return Path(s)

        if isinstance(paths, list):
            path_list = [p if isinstance(p, Path) else normalize(p) for p in paths]
    
        elif isinstance(paths, str):
            # split on commas, strip whitespace/quotes, drop empties
            raw_parts = paths.split(',')
            parts = [part.strip().strip('\'"') for part in raw_parts if part.strip().strip('\'"')]
            path_list = [Path(part) for part in parts]

        if raw_ds != 'solvent':
            model.path = path_list
        else:
            model.solvent_path = path_list

    def _data_from_txt(self, path: Path, header, delimiter,) -> tuple[NDArray, list]:
        ''' helper that reads a txt and splits it to data_array and metadata '''
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.call_statusbar("error", msg.Error.e03)
            raise
        except UnicodeDecodeError:
            self.call_statusbar("error", msg.Error.e02+msg.Error.e30)
            raise
        data_rows = []
        metadata_list = []

        # Process lines after header rows
        for line in lines[header:]:
            stripped_line = line.strip()
            if not stripped_line:  # Skip empty lines
                continue
            try:
                # Try converting the tokens into floats, ignores trailing delimiter
                numbers = [float(token)
                           for token in stripped_line.split(delimiter)if token.strip() != '']
                data_rows.append(numbers)
            except ValueError:
                # If conversion fails, treat it as metadata
                metadata_list.append(stripped_line)
            # Convert list of numeric rows into a NumPy array
        data_array = np.array(data_rows, dtype=np.float32)

        return data_array, metadata_list

    def load_data(self, raw_ds: str, delimiter_idx: int, header: int, time_unit: int | bool = False,
                  energy_unit: str | bool = False, delA_unit: bool | str = True, matrix_orientation: int = 0) -> None:
        '''
        loads the textfile(s), converts them to float32 arrays, detects metadata for TA experiments
        and dumps them to the metdata['notes']. Averages multiple TA files.
        Sorty multiple ss inputs into before and after TA experiments data.
        Uses _data_from_txt helper function.

        Parameters
        ----------
        raw_ds : str
        Identifier of the dataset to load. Must be one of:
        - `'ta'`      : Transient Absorption
        - `'solvent'`: Solvent background
        - `'abs'`     : Steady‐state absorbance
        - `'em'`      : Steady‐state emission
        delimiter_idx : int
            Index into the delimiter list `[',', ';', '\\t']` to choose how columns are separated.
        header : int
            Number of header lines to skip before the numeric data begins.
        time_unit : int, optional
            Index for delay‐time scaling (default False → 0). Interpreted as:
              0 → picoseconds
              1 → nanoseconds
              2 → microseconds
              3 → milliseconds
              4 → seconds
        energy_unit : str, optional
            Unit of the wavelength axis (default False → `'m'`):
            - `'nm'` for nanometers
            - `'m'`  for meters
        delA_unit : str, optional
            Unit of ΔA (change in absorbance) (default True → `'mOD'`):
            - `'mOD'` for milli‐optical‐density
            - `'OD'`  for optical‐density
        matrix_orientation : int, optional
            - 0     wavelength in column,
            - 1     wavelength in row, default orientation in TAPAS

        Returns
        -------
        None.

        '''
        # -------- get model, data and unit conversions --------------------------------------------
        model = self.get_rawdata_model(raw_ds)
        paths = model.path if raw_ds != 'solvent' else model.solvent_path

        if not paths:
            self.call_statusbar("error", msg.Error.e03)
            return

        delimiter_list = [",", ";", "\t"]
        delimiter = delimiter_list[delimiter_idx]

        if energy_unit == 'nm':
            SI_energy_unit = 1e-9
        elif energy_unit == 'm':
            SI_energy_unit = 1

        # -------- load TA or solvent data ---------------------------------------------------------
        if model == self.ta_model:
            SI_time_factor = 10**-(12 - 3 * int(time_unit))

            if delA_unit == 'mOD':
                SI_delA_unit = 1
            elif delA_unit == 'OD':
                SI_delA_unit = 1e3

            all_data = []
            for path in paths:
                try:
                    data_array, metadata_list = self._data_from_txt(
                        path, header, delimiter,)

                except (FileNotFoundError, UnicodeDecodeError):
                    self.call_statusbar("error", msg.Error.e03)
                    return
                except ValueError:
                    self.call_statusbar("error", msg.Error.e02)
                    return
                if data_array.size == 0:
                    self.clear_data(raw_ds=raw_ds)
                    self.call_statusbar("error", msg.Error.e02+msg.Error.e30)

                    return

                all_data.append(data_array)

            # loop creates list of np arrays
            average_data = np.average(all_data, axis=0)

            rawdata = {}
            rawdata['wavelength'] = average_data[1:, 0] * SI_energy_unit  # wavelength SI (m)
            rawdata['delay'] = average_data[0, 1:] * SI_time_factor  # delay time SI (s)
            if matrix_orientation == 0:
                rawdata['wavelength'] = average_data[1:, 0] * SI_energy_unit  # wavelength SI (m)
                rawdata['delay'] = average_data[0, 1:] * SI_time_factor  # delay time SI (s)
                dataZ = np.transpose(average_data[1:, 1:]) * SI_delA_unit  # mOD
            else:
                rawdata['wavelength'] = average_data[0, 1:] * SI_energy_unit  # wavelength SI (m)
                rawdata['delay'] = average_data[1:, 0] * SI_time_factor  # delay time SI (s)
                dataZ = (average_data[1:, 1:]) * SI_delA_unit  # mOD

            rawdata['delA'] = np.nan_to_num(dataZ, nan=0)

            if raw_ds == 'solvent':
                model.solvent = rawdata

            else:
                model.rawdata = rawdata

                # only metadata of the last averaged file used:
                metadata_buffer = model.metadata
                metadata_buffer['notes'] = "\n".join(metadata_list)
                model.metadata = metadata_buffer  # needed to trigger the model setter

        # -------- load absorbance or emission data ------------------------------------------------
        else:
            if len(paths) > 2:
                self.call_statusbar("error", msg.Error.e34)
                return
            # two files expeced, while the first file beeing the data before the TA exp
            if len(paths) == 2:

                try:
                    data_array, _ = self._data_from_txt(
                        paths[1], header, delimiter,)
                except (FileNotFoundError, UnicodeDecodeError):
                    self.call_statusbar("error", msg.Error.e03)
                    return
                if data_array.size == 0:
                    self.clear_data(raw_ds=raw_ds)
                    self.call_statusbar(
                        "error", msg.Error.e02+msg.Error.e30)

                    return

                if np.shape(data_array)[1] != 2:
                    self.call_statusbar("error", msg.Error.e32)
                    return
                data_array_corr = data_array[:, 0] * SI_energy_unit

                model.rawdata_after = np.column_stack(
                    (data_array_corr, data_array[:, 1]))

            try:
                data_array, _ = self._data_from_txt(
                    paths[0], header, delimiter,)
            except (FileNotFoundError, UnicodeDecodeError):
                self.call_statusbar("error", msg.Error.e03)
                return
            except ValueError:
                self.call_statusbar("error", msg.Error.e02)
                return
            if data_array.size == 0:
                self.clear_data(raw_ds=raw_ds)
                self.call_statusbar("error", msg.Error.e02+msg.Error.e30)

                return

            # one file expected with the first two columns beeing the data before the TA exp
            if np.shape(data_array)[1] == 4:
                data_buffer_corr2 = data_array[:, 2] * SI_energy_unit
                model.rawdata_after = np.column_stack(
                    (data_buffer_corr2, data_array[:, 3]))

            if np.shape(data_array)[1] == 4 or np.shape(data_array)[1] == 2:
                data_array_corr = data_array[:, 0] * SI_energy_unit

                model.rawdata_before = np.column_stack(
                    (data_array_corr, data_array[:, 1]))
            else:
                self.call_statusbar("error", msg.Error.e33)
                return

    def clear_all(self, raw_ds: str) -> None:
        ''' clears all the data of the raw_ds model, if requested by the user '''
        model = self.get_rawdata_model(raw_ds)
        self.clear_data(raw_ds)
        model.path = None
        model.solvent_path = None

        self.call_statusbar("info", msg.Status.s03)

    def clear_data(self, raw_ds: str) -> None:
        ''' clears the arrays of the raw_ds model, if input errors are detected '''
        model = self.get_rawdata_model(raw_ds)
        if raw_ds != 'solvent':
            model.rawdata_before = None
            model.rawdata_after = None
            model.rawdata = None
        else:
            model.solvent = None

    def delete_ds(self) -> None:
        ''' triggered if TA rawdata is changed in model '''

        self.ta_model_ds1.dsX, self.ta_model_ds1.dsY, self.ta_model_ds1.dsZ = None, None, None
        self.ta_model_ds2.dsX, self.ta_model_ds2.dsY, self.ta_model_ds2.dsZ = None, None, None
        self.ta_model_ds3.dsX, self.ta_model_ds3.dsY, self.ta_model_ds3.dsZ = None, None, None

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)
