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
from pathlib import Path
from ..utils import utils
import datetime
from ..configurations import messages as msg, exceptions as exc
from PyQt6.QtWidgets import QCheckBox

logger = logging.getLogger(__name__)


class VisualizeController(QObject):

    def __init__(self, abs_model, em_model, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()
        self.abs_model = abs_model
        self.em_model = em_model
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
    status_signal = pyqtSignal(str, str)

    def save_data(self, ds: str, datatype: str, metadata: str, data_array: NDArray, results_dir: Path) -> None:
        '''
        view export data helper that saves the data and metadata to csv.

        Parameters
        ----------
        ds : str
            current dataset.
        datatype : str
            plottype (hyperspectrum, delAcuts etc).
        metadata : str
            metadata bundeled with data.
        data_array : NDArray
            data array.
        results_dir : Path
            Path to file.

        Returns
        -------
        None
        '''

        timestamp = datetime.datetime.now().strftime("%y%m%d_")
        file = results_dir / str(timestamp + datatype + '_ds' + ds + '.txt')

        np.savetxt(file, data_array, header=metadata,
                   comments='', fmt='%3.6e', delimiter='\t')

    def get_hyperspectrum(self, ds: str, ukey: str | bool = None, datatype: str | bool = None) -> tuple[str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        ukey : str | bool, optional
            if set, the globally fitted 2D spectrum is returned. The default is None.
        datatype : str | bool, optional
            either simulated or hyperspectrum. The default is None.

        Returns
        -------
        (str, NDArray)
            metadata and data array.

        '''

        x, y, z = self.get_ds_data(ds=ds)
        title = self.get_title()

        data_array = np.zeros((len(x) + 1, len(y) + 1), dtype=np.float32)

        data_array[1:, 0] = x
        data_array[0, 1:] = y
        if ukey:
            fit_dict = self.get_global_fit_list(ds=ds)
            z_calc = fit_dict[ukey]['delA_calc']
            if datatype == 'simulated':
                metadata = f'# simulated hyperspectrum: | {title}{ukey}'
                data_array[1:, 1:] = z_calc.T * 1e-3  # export in OD
            else:
                metadata = f'# hyperspectrum residuals (experimental - fitted): | {title}{ukey}'
                data_array[1:, 1:] = (z - z_calc).T * 1e-3  # export in OD
        else:
            metadata = f'# hyperspectrum: | {title}'
            data_array[1:, 1:] = z.T * 1e-3  # export in OD
        metadata += '\n# first row: delay time (s), first column: wavelength (m), matrix: delA (OD)'

        return metadata, data_array

    def get_delA_plot(self, ds: str, delay_cut_list: list, ukey: str | bool = False) -> tuple[str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        delay_cut_list : list
            list of delay times for cut plots
        ukey : str | bool, optional
            if set, the globally fitted data is returned. The default is None.

        Returns
        -------
        (str, NDArray)
            metadata and data array.

        '''
        x, y, z = self.get_ds_data(ds=ds)
        title = self.get_title()

        delay_cut_list = np.sort(delay_cut_list)  # now ndarray

        ind_delay_found = []
        for v in delay_cut_list:
            idx = (abs(y - v)).argmin()
            ind_delay_found.append(idx)

        if ukey:
            fit_dict = self.get_global_fit_list(ds=ds)
            z_calc = fit_dict[ukey]['delA_calc']

            metadata = (f'# delA Plots global fit vs experimental: | {title}{ukey}\n'
                        '# first row: delay times (s), first column: wavelength (m), second column:'
                        ' delA fit (OD), third column: delA exp (OD)\n')
            data_array = np.zeros(
                (len(x), 3 * len(ind_delay_found)), dtype=np.float32)
            for i, ind in enumerate(ind_delay_found):

                data_array[:, 3 * i] = x
                data_array[:, 3 * i + 1] = z_calc[ind, :] * 1e-3
                data_array[:, 3 * i + 2] = z[ind, :] * 1e-3
                cut = np.round(y[ind], decimals=14)
                metadata += f'{cut:3.6e}\t\t\t'
            metadata = metadata[:-2]  # remove trailling delimiter

        else:
            metadata = (f'# delA Plots: | {title}\n# first row: delay times (s), first column:'
                        ' wavelength (m), second column: delA (OD)\n')
            data_array = np.zeros(
                (len(x), 2 * len(ind_delay_found)), dtype=np.float32)
            for i, ind in enumerate(ind_delay_found):

                data_array[:, 2 * i] = x
                data_array[:, 2 * i + 1] = z[ind, :] * 1e-3
                cut = np.round(y[ind], decimals=14)
                metadata += f'{cut:3.6e}\t\t'
            metadata = metadata[:-1]  # remove trailling delimiter
        return metadata, data_array

    def get_kin_trace(self, ds, wavelength_trace_list: list, ukey: str | bool = False) -> tuple[str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        wavelength_trace_list : list
            list of wavelengths for cut plots
        ukey : str | bool, optional
            if set, the globally fitted data is returned. The default is None.

        Returns
        -------
        (str, NDArray)
            metadata and data array.

        '''
        x, y, z = self.get_ds_data(ds=ds)
        title = self.get_title()

        wavelength_trace_list = np.sort(wavelength_trace_list)  # now ndarray

        ind_wavelengths_found = []
        for v in wavelength_trace_list:
            idx = (abs(x - v)).argmin()
            ind_wavelengths_found.append(idx)

        if ukey:  # global fit data
            fit_dict = self.get_global_fit_list(ds=ds)
            z_calc = fit_dict[ukey]['delA_calc']
            data_array = np.zeros(
                (len(y), 3 * len(ind_wavelengths_found)), dtype=np.float32)
            metadata = (f'# kinetic traces global fit vs experimental: | {title}{ukey}\n'
                        '# first row: wavelengths (m), first column: delay times (s), '
                        'second column: delA fit (OD), third column: delA exp (OD)\n')
            for i, ind in enumerate(ind_wavelengths_found):
                data_array[:, 3 * i] = y
                data_array[:, 3 * i + 1] = z_calc[:, ind] * 1e-3
                data_array[:, 3 * i + 2] = z[:, ind] * 1e-3
                cut = np.round(x[ind], decimals=14)
                metadata += f'{cut:3.6e}\t\t\t'
            metadata = metadata[:-2]  # remove trailling delimiter
        else:  # only kinetic trace
            data_array = np.zeros(
                (len(y), 2 * len(ind_wavelengths_found)), dtype=np.float32)
            metadata = (f'# kinetic traces: | {title}\n# first row: wavelengths (m), first column:'
                        ' delay times (s), second column: delA (OD)\n')
            for i, ind in enumerate(ind_wavelengths_found):
                data_array[:, 2 * i] = y
                data_array[:, 2 * i + 1] = z[:, ind] * 1e-3
                cut = np.round(x[ind], decimals=14)
                metadata += f'{cut:3.6e}\t\t'
            metadata = metadata[:-1]  # remove trailling delimiter
        return metadata, data_array

    def get_emcee_flatchain(self, ds: str, ukey: str, local: bool = False) -> tuple[str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        ukey : str 
            key of the globally fitted data.

        Returns
        -------
        (str, NDArray)
            metadata and data array.

        '''
        if local:
            fit_dict = self.get_local_fit_list(ds=ds)
        else:
            fit_dict = self.get_global_fit_list(ds=ds)
        array = fit_dict[ukey]['emcee']['flatchain']
        params = list(fit_dict[ukey]['emcee']['params'].keys())
        title = self.get_title()
        metadata = (f'# emcee flatchain: | {title}{ukey}\n# shape: num steps x num walkers, num params\n'
                    f'# colums: {params} (in s). __lnsigma refers to the log of the standard error of the noise')
        return metadata, array

    def get_global_ukey(self, checkboxes: list):
        '''helper that returns the text of the checked checkbox of a list of checkboxes '''

        checked_box = [cb for cb in checkboxes if cb.isChecked()]
        if not checked_box:
            raise exc.NoSelectionError()
        return checked_box[0].text()

    def get_local_fit(self, ds: str, selected_ukeys: list[str]) -> tuple[str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        selected_ukeys : list[str] 
            ukeys of selected fits.

        Returns
        -------
        (str, NDArray)
            metadata and data array.

        '''
        x, y, z = self.get_ds_data(ds=ds)
        fit_dict = self.get_local_fit_list(ds=ds)
        title = self.get_title()
        metadata = (f'# local fit traces: | {title}\n'
                    '# first row: wavelengths (m), '
                    'first column: delay times (s), second column: delA fit (OD), '
                    'third column: delA exp (OD), '
                    'fourth onwards: concentration profiles (OD) (if more than one component)\n')

        n_cols = 0
        for ukey in selected_ukeys:
            info = fit_dict[ukey]
            conc = info['conc']
            # always 3 cols: delay, delA_calc, delA
            n_cols += 3
            # plus one column per concentration component (only if >1)
            if conc.shape[1] >= 2:
                n_cols += conc.shape[1]

        # 2) allocate the array
        data_array = np.zeros((len(y), n_cols), dtype=np.float32)

        # 3) fill it in
        col = 0
        for ukey in selected_ukeys:
            info = fit_dict[ukey]
            conc = info['conc']
            Amp = info['Amp']
            delay = info['delay']
            delA_calc = info['delA_calc']
            delA = info['delA']

            # add wavelength to metadata line
            cut = np.round(info['wavelength'], decimals=14)
            metadata += f'{cut:3.6e}\t\t\t'
            if conc.shape[1] >= 2:
                metadata += "\t" * conc.shape[1]

            # fill in the 3 standard columns
            data_array[:, col] = delay
            data_array[:, col + 1] = delA_calc
            data_array[:, col + 2] = delA

            # fill in any extra concentration profiles
            if conc.shape[1] >= 2:
                for k in range(conc.shape[1]):
                    data_array[:, col + 3 + k] = conc[:, k] * Amp[k]
                col += 3 + conc.shape[1]
            else:
                col += 3
        metadata = metadata[:-1]  # remove trailling delimiter
        return metadata, data_array

    def get_conc(self, ds: str, ukey: str) -> tuple[str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        ukey : str 
            key of the globally fitted data.

        Returns
        -------
        (str, NDArray)
            metadata and data array.

        '''
        fit_dict = self.get_global_fit_list(ds=ds)
        components = fit_dict[ukey]['meta']['components']
        _, y, _ = self.get_ds_data(ds=ds)
        title = self.get_title()
        metadata = (f'# global fit concentration profile: | {title}{ukey}\n'
                    '# first column: delay times (ms), '
                    f'second column onwards: normalized concentration {components}')
        data_array = np.zeros(
            (fit_dict[ukey]['conc'].shape[0], fit_dict[ukey]['conc'].shape[1]+1), dtype=np.float32)
        data_array[:, 0] = y
        data_array[:, 1:] = fit_dict[ukey]['conc']

        return metadata, data_array

    def get_EASDAS(self, ds: str, ukey: str,) -> tuple[str, str, NDArray]:
        '''
        view export data helper that gets the model data and sets metadata

        Parameters
        ----------
        ds : str
            current dataset.
        ukey : str 
            key of the globally fitted data.

        Returns
        -------
        (str, str, NDArray)
            datatpe (eg EAS/DAS/SAS), metadata and data array.

        '''
        fit_dict = self.get_global_fit_list(ds=ds)
        fit_model = fit_dict[ukey]['meta']['model']
        components = fit_dict[ukey]['meta']['components']
        if fit_model == 'parallel':
            datatype = 'DAS'
        elif fit_model == 'sequential':
            datatype = 'EAS'
        else:
            datatype = 'SAS'
        x, y, z = self.get_ds_data(ds=ds)
        title = self.get_title()
        metadata = (f'# global fit {datatype}: | {title}{ukey}\n'
                    '# first column: wavelengths (m), '
                    f'second column onwards: delA (OD) {components}')
        data = fit_dict[ukey][datatype]

        data_array = np.zeros(
            (data.shape[0], data.shape[1]+1), dtype=np.float32)
        data_array[:, 0] = x
        data_array[:, 1:] = data

        return datatype, metadata, data_array

    def get_pump(self) -> float:
        '''
        retrives the metadata of excitation wavelength and converts it into float.
        Used to visualize the pump wavelength

        Returns
        -------
        pump (float)


        '''
        pump_meta = self.ta_model.metadata['excitation wavelength']

        return utils.Converter.convert_str_input2float(pump_meta)

    def get_title(self) -> str:
        ''' gets the meta information of the TA model and returns it as convenient string '''
        meta = self.ta_model.metadata
        title = ''
        if meta['sample'] != '':
            title += meta['sample'] + '  |  '
        if meta['excitation wavelength'] != '':
            title += meta['excitation wavelength'] + '  |  '
        if meta['excitation intensity'] != '':
            title += meta['excitation intensity'] + '  |  '
        if meta['solvent'] != '':
            title += meta['solvent'] + '  |  '
        return title

    def verify_rawdata(self) -> bool:
        ''' checks if rawdata is set in the model '''
        if not self.ta_model.rawdata:
            self.call_statusbar("error", msg.Error.e05)
            return False
        else:
            return True

    def verify_steadystate(self) -> bool:
        ''' checks if steady-state data is set in the model '''
        if self.abs_model.rawdata_before is not None or self.em_model.rawdata_before is not None:
            return True
        else:
            return False

    def get_local_fit_list(self, ds: str) -> dict:
        ''' returns the local fit list stored in TA ds model '''
        ds_model = self._get_ds_model(ds)
        return ds_model.local_fit

    def get_global_fit_list(self, ds: str) -> dict:
        ''' returns the global fit list stored in TA ds model '''
        ds_model = self._get_ds_model(ds)
        return ds_model.global_fit

    def _get_rawdata_model(self, raw_ds: str) -> object:
        ''' helper that returns the model object of the name raw_ds'''
        if raw_ds == 'abs':
            return self.abs_model
        elif raw_ds == 'em':
            return self.em_model
        elif raw_ds == 'ta':
            return self.ta_model

    def get_ss_data(self, raw_ds: str) -> tuple[NDArray, NDArray]:
        ''' returns the steady-state data array (before and after) of a given raw_ds (abs/em) '''
        model = self._get_rawdata_model(raw_ds)
        return model.rawdata_before, model.rawdata_after

    def get_ds_data(self, ds: str) -> tuple[NDArray, NDArray, NDArray]:
        ''' returns the data stored in a given dataset ds, or returns the rawdata if ds is empty '''
        model = self._get_ds_model(ds)
        if model.dsZ is None:
            return self.ta_model.rawdata['wavelength'], self.ta_model.rawdata['delay'], self.ta_model.rawdata['delA']
        else:
            return model.dsX, model.dsY, model.dsZ

    def _get_ds_model(self, ds: str) -> object:
        ''' helper that returns the model object of the name ds '''
        if ds == '1':
            return self.ta_model_ds1
        if ds == '2':
            return self.ta_model_ds2
        if ds == '3':
            return self.ta_model_ds3

    def get_exp_names(self) -> tuple[str, str]:
        ''' reutrns the ta model metadata experiment and sample name'''
        return self.ta_model.metadata['experiment'], self.ta_model.metadata['sample']

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)
