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
import scipy.signal
import scipy.optimize
from scipy.interpolate import (
    make_interp_spline, PchipInterpolator, Akima1DInterpolator
)
import logging
import logging.config
from ..configurations import messages as msg, exceptions as exc

from scipy.optimize import least_squares
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class PreprocController(QObject):

    def __init__(self, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()

        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
    status_signal = pyqtSignal(str, str)
    data_changed = pyqtSignal()

    def verify_rawdata(self) -> bool:
        ''' checks if rawdata is set in the model '''
        if not self.ta_model.rawdata:
            self.call_statusbar("error", msg.Error.e05)
            return False
        else:
            return True

    def get_ds_model(self, ds: str) -> object:
        ''' helper that returns the model object of the name ds '''
        if ds == '1' or ds == 'ds 1':
            return self.ta_model_ds1
        if ds == '2' or ds == 'ds 2':
            return self.ta_model_ds2
        if ds == '3' or ds == 'ds 3':
            return self.ta_model_ds3

    def get_solvent(self) -> tuple[NDArray, NDArray, NDArray]:
        if self.verify_rawdata() is False:
            raise exc.NoDataError()
        return self.ta_model.solvent['wavelength'], self.ta_model.solvent['delay'], self.ta_model.solvent['delA']

    def get_ds_data(self, ds: str) -> tuple[NDArray, NDArray, NDArray]:
        ''' returns the data stored in a given dataset ds, or returns the rawdata if ds is empty '''
        model = self.get_ds_model(ds)
        if model.dsZ is None:
            return self.ta_model.rawdata['wavelength'], self.ta_model.rawdata['delay'], self.ta_model.rawdata['delA']
        else:
            return model.dsX, model.dsY, model.dsZ

    def correct_t0(self, ds: str, t0: float) -> NDArray:
        ''' substracts t0 from the model delay vector '''
        model = self.get_ds_model(ds)
        if self.verify_rawdata() is False:
            raise exc.NoDataError()

        if model.dsX is None:
            return self.ta_model.rawdata['delay'] - t0
        else:
            return model.dsY - t0

    def apply_t0(self, ds: str, new_dsY: NDArray, t0: float) -> None:
        ''' saves new delay vector to model and updates meta data '''
        model = self.get_ds_model(ds)
        if model.dsZ is None:
            model.dsX = self.ta_model.rawdata['wavelength']
            model.dsY = new_dsY
            model.dsZ = self.ta_model.rawdata['delA']
        else:
            model.dsY = new_dsY
            self.data_changed.emit()
        meta = f'time zero changed to {t0:.3} s'
        model.update_metadata(meta)
        self.call_statusbar("info", msg.Status.s27)

    def substract_surface(self, ds: str, background: NDArray) -> NDArray:
        ''' substracts a 2D background from the model delA data and returns the array '''
        if self.verify_rawdata() is False:
            raise exc.NoDataError()
        _, _, delA = self.get_ds_data(ds)
        if delA.shape != background.shape:
            raise exc.InputLogicError
        return delA - background

    def substract_area(self, ds: str, method: str, ymin: float, ymax: float, corr_max: float) -> NDArray:
        '''
        Subtract a background (mean or median) along the y-axis slice [ymin,ymax].

        Parameters
        ----------
        ds : str
            current dataset.
        method : str
            averaging method.
        ymin : float
            lower bound of area from which background is calculated, if None: min value.
        ymax : float
            upper bound of area from which background is calculated.
        corr_max : float, bool
            upper bound to where background is applied. if None: whole dataset.

        Raises
        ------
        exc.InputLogicError
            if ymin larger than ymax.

        Returns
        -------
        NDArray
            background corrected delA matrix.

        '''
        model = self.get_ds_model(ds)

        if model.dsX is None:
            y = self.ta_model.rawdata['delay']
            z = self.ta_model.rawdata['delA']
        else:
            y = model.dsY
            z = model.dsZ

        idx_min = 0 if ymin is None else int(np.abs(y - ymin).argmin())
        idx_max = int(np.abs(y - ymax).argmin())

        corr_max_idx = None if corr_max is None else int(np.abs(y - corr_max).argmin())

        if idx_min >= idx_max:
            raise exc.InputLogicError

        slice_ = z[idx_min:idx_max]
        if method == 'mean':
            corr = np.mean(slice_, axis=0)
        elif method == 'median':
            corr = np.median(slice_, axis=0)
        z_corr = z.copy()
        z_corr[:corr_max_idx, :] = z_corr[:corr_max_idx, :] - corr
        return z_corr

    def regularize_grid(self, ds: str, datapoints: int, axis_str: str, method: str) -> tuple[NDArray, NDArray, NDArray]:
        """
        Resample one axis of a 2D dataset (wavelength,delay, ΔA) to an evenly spaced grid.

        Parameters
        ----------
        ds : str
            Dataset identifier for self.get_ds_data(ds).
        datapoints : int
            Number of points on the new, regular grid.
        axis_str : str
            Which axis to regularize: 'timepoints' -> axis 0, otherwise axis 1.
        method : str
            Interpolation method: 'linear', 'quadratic', 'cubic', 'pchip', 'akima'.

        Returns
        -------
        x_full : (M,) ndarray
            New wavelength grid if axis = 'wavelength', otherwise unchanged.
        y_full : (N,) ndarray
            New delay grid if axis = 'timepoints', otherwise unchanged.
        delA_full : (M, N) ndarray
            ΔA resampled onto the regularized axis.
        """
        # -------- extract data --------------------------------------------------------------------
        x, y, delA = self.get_ds_data(ds)
        interp_axis = 0 if axis_str == "timepoints" else 1

        if interp_axis == 0:
            coord_old = y
            other_old = x
        else:
            coord_old = x
            other_old = y

        coord_new = np.linspace(coord_old.min(),
                                coord_old.max(),
                                datapoints)
        if interp_axis == 0:
            data_sec = delA             # shape (ntime, nwave)
        else:
            data_sec = delA.T           # shape (nwave, ntime)

        # -------- create spline and interpolator object -------------------------------------------
        m = method.lower()
        if m in ("linear", "quadratic", "cubic"):
            k = {"linear": 1, "quadratic": 2, "cubic": 3}[m]
            spline = make_interp_spline(coord_old,
                                        data_sec,
                                        k=k,
                                        )
        elif m == "pchip":
            spline = PchipInterpolator(coord_old,
                                       data_sec, extrapolate=True)
        elif m in ("akima", "makima"):
            spline = Akima1DInterpolator(coord_old,
                                         data_sec, extrapolate=True)

        # -------- interpolat to new vector --------------------------------------------------------
        sampled = spline(coord_new)     # shape (datapoints, other_len)
        if interp_axis == 0:
            delA_full = sampled         # rows replaced
            x_full, y_full = other_old, coord_new
        else:
            delA_full = sampled.T       # transpose back
            x_full, y_full = coord_new, other_old

        return x_full, y_full, delA_full

    def resample_grid(self, ds: str, factor: int | float, interval: tuple, resample_delay: bool, method: str) -> tuple[NDArray, NDArray, NDArray]:
        '''
        resample a (irregular) grid over one axis and a given interval by a factor.

        Parameters
        ----------
        ds : str
            model identifyer.
        factor : int | float
            a factor higher 1 will upsample the given grid, a factor between 0 and 1 will downsample
        interval : tuple
            interval on the given axes to which the sampling is applied.
        resample_delay : bool
            if true, delay axis will be resampled, if false then the wavelength axis.
        method : str
            method used by the interpolator .

        Raises
        ------
        exc.InputLogicError
            raised if invalid interval is given

        Returns
        -------
        x_full : (M,) ndarray
            New wavelength grid if axis = 'wavelength', otherwise unchanged.
        y_full : (N,) ndarray
            New delay grid if axis = 'timepoints', otherwise unchanged.
        delA_full : (M, N) ndarray
            ΔA resampled onto the regularized axis.
        '''
        # -------- extract data --------------------------------------------------------------------
        x, y, delA = self.get_ds_data(ds)

        if resample_delay:                      # rows (axis 0)
            coord, other, axis = y, x, 0
        else:                                   # columns (axis 1)
            coord, other, axis = x, y, 1

        lo_idx = 0 if interval[0] is None else int(np.abs(coord - interval[0]).argmin())
        hi_idx = (
            len(coord)-1) if interval[1] is None else int(np.abs(coord - interval[1]).argmin())

        if hi_idx - lo_idx < 1:                    # need at least two samples
            raise exc.InputLogicError

        slc = slice(lo_idx, hi_idx + 1)            # inclusive

        # coord/other are unchanged
        coord_sec = coord[slc]
        data_sec = np.take(delA, np.arange(lo_idx, hi_idx + 1), axis=axis)
        N_old = coord_sec.size
        N_new = max(int(round(N_old * factor)), 2)
        idx_old = np.arange(N_old)
        idx_new = np.linspace(0, N_old - 1, N_new)

        # map back to real coordinates (preserves original spacing shape)
        coord_new = np.interp(idx_new, idx_old, coord_sec)

        # -------- create spline and interpolator object -------------------------------------------
        method = method.lower()
        if method in ("linear", "quadratic", "cubic"):
            k = {"linear": 1, "quadratic": 2, "cubic": 3}[method]
            spline = make_interp_spline(coord_sec, data_sec,
                                        k=k, axis=axis)       # extrapolate=True by default
        elif method == "pchip":
            spline = PchipInterpolator(coord_sec, data_sec,
                                       axis=axis, extrapolate=True)
        elif method in ("akima", "makima"):
            spline = Akima1DInterpolator(coord_sec, data_sec,
                                         axis=axis, method=method,
                                         extrapolate=True)

        # -------- interpolat to new vector and stich together data --------------------------------
        data_new = spline(coord_new)             # evaluates + slope‑extrapolates

        # full coordinate vector:
        coord_full = np.concatenate([
            coord[:lo_idx],
            coord_new,
            coord[hi_idx+1:]])

        if axis == 1:
            # resampling columns ↔ concatenate along axis=1
            left = delA[:, :lo_idx]
            right = delA[:, hi_idx+1:]
            delA_full = np.concatenate([left, data_new, right], axis=1)
            x_full, y_full = coord_full, other
        else:
            # resampling rows ↔ concatenate along axis=0
            top = delA[:lo_idx, :]
            bottom = delA[hi_idx+1:, :]
            delA_full = np.concatenate([top, data_new, bottom], axis=0)
            x_full, y_full = other, coord_full

        return x_full, y_full, delA_full

    def apply_resampling(self, ds: str, x: NDArray, y: NDArray, z: NDArray, meta: str) -> None:
        ''' update model and metadata with the given data arrays and meta information '''
        model = self.get_ds_model(ds)
        model.dsX = x
        model.dsY = y
        model.dsZ = z
        model.update_metadata(meta)
        self.call_statusbar("info", msg.Status.s26)

    def apply_background(self, corr_type: str,  ds: str, corrected_dsZ: NDArray,
                         method: str | bool = None, area_lim: tuple | bool = None,
                         ymax: float = None, meta_file: str | bool = None,
                         meta_ds: str | bool = None, meta_background: NDArray | bool = False) -> None:
        ''' update model and metadata with the given data arrays and meta information '''
        model = self.get_ds_model(ds)
        if corrected_dsZ is None:  # no data fetched
            self.call_statusbar("error", msg.Error.e05)
            return

        if model.dsX is None:  # apply to rawdata
            if np.array_equal(corrected_dsZ, self.ta_model.rawdata['delA']):
                self.call_statusbar("error", msg.Error.e13)
                return
            model.dsX = self.ta_model.rawdata['wavelength']
            model.dsY = self.ta_model.rawdata['delay']

        else:  # apply to current ds
            if np.array_equal(corrected_dsZ, model.dsZ):
                self.call_statusbar("error", msg.Error.e13)
                return

        model.dsZ = corrected_dsZ

        if corr_type == 'area':
            if area_lim[0] is None:
                area_lim[0] = 'earliest time'
            if ymax is None:
                ymax_str = 'from the whole dataset'
            else:
                ymax_str = f'up to {ymax:.3} (s) from the dataset'
            meta = f'Background corrected by calculating the {method} from {area_lim[0]} to {area_lim[1]} (s) and substracting it {ymax_str}.'
        elif corr_type == 'file':
            meta = f'Background corrected by substracting dataset {meta_ds} from project {meta_file}.'
            model.set_background_surf(meta_background)
        elif corr_type == 'solvent':
            meta = 'Solvent substracted'
        model.update_metadata(meta)
        self.call_statusbar("info", msg.Status.s09)

    def apply_filter(self, ds: str, corrected_dsZ: NDArray, filter_idx: int, filter_axis: int,
                     window: int, order: int) -> None:
        ''' update model and metadata with the given data arrays and meta information '''
        model = self.get_ds_model(ds)
        if corrected_dsZ is None:  # no data fetched
            self.call_statusbar("error", msg.Error.e05)
            return

        if model.dsX is None:  # apply to rawdata
            if np.array_equal(corrected_dsZ, self.ta_model.rawdata['delA']):
                self.call_statusbar("error", msg.Error.e13)
                return
            model.dsX = self.ta_model.rawdata['wavelength']
            model.dsY = self.ta_model.rawdata['delay']
            model.dsZ = corrected_dsZ

        else:  # apply to current ds
            if np.array_equal(corrected_dsZ, model.dsZ):
                self.call_statusbar("error", msg.Error.e13)
                return
        if filter_idx == 0:
            filter_type = f'Savitzki-Golay Filter with Window {window} and order {order}'
        elif filter_idx == 1:
            filter_type = f'Moving Median Filter with size {window}'
        elif filter_idx == 2:
            filter_type = f'Moving Average Filterwith size {window}'

        if filter_axis == 0:  # filter along timepoints
            axis = 'along timepoint axis'
        else:
            axis = 'along wavelength axis'

        meta = f'{filter_type} applied {axis}.'
        model.update_metadata(meta)
        self.call_statusbar("info", msg.Status.s15)

    def apply_trimm(self, ds: str, xmin: float | None, xmax: float | None, ymin: float | None, ymax: float | None):
        '''
        checks if trimm values are set and within selected ds/rawdata
        and updates the models dataset.

        Parameters
        ----------
        ds : str
            current dataset.
        xmin, xmax,ymin, ymax : float or None
            requested trimm range

        '''
        model = self.get_ds_model(ds)

        if self.verify_rawdata() is False:
            return
        if model.dsX is None:
            if xmin is None or xmin < np.min(self.ta_model.rawdata['wavelength']):
                xmin = np.min(self.ta_model.rawdata['wavelength'])
            if xmax is None or xmax > np.max(self.ta_model.rawdata['wavelength']):
                xmax = np.max(self.ta_model.rawdata['wavelength'])
            if ymin is None or ymin < np.min(self.ta_model.rawdata['delay']):
                ymin = np.min(self.ta_model.rawdata['delay'])
            if ymax is None or ymax > np.max(self.ta_model.rawdata['delay']):
                ymax = np.max(self.ta_model.rawdata['delay'])

            idx_xmin, idx_xmax = (np.abs(self.ta_model.rawdata['wavelength'] - xmin)
                                  ).argmin(), (np.abs(self.ta_model.rawdata['wavelength'] - xmax)).argmin()
            idx_ymin, idx_ymax = (np.abs(self.ta_model.rawdata['delay'] - ymin)
                                  ).argmin(), (np.abs(self.ta_model.rawdata['delay'] - ymax)).argmin()

            model.dsX = self.ta_model.rawdata['wavelength'][idx_xmin: idx_xmax + 1]
            model.dsY = self.ta_model.rawdata['delay'][idx_ymin: idx_ymax + 1]
            model.dsZ = self.ta_model.rawdata['delA'][idx_ymin: idx_ymax +
                                                      1, idx_xmin: idx_xmax + 1]

        else:
            if xmin is None or xmin < np.min(model.dsX):
                xmin = np.min(model.dsX)
            if xmax is None or xmax > np.max(model.dsX):
                xmax = np.max(model.dsX)
            if ymin is None or ymin < np.min(model.dsY):
                ymin = np.min(model.dsY)
            if ymax is None or ymax > np.max(model.dsY):
                ymax = np.max(model.dsY)

            idx_xmin, idx_xmax = (np.abs(model.dsX - xmin)
                                  ).argmin(), (np.abs(model.dsX - xmax)).argmin()
            idx_ymin, idx_ymax = (np.abs(model.dsY - ymin)
                                  ).argmin(), (np.abs(model.dsY - ymax)).argmin()

            model.dsX = model.dsX[idx_xmin: idx_xmax + 1]
            model.dsY = model.dsY[idx_ymin: idx_ymax + 1]
            model.dsZ = model.dsZ[idx_ymin: idx_ymax +
                                  1, idx_xmin: idx_xmax + 1]
        meta = f'Data trimmed from {xmin:.3} m to {xmax:.3} nm and {ymin:.3} (s) to {ymax:.3} (s).'
        model.update_metadata(meta)
        self.call_statusbar("info", msg.Status.s13)

    def savgol_filter_data(self, ds: str, window: int, order: int, filter_axis: int) -> NDArray:
        '''
        Apply a Savitzky-Golay filter to an array.

        Parameters
        ----------
        ds : str
            holds the name of the model.
        window : int
            window length used by the filter.
        order : int
            order of the polynomial used to fit the data.
        filter_axis : int
            0: filter timepoints/delay; 1: filter wavelength.

        Raises
        ------
        AttributeError
            if no raw_data is found.

        Returns
        -------
        NDArray
            the filtered delA 2D array.

        '''
        model = self.get_ds_model(ds)
        if self.verify_rawdata() is False:
            raise exc.NoDataError()

        if model.dsX is None:  # apply to raw_data
            new_dsZ = np.empty(self.ta_model.rawdata['delA'].shape)

            if filter_axis == 0:  # filter along timepoints

                for i in range(len(self.ta_model.rawdata['wavelength'])):
                    new_dsZ[:, i] = scipy.signal.savgol_filter(
                        self.ta_model.rawdata['delA'][:, i], window, order)

            if filter_axis == 1:  # filter along wavelength

                for i in range(len(self.ta_model.rawdata['delay'])):
                    new_dsZ[i, :] = scipy.signal.savgol_filter(
                        self.ta_model.rawdata['delA'][i, :], window, order)

        else:  # apply to ds
            new_dsZ = np.empty(model.dsZ.shape)

            if filter_axis == 0:  # filter along timepoints
                for i in range(len(model.dsX)):
                    new_dsZ[:, i] = scipy.signal.savgol_filter(
                        model.dsZ[:, i], window, order)

            if filter_axis == 1:  # filter along wavelength
                for i in range(len(model.dsY)):
                    new_dsZ[i, :] = scipy.signal.savgol_filter(
                        model.dsZ[i, :], window, order)

        return new_dsZ

    def moving_filter_data(self, ds: str, method: str, size: int, filter_axis: int) -> NDArray:
        '''
        Apply a moving median or uniform moving average filter to the delA array over the provided axis.

        Parameters
        ----------
        ds : str
            holds the name of the model.
        method : str
            median or average.
        size : int
            the size of the filter.
        filter_axis : int
            0: filter timepoints/delay; 1: filter wavelength.

        Raises
        ------
        AttributeError
            raised if no raw_data is found.

        Returns
        -------
        NDArray
            the filtered delA 2D array.

        '''
        model = self.get_ds_model(ds)
        if self.verify_rawdata() is False:
            raise exc.NoDataError()

        if model.dsX is None:  # apply to raw_data
            if method == 'median':
                new_dsZ = scipy.ndimage.median_filter(
                    input=self.ta_model.rawdata['delA'], size=size, mode='nearest', axes=filter_axis)
            elif method == 'average':
                new_dsZ = scipy.ndimage.uniform_filter(
                    input=self.ta_model.rawdata['delA'], size=size, mode='nearest', axes=filter_axis)

        else:  # apply to ds
            if method == 'median':
                new_dsZ = scipy.ndimage.median_filter(
                    input=model.dsZ, size=size, mode='nearest', axes=filter_axis)
            elif method == 'average':
                new_dsZ = scipy.ndimage.uniform_filter(
                    input=model.dsZ, size=size, mode='nearest', axes=filter_axis)

        return new_dsZ

    def autofind_chirp(self, ds: str, threshold_chirp: float | None, cb_index: int) -> NDArray:
        '''
        finds valid chirp points in the 2D delA matrix and returns their coordinates.

        The algorithm loops through every wavelength and tries to find the point where the threshold
        condition is met (absolute, relative or max) of the filtered traces. The points must be at
        least twice as high as the estimated noise level or will be discarded.

        Parameters
        ----------
        ds : str
            holds the name of the model..
        threshold_chirp : float | None
            threshold value, either mOD (default: 0.1 mOD) or % (default: 10%).
        cb_index : int
            0: threshold in mOD, 1: relative threshold, 2: use maximum.

        Raises
        ------
        exc.NoDataError
            raised if no raw_data is found.
        ValueError
            raised if less than 5 valid points where found.

        Returns
        -------
        NDArray
            chirp points (x,y).

        '''

        # -------- extract data --------------------------------------------------------------------
        model = self.get_ds_model(ds)
        # Check for required raw data.
        if self.verify_rawdata() is False:
            raise exc.NoDataError()
        # Set default threshold values if none provided.
        if threshold_chirp is None:
            threshold_chirp = 0.1 if cb_index == 0 else 10

        # Choose the appropriate dataset.
        if model.dsX is None:
            x_data = self.ta_model.rawdata['wavelength']
            y_data = self.ta_model.rawdata['delay']
            z_data = self.ta_model.rawdata['delA']
        else:
            x_data = model.dsX
            y_data = model.dsY
            z_data = model.dsZ

        detected_x = []
        detected_y = []
        n_data_points = len(x_data)

        # -------- use heuristic to find valid chirp points ----------------------------------------
        for col_idx in range(n_data_points):
            # Process each column (data point) using a Savitzky-Golay filter.
            current_signal = z_data[:, col_idx]
            filtered_signal = scipy.signal.savgol_filter(current_signal, 11, 3)
            noise_level = (np.mean(np.abs(current_signal[:10]))
                           if current_signal.shape[0] >= 10
                           else np.mean(np.abs(current_signal)))

            if cb_index == 0:  # Threshold in mOD.
                # Find the first index where the filtered signal exceeds the threshold.
                indices = np.where(np.abs(filtered_signal)
                                   > threshold_chirp)[0]
                if indices.size > 0:
                    index = indices[0]
                    if index != 0 and abs(current_signal[index]) > 2 * noise_level:
                        detected_x.append(x_data[ col_idx])
                        detected_y.append(y_data[index])

            elif cb_index == 1:  # Threshold in percentage.
                max_val = np.max(np.abs(current_signal))
                indices = np.where(np.abs(filtered_signal) >=
                                   max_val * threshold_chirp / 100)[0]
                if indices.size > 0:
                    index = indices[0]
                    if index != 0 and abs(current_signal[index]) > 2 * noise_level:
                        detected_x.append(x_data[ col_idx])
                        detected_y.append(y_data[index])

            elif cb_index == 2:  # Maximum value detection.

                index = np.argmax(np.abs(filtered_signal))
                if index != 0 and abs(current_signal[index]) > 2 * noise_level:
                    detected_x.append(x_data[ col_idx])
                    detected_y.append(y_data[index])

        if len(detected_x) < 5:
            raise ValueError("Insufficient detected points.")

        return np.column_stack((detected_x, detected_y))

    def fit_chirp(self, chirp_points: NDArray) -> NDArray:
        '''
        Fit a 4th degree polynomial to the chirp points using a robust least squares method.

        Parameters
        ----------
        chirp_points : np.array
            A 2D array where the first column is the wavelength (wl)
                                    and the second column is the delay time (delay)

        Returns
        -------
        coeff (ndarray)
            Fitted polynomial coefficients in order of increasing degree (i.e., constant term first

        '''
        wl = chirp_points[:, 0]
        delay = chirp_points[:, 1]

        def residuals(params, x, y):
            return y - np.polynomial.polynomial.polyval(x, params)

        # Initial guess for polynomial coefficients (e.g., for a 2nd degree polynomial)
        initial_guess = np.polynomial.polynomial.polyfit(wl, delay, 4)
        # Perform robust fitting using least_squares with Cauchy loss
        result = least_squares(residuals, initial_guess, args=(
            wl, delay), loss='cauchy', f_scale=1.0)

        coeff = result.x

        return coeff

    def calculate_chirp_corrected_dsZ(self, ds: str, chirp_coeff: NDArray) -> NDArray:
        '''
        calculates chirp giving the results of the correction fit (chirp_coeff).

        at each wavelength set in the Z data (column), the data needs to be shifted by the chirp time (dataY + polyval(fit_coeff, wavelength)).
        This is done with np.interp(x, xp, fp). 
        In order to rowwise loop through all the wavelengths efficiently, z needs to be transposed first

        Parameters
        ----------
        ds : str
            holds the name of the model.
        chirp_coeff : ndarray
            Fitted polynomial coefficients in order of increasing degree (i.e., constant term first. returned from fit_chirp fnc

        Returns
        -------
        dsZ : ndarray
            Chirp corrected delA data matrix.

        '''
        model = self.get_ds_model(ds)

        if chirp_coeff is None:
            raise ValueError
            return

        if model.dsX is None:
            dsX = self.ta_model.rawdata['wavelength']
            dsY = self.ta_model.rawdata['delay']
            dsZ = self.ta_model.rawdata['delA']

        else:
            dsX = model.dsX
            dsY = model.dsY
            dsZ = model.dsZ

        # Transpose dsZ to work columnwise
        new_dsZ = np.empty(dsZ.T.shape)
        for i, v in enumerate(dsZ.T):
            shift = np.polynomial.polynomial.polyval(dsX[i], chirp_coeff)
            new_dsZ[i, :] = np.interp(
                x=dsY + shift,
                xp=dsY,
                fp=v
            )

        return new_dsZ.T

    def apply_chirp(self, ds: str, chirp_coeff: NDArray) -> None:
        '''
        applies given chirp coefficients to given model:


        Parameters
        ----------
        ds : str
            holds the name of the model.
        chirp_coeff : ndarray
            Fitted polynomial coefficients in order of increasing degree (i.e., constant term first. returned from fit_chirp fnc


        Returns
        -------
        None.

        '''
        if self.verify_rawdata() is False:
            return
        model = self.get_ds_model(ds)
        if chirp_coeff is None:
            self.call_statusbar("error", msg.Error.e07)
            return

        new_dsZ = self.calculate_chirp_corrected_dsZ(
            ds=ds, chirp_coeff=chirp_coeff)
        if model.dsX is None:
            model.dsX = self.ta_model.rawdata['wavelength']
            model.dsY = self.ta_model.rawdata['delay']
        model.dsZ = new_dsZ
        meta = f'Chirp corrected using a 4th order polynomial. Fit coefficients in order of increasing degree {chirp_coeff}'
        model.update_metadata(meta)
        model.set_chirp(chirp_coeff)
        self.call_statusbar("info", msg.Status.s12)

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)

    def clear_ds(self, ds):
        model = self.get_ds_model(ds)
        model.dsX, model.dsY, model.dsZ,  = None, None, None
        model.clear_metadata()
