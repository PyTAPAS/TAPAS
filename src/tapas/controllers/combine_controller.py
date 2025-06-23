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
from pathlib import Path
from numpy.typing import NDArray
from scipy.interpolate import interpn
from scipy.ndimage import distance_transform_edt
import logging
import logging.config
from ..configurations import messages as msg, exceptions as exc
import h5py

logger = logging.getLogger(__name__)


class CombineController(QObject):

    def __init__(self, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()

        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self._init_p1()
        self._init_p2()
        self.x_new, self.y_new, self.z_new = None, None, None

    status_signal = pyqtSignal(str, str)

    def _init_p1(self) -> None:
        ''' helper that resets cache for project 1 '''
        self.p1_rawdata, self.p1_ds = None, None
        self.p1_dsX, self.p1_dsY, self.p1_dsZ = None, None, None

    def _init_p2(self) -> None:
        ''' helper that resets cache for project 2 '''
        self.p2_rawdata, self.p2_ds = None, None
        self.p2_dsX, self.p2_dsY, self.p2_dsZ = None, None, None

    def _read_dict_from_hdf(self, group: 'h5py.group') -> dict:
        """
        Recursively reads an HDF5 group and returns a dictionary.

        Parameters:
          group: an h5py Group to read data from.

        Returns:
          A dictionary representation of the HDF5 group.
        """
        result = {}

        # First, if there are any attributes, add them as a 'meta' dictionary.
        if group.attrs:
            result['meta'] = {attr: group.attrs[attr] for attr in group.attrs}

        # Then iterate over keys in the group.
        for key, item in group.items():
            if isinstance(item, type(group)):  # if item is a subgroup
                result[key] = self._read_dict_from_hdf(item)
            else:
                # Retrieve the dataset's value.
                value = item[()]
                result[key] = value
        return result

    def load_project(self, path: str, ds: str) -> tuple[float | None, float | None, float | None, float | None]:
        '''
        loads the project, caches the X,Y,Z matrices of the selected project P1 or P2
        for fun combine_ds and caches the rawdata history for the metadata.

        Parameters
        ----------
        path : str
            points to the .hdf to be readout.
        ds : str
            either raw or 'ds 1', 'ds 2' etc.

        Returns
        -------
        TYPE
            min wavelength.
        TYPE
            max wavelength.
        TYPE
            min delay.
        TYPE
            max delay.
        returns None, if file not found, ds not found etc.

        '''
        ds = 'raw Data' if ds == 'raw' else ds

        # ---- normalize the path input ----
        if not isinstance(path, Path):
            s = str(path).strip().strip('\'"')
            path_obj = Path(s)
        else:
            path_obj = path

        if not path_obj.exists():
            self.call_statusbar("error", msg.Error.e03)
            return None, None, None, None
        if not path_obj.is_file():
            self.call_statusbar("error", msg.Error.e02)
            return None, None, None, None

        if not h5py.is_hdf5(path_obj):
            self.call_statusbar("error", msg.Error.e48)
            return None, None, None, None

        # read the h5py files
        try:
            with h5py.File(path_obj, "r", ) as p:

                if 'TA Data/raw Data' in p:
                    rawdata_group = p['TA Data/raw Data']
                    rawdata = self._read_dict_from_hdf(group=rawdata_group)
                    rawdata['meta'] = {}
                    for k in p['TA Data'].attrs.keys():
                        rawdata['meta'][k] = p['TA Data'].attrs[k]

                    if ds != 'raw Data':
                        ds_dict = {}
                        if f'TA Data/{ds}' in p:

                            ds_group = p[f'TA Data/{ds}']
                            if ds_group.attrs:
                                ds_dict['meta'] = {attr: ds_group.attrs[attr]
                                                   for attr in ds_group.attrs}
                                ds_dict['wavelength'] = p[f'TA Data/{ds}/wavelength'][()]
                                ds_dict['delay'] = p[f'TA Data/{ds}/delay'][()]
                                ds_dict['delA'] = p[f'TA Data/{ds}/delA'][()]

                        if not ds_dict:  # ds set as input, but no data
                            self.call_statusbar("error", msg.Error.e38)
                            return None, None, None, None
                if not rawdata:
                    self.call_statusbar("error", msg.Error.e39)
                    return None, None, None, None

        except FileNotFoundError:
            self.call_statusbar("error", msg.Error.e03)
            return None, None, None, None
        
        except OSError:
            self.call_statusbar("error", msg.Error.e02)
            return None, None, None, None
        
        except KeyError:
            self.call_statusbar("error", msg.Error.e40)
            return None, None, None, None

        except Exception:
            logger.exception("unknown exception occurred")
            self.call_statusbar("error", msg.Error.e01)
            return None, None, None, None

        # write data and metadata to cache
        if self.sender().objectName() == 'pb_load_p1':
            self._init_p1()
            if ds == 'raw Data':
                self.p1_rawdata = rawdata
                self.p1_dsX, self.p1_dsY, self.p1_dsZ = rawdata[
                    'wavelength'], rawdata['delay'], rawdata['delA']

            else:
                self.p1_rawdata = {}
                self.p1_rawdata['meta'] = rawdata['meta']
                self.p1_rawdata['rawdata'] = rawdata
                self.p1_rawdata['processed'] = ds_dict
                self.p1_dsX, self.p1_dsY, self.p1_dsZ = ds_dict[
                    'wavelength'], ds_dict['delay'], ds_dict['delA']
            self.call_statusbar("info", msg.Status.s02)
            return np.nanmin(self.p1_dsX), np.nanmax(self.p1_dsX), np.nanmin(self.p1_dsY), np.nanmax(self.p1_dsY)

        elif self.sender().objectName() == 'pb_load_p2':
            self._init_p2()
            if ds == 'raw Data':
                self.p2_rawdata = rawdata
                self.p2_dsX, self.p2_dsY, self.p2_dsZ = rawdata[
                    'wavelength'], rawdata['delay'], rawdata['delA']

            else:
                self.p2_rawdata = {}
                self.p2_rawdata['meta'] = rawdata['meta']
                self.p2_rawdata['rawdata'] = rawdata
                self.p2_rawdata['processed'] = ds_dict
                self.p2_dsX, self.p2_dsY, self.p2_dsZ = ds_dict[
                    'wavelength'], ds_dict['delay'], ds_dict['delA']
            self.call_statusbar("info", msg.Status.s02)
            return np.nanmin(self.p2_dsX), np.nanmax(self.p2_dsX), np.nanmin(self.p2_dsY), np.nanmax(self.p2_dsY)

    def return_ds(self, project: str) -> tuple[NDArray, NDArray, NDArray]:
        ''' called by the view to return the cached data for plotting '''
        if project == 'p1':
            return self.p1_dsX, self.p1_dsY, self.p1_dsZ

        if project == 'p2':
            return self.p2_dsX, self.p2_dsY, self.p2_dsZ

        if project == 'combined':

            return self.x_new, self.y_new, self.z_new

    def _fill_nans(self, y: NDArray, z: NDArray) -> NDArray:
        """
        1) Extrapolate each ROW out to the full x-range with 1D interp
        2) For any still-missing points (e.g. if a full row was NaN),
           fill by nearest-neighbor via a distance transform.
        """
        z2 = z.copy()
        ny, nx = z2.shape

        # column-by-column extrapolation along y
        for j in range(nx):
            col = z2[:, j]
            mask = np.isnan(col)
            if mask.any():
                valid = ~mask
                if valid.sum() >= 2:
                    # np.interp will extrapolate outside [y_min, y_max]
                    z2[:, j] = np.interp(y, y[valid], col[valid])

        # nearest-neighbor via distance transform
        nan_mask = np.isnan(z2)
        if nan_mask.any():
            # distance_transform_edt gives, for each NaN location,
            # the indices of the nearest non-NaN cell
            _, inds = distance_transform_edt(nan_mask,
                                             return_distances=True,
                                             return_indices=True)
            z2[nan_mask] = z2[tuple(inds[:, nan_mask])]

        return z2

    def combine_ds(self, overlap_use: int, interpol_method: str, extrapol_method: str) -> None:
        '''
        Resample two cached 2-D datasets onto a common grid, resolve any spatial
        overlap, and optionally extrapolate into gaps.

        Parameters
        ----------
        overlap_use : {0, 1, 2}
            Policy for resolving duplicate grid points (see table above).
        interpol_method : str
            Interpolation kernel passed to ``scipy.interpolate.interpn``.
            GUI alias *“bicubic”* is internally converted to ``"splinef2d"``.
        extrapol_method : str
            ``"zero"`` to fill all remaining NaNs with 0; anything else delegates
            to 1D_interp.

        Returns
        -------
        None


        '''
        interpol_method = 'splinef2d' if interpol_method == 'bicubic' else interpol_method
        if self.p1_dsX is None or self.p2_dsX is None:
            self.x_new = self.y_new = self.z_new = None
            self.call_statusbar("error", msg.Error.e05)
            raise exc.NoDataError

        # unify axes
        x_new = np.union1d(self.p1_dsX, self.p2_dsX)
        y_new = np.union1d(self.p1_dsY, self.p2_dsY)

        # interpolate each dataset onto the new grid
        def interp(dsX, dsY, dsZ):
            pts = (dsY, dsX)
            Xg, Yg = np.meshgrid(x_new, y_new, indexing="xy")
            return interpn(pts, dsZ, (Yg, Xg),
                           method=interpol_method,
                           bounds_error=False,
                           fill_value=np.nan)

        p1z = interp(self.p1_dsX, self.p1_dsY, self.p1_dsZ)
        p2z = interp(self.p2_dsX, self.p2_dsY, self.p2_dsZ)

        # merge according to overlap_use
        if overlap_use == 0:
            z_new = np.where(~np.isnan(p1z), p1z, p2z)
        elif overlap_use == 1:
            z_new = np.where(~np.isnan(p2z), p2z, p1z)
        elif overlap_use == 2:
            z_new = np.nanmean(np.stack([p1z, p2z]), axis=0)

        if extrapol_method == 'zero':
            z_new = np.nan_to_num(z_new, nan=0)
        else:
            z_new = self._fill_nans(y_new, z_new)

        self.x_new, self.y_new, self.z_new = x_new, y_new, z_new
        self.call_statusbar("info", msg.Status.s28)
        if self.sender().objectName() == "pb_apply_combine_projects":
            self.apply_combined_ds()
            self.call_statusbar("info", msg.Status.s29)

    def apply_combined_ds(self) -> None:
        '''
        called by combine_ds when user pushes the apply button.
        changes the models rawdata according to cached datasets

        Returns
        -------
        None.

        '''
        rawdata = {}
        rawdata['wavelength'] = self.x_new
        rawdata['delay'] = self.y_new
        rawdata['delA'] = self.z_new
        rawdata['combined from'] = {}
        rawdata['combined from']['project 1'] = self.p1_rawdata
        rawdata['combined from']['project 2'] = self.p2_rawdata
        self.ta_model.rawdata = rawdata

    def delete_project(self, sender: str):
        ''' deletes cached project data '''
        if sender == 'p1':
            self.p1_dsX, self.p1_dsY, self.p1_dsZ = None, None, None
        else:
            self.p2_dsX, self.p2_dsY, self.p2_dsZ = None, None, None
        self.call_statusbar("info", msg.Status.s03)

    def call_statusbar(self, level, message):
        self.status_signal.emit(level, message)
