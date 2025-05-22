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
from scipy.linalg import svd
from ..configurations import messages as msg

logger = logging.getLogger(__name__)


class ComponentController(QObject):

    def __init__(self,  ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()

        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.U, self.s, self.Vh = None, None, None
    status_signal = pyqtSignal(str, str)

    def _get_ds_model(self, ds: str) -> object:
        ''' helper that returns the model object of the name ds '''
        if ds == '1':
            return self.ta_model_ds1
        if ds == '2':
            return self.ta_model_ds2
        if ds == '3':
            return self.ta_model_ds3

    def verify_rawdata(self) -> bool:
        ''' checks if rawdata is set in the model '''
        if not self.ta_model.rawdata:
            self.call_statusbar("error", msg.Error.e05)
            return False
        else:
            return True

    def get_data(self,  ds: str) -> tuple[NDArray, NDArray, NDArray]:
        ''' returns the data stored in a given dataset ds, or returns the rawdata if ds is empty '''
        ds_model = self._get_ds_model(ds)
        if ds_model.dsZ is None:
            return self.ta_model.rawdata['wavelength'], self.ta_model.rawdata['delay'], self.ta_model.rawdata['delA']

        else:
            return ds_model.dsX, ds_model.dsY, ds_model.dsZ

    def calculate_svd(self, ds: str) -> None:
        '''
        Compute the singular value decomposition (SVD) of the dataset and store the factors.
        left singular vectors (U), singular values (s), right singular vectors (Vh)
        Parameters
        ----------
        ds : str
            current dataset.

        Returns
        -------
        None
            The results are stored in `self.U`, `self.s`, and `self.Vh`.

        '''
        if not self.verify_rawdata():
            return

        _, _, Z = self.get_data(ds=ds)

        self.U, self.s, self.Vh = svd(Z, full_matrices=False)

    def get_svd(self, ds: str, components: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        '''
        Retrieve the leading SVD components and reconstructed approximation.

        Parameters
        ----------
        ds : str
            current dataset.
        components : int
            Number of leading singular components to return.

        Returns
        -------
        U_reduced : NDArray
            Matrix of shape (n_samples, components) containing the top left singular vectors.
        s_normalized : NDArray
            Array of length `components` giving the singular values normalized by the
            largest singular value.
        Vh_reduced : NDArray
            Matrix of shape (components, n_features) containing the top right singular vectors.
        Z_approx : NDArray
            Rank-`components` approximation of the original data matrix, i.e.
            `U_reduced @ diag(s[:components]) @ Vh_reduced`.

        '''

        if self.U is None:
            self.calculate_svd(ds)
        s_norm = self.s/self.s.max()
        s_norm = s_norm[:components]
        U = self.U[:, :components]
        Vh = self.Vh[:components, :]
        S = np.diag(self.s[:components])
        Z_calc = U @ (S@Vh)
        return U, s_norm, Vh, Z_calc

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)
