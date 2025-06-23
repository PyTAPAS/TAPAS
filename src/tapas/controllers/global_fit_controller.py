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

# Standard Library Imports
import copy
import datetime
import logging
import time
import math
from typing import Callable

# Third‑Party Imports
from numpy.fft import fftn, ifftn
# PyQt6 Imports
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# Numerical and Scientific Libraries
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from jax import jit, jacfwd
from scipy.linalg import svd
from scipy.interpolate import make_interp_spline

# Other Third‑Party Imports
import emcee.autocorr
import matplotlib.ticker as tk
from lmfit import Minimizer, Parameters
from PyQt6.QtWidgets import QWidget
from emcee.autocorr import AutocorrError

# Local Application Imports
from ..configurations import messages as msg, exceptions as exc
from ..utils import utils, model_functions as ModelFunctions


logger = logging.getLogger(__name__)


class EmceeWorker(QThread):
    ''' Worker that performs the computationally heavy emcee analysis'''
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, bool)
    results_signal = pyqtSignal(object, int)

    def __init__(self, results: dict, delay: NDArray, delA: NDArray, burn: int, init: int, thin: int, target_ratio: int):
        super().__init__()

        self.results = results
        self.delay = delay
        self.delA = delA
        self.burn = burn  # number of discarded samples from the start
        self.init = init  # number of initial samples drawn from each walker
        self.thin = thin  # accept only one sample in every x samples
        # the chain should be longer than {target_ratio} times the integrated autocorr time
        self.target_ratio = target_ratio
        self._abort = False

    def abort(self) -> None:
        ''' triggered if user wants to abort the analysis'''
        self._abort = True

    def run(self) -> None:
        ''' perform the emcee analysis

        1) the minimizer object is created with the fitting data.
        2) #burn steps will be performed to find a good starting point.
        3) a first round with #init steps is performed to estimate the needed time
           and number of samples for reliable estimates depending on the set target ratio.
        4) the number of estimated samples will be run and added to the first round
        5) if the target ratio isn't met yet, another round will be performed'

        see https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee
        and https://emcee.readthedocs.io/en/stable/
        for more information

        '''

        # -------- minimizer object ----------------------------------------------------------------
        minimizer = Minimizer(
            GlobalFitController.model_theta_log_posterior_wrapper,
            self.results['opt_params'],
            nan_policy='omit',
            float_behavior='posterior',      # tells lmfit to forward scalar as log‑post
            fcn_kws={
                'delay': self.delay,
                'delA': self.delA,
                **self.results.get('meta', {}),
                'weights': jnp.array(self.results.get('weight_vector', jnp.array([1]))),
            }
        )

        info = (
            f'lets start with warming up and take {self.burn} steps to find '
            'a good equilibrium and to estimate the needed time\nthis might '
            'already take a few minutes...\n')
        self.progress.emit(info)

        # -------- warmup --------------------------------------------------------------------------
        t_start = time.time()
        warm_up = minimizer.emcee(


            burn=0,
            steps=self.burn,
            thin=1,
            is_weighted=False,
            progress=False,
            workers=1,  # already parallelized with jax/jit
            reuse_sampler=False
        )
        t_end = time.time()
        time_needed = (t_end - t_start)/60
        time_per_step = time_needed / self.burn
        last_positions = warm_up.chain[-1].copy()  # shape: (nwalkers, nvars)
        info = f"""\
        Warmup finished successfully. This took {time_needed:.1f} min.

        The expected time per unthinned step is {(time_per_step*60):.2f} s.
        Lets do a first "quick" exploration with {self.init} samples ({self.init*self.thin} unthinned steps)
        which might take about {(time_per_step*self.init*self.thin):.1f} min to get an idea
        of the parameter space and to refine the time estimates…
        """

        self.progress.emit(info)
        if self._abort:
            self.finished.emit(None, True)
            return

        # -------- init round ----------------------------------------------------------------------
        t_start = time.time()
        init_emcee = minimizer.emcee(
            burn=0,          # no additional burn-in
            steps=self.init*self.thin,
            thin=self.thin,
            is_weighted=False,
            progress=False,
            workers=1,
            pos=last_positions,   # use last_positions as the initial positions for each walker
            reuse_sampler=False   # start with a fresh sampler to avoid issues with reuse_sampler
        )
        t_end = time.time()
        self.results_signal.emit(init_emcee, 1)
        init_time = (t_end - t_start)/60
        tau = emcee.autocorr.integrated_time(init_emcee.chain, tol=0)
        max_tau = np.max(tau)
        samples_total = init_emcee.chain.shape[0]
        if samples_total > self.target_ratio * max_tau:
            info += ('target ratio already satisfied')
            self.progress.emit(info)

            self.finished.emit(init_emcee, False)
            return

        # -------- 2nd round -----------------------------------------------------------------------
        required_samples = int(self.target_ratio * max_tau - samples_total)
        time_per_samle = init_time/samples_total
        additional_time = (time_per_samle * required_samples)
        info = (
            f'took {init_time:.1f} min, we expected {(time_per_step*self.init*self.thin):.1f} min\n'
        )
        info += (
            f'adjusted time per unthinned step is {(init_time/(self.init*self.thin)*60):.2f} s\n'
        )
        info += (f"We might need at least {required_samples} additional samples "
                 f"({required_samples*self.thin} unthinned steps) to fulfill the target ratio, "
                 f"which will take approx {additional_time:.1f} min\n")
        self.progress.emit(info)

        t_start = time.time()
        if self._abort:
            self.finished.emit(init_emcee, True)
            return

        emcee_results = minimizer.emcee(

            burn=0,
            steps=required_samples*self.thin,
            thin=self.thin,
            is_weighted=False,
            progress=False,
            workers=1,
            reuse_sampler=True
        )
        self.results_signal.emit(emcee_results, 2)
        t_end = time.time()
        second_runtime = (t_end - t_start)/60
        tau = emcee.autocorr.integrated_time(emcee_results.chain, tol=0)
        max_tau = np.max(tau)
        time_per_step = (init_time + second_runtime) / \
            (emcee_results.chain.shape[0]+self.burn)
        samples_total_2 = emcee_results.chain.shape[0]
        info = f'ok, so now we have performed {samples_total_2} samples\n'
        info += (f'this took {second_runtime:.1f} min, we expected {additional_time:.1f} min\n')

        tau = emcee.autocorr.integrated_time(emcee_results.chain, tol=0)
        max_tau = np.max(tau)
        required_samples2 = int(self.target_ratio * max_tau - samples_total_2)
        if emcee_results.chain.shape[0] > self.target_ratio * max_tau:
            info += ('looks good now...')
            self.progress.emit(info)
            self.finished.emit(emcee_results, False)
            return
        else:
            info += (
                f'still not finished.. we might need {required_samples2} more samples '
                f' altough we have already {samples_total_2}. Lets try one more time '
                f'({time_per_samle*required_samples2:.1f} min)...')
            self.progress.emit(info)

        # -------- 3rd round -----------------------------------------------------------------------
        t_start = time.time()
        if self._abort:
            self.finished.emit(emcee_results, True)
            return
        emcee_results = minimizer.emcee(
            burn=0,
            steps=required_samples2,
            thin=self.thin,
            is_weighted=False,
            progress=False,
            workers=1,
            reuse_sampler=True)

        self.results_signal.emit(emcee_results, 3)
        t_end = time.time()
        third_runtime = (t_end - t_start)/60
        tau = emcee.autocorr.integrated_time(emcee_results.chain, tol=0)
        max_tau = np.max(tau)
        time_per_step = (init_time + third_runtime) / \
            (emcee_results.chain.shape[0]+self.burn)
        samples_total_3 = emcee_results.chain.shape[0]
        required_samples3 = int(self.target_ratio * max_tau - samples_total_3)
        info = (f'ok, so now we have performed {samples_total_3} samples\n')
        if emcee_results.chain.shape[0] > self.target_ratio * max_tau:
            info += ('looks good now')
            self.progress.emit(info)
            self.finished.emit(emcee_results, False)
            return
        else:
            info += (
                f'still not finished.. we might need {required_samples3} more '
                f'samples altough we have already {samples_total_3}. Lets quit, should be enough...')
            self.progress.emit(info)

            self.finished.emit(emcee_results, True)


class GlobalFitController(QObject):

    status_signal = pyqtSignal(str, str)
    worker_progress = pyqtSignal(str)
    worker_results = pyqtSignal(object, int)
    emcee_finished = pyqtSignal(object, bool)

    def __init__(self, abs_model, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__()
        self.abs_model = abs_model
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3

        def _jacobian_model_wrapper(raw_theta: NDArray, delay: NDArray, delA: NDArray, Ainf: bool,
                                    model: str, weights: NDArray, use_threshold_t0: bool,
                                    substeps: int = 10, gs: bool = False, use_bleach=False,
                                    gs_spec=False, ca_order: int = 0, output: bool = False, **kwargs) -> "model function":
            '''
            wrapper for the jacobian calculation. transforms t0 for the XLA model fitting and returns model function

            Parameters
            ----------
            raw_theta : NDArray
                array that holds the fitted parameter values [t0, IRF, t1, ..tn].
            delay : NDArray
                delay time vector.
            delA : NDArray
                2D delA (delay x wavelength).
            Ainf : bool
                Whether to include an infinite-time offset pool.
            model : str
                used fitting model.
            weights : NDArray
                weights vector, length wavelength, values between 0 and 1.
            use_threshold_t0 : bool
                if ture, the t0 parameter input referres to the 5% rising of the signal.
                else the mean of the IRF centroid. Internally, the centroid is used for fitting
            substeps : int, optional
                Number of RK4 sub-steps to take within each interval `delay[i] → delay[i+1]`.
                Higher values improve integration accuracy (smaller local error per interval)
                at the cost of proportional extra compute. Defaults to 10.
            gs : bool, optional
                Whether to include an explicit ground-state component in the fit.
                The default is False.
            use_bleach : TYPE, optional
                Whether to model an additional bleach component.
            gs_spec : TYPE, optional
                If `gs` and `use_bleach`, whether to use external abs spectrum for GS.
                The default is False.
            output : bool, optional
                If True, return `(delA_cal, c, eps.T)` instead of the flattened residuals.
                The default is False.

            Returns
            -------
            "model function"
                objective function of minimization problem

            '''

            taus = raw_theta[2:]
            t0_raw = raw_theta[0]
            fwhm = raw_theta[1]

            if use_threshold_t0:
                sigma = fwhm / 2.35482
                # t(5%) = t0(center) + sqrt(2) sigma erf^-1(2*5%-1)
                k05 = -1.6448536269514729  # = sqrt(2) sigma erf^-1(2*5%-1)
                t0_centroid = t0_raw - k05 * sigma
            else:
                t0_centroid = t0_raw
            theta = jnp.concatenate([jnp.array([t0_centroid, fwhm]), taus])
            model_function = self._get_model_function(model=model)

            if model == 'parallel':
                return model_function(
                    theta=theta, delay=delay, delA=delA, Ainf=Ainf, weights=weights, gs=gs,
                    use_bleach=use_bleach, gs_spec=gs_spec, ca_order=ca_order,  output=output)

            else:
                return model_function(
                    theta=theta, delay=delay, delA=delA, Ainf=Ainf, weights=weights,
                    substeps=substeps, gs=gs, use_bleach=use_bleach, gs_spec=gs_spec, ca_order=ca_order, output=output)

        self.jacobian_func = jit(jacfwd(
            _jacobian_model_wrapper, argnums=0), static_argnames=('Ainf', 'model', 'use_threshold_t0',
                                                                  'substeps', 'gs', 'use_bleach', 'ca_order', 'output'))

    @staticmethod
    def model_theta_log_posterior_wrapper(params: Parameters,
                                          delay: NDArray,
                                          delA: NDArray,
                                          Ainf: bool,
                                          model: str,
                                          weights: NDArray,
                                          use_threshold_t0: bool,
                                          substeps: int = 6,  gs=False, gs_spec=False, ca_order=0,
                                          n_eff: int | float = 1.0, **extra) -> float:
        '''
        Return scalar *log‑posterior* for lmfit–emcee (float_behavior='posterior')

        Parameters
        ----------
        params : Parameters
            lmfit Parameters object.
        delay : NDArray
            1D delay time vector.
        delA : NDArray
            2D delA values.
        Ainf : bool
            if true, a non-decaying component is added.
        model : str
            used fitting model.
        weights : NDArray
            weights vector, length wavelength, values between 0 and 1.
        use_threshold_t0 : bool
            if ture, the t0 parameter input referres to the 5% rising of the signal.
            else the mean of the IRF centroid. Internally, the centroid is used for fitting
        substeps : int, optional
            each experimental delay bin is subdivided into n microsteps for the RK-4 method.
            The default is 10. Not used for the parallel model
        gs : bool, optional
            Whether to include an explicit ground-state component in the fit.
            The default is False.
        gs_spec : TYPE, optional
            If `gs` and `use_bleach`, whether to use external abs spectrum for GS.
            The default is False.
        n_eff : int | float, optional
            effective sample size taking IAC of the residuals into account using 
            n_eff = residuals.size / tau

        Returns
        -------
        float
            Gaussian log‑likelihood with effective sample size.

        '''

        # -------- init parameters -----------------------------------------------------------------
        for par in params.values():
            if (par.min is not None and par.value < par.min) or \
               (par.max is not None and par.value > par.max):
                return -jnp.inf                # zero probability outside bounds

        sigma = jnp.exp(params['__lnsigma'].value)   # lmfit ensures parameter exists

        # -------- built XLA compartible theta and calculate residuals -----------------------------
        theta = GlobalFitController._params_to_theta(params, use_threshold_t0, gs_spec)
        model_function = GlobalFitController._get_model_function(model)
        use_bleach = True if gs_spec is not False else False
        if model == 'parallel':
            resid = model_function(theta, delay, delA, Ainf, weights,
                                   gs, use_bleach, gs_spec, ca_order=ca_order, output=False)
        else:
            resid = model_function(theta, delay, delA, Ainf, weights,
                                   substeps,  gs, use_bleach, gs_spec, ca_order=ca_order,  output=False)

        # -------- calculate Gaussian log-likelihood with effective sample size --------------------
        ssr = jnp.dot(resid, resid)
        loglike = -0.5 * ssr / sigma**2 - n_eff * jnp.log(sigma)

        return float(loglike)

    def abort_emcee(self) -> None:
        ''' called by the view if user wants to abort the emcee analysis and informs the worker '''
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.call_statusbar("info", msg.Status.s35)
            info = 'calculation will be aborted after the next run. Data will be saved...'
            self.worker_progress.emit(info)
            self.worker.abort()

    def _redirect_worker_info(self, info: str) -> None:
        ''' redirects the emcee worker signal to the view '''
        self.worker_progress.emit(info)

    def _redirect_worker_results(self, results: object, run: int):
        ''' redirects the emcee worker signal (results: lmfit.Results, run: number of performed emcee cycle) to the view '''
        self.worker_results.emit(results, run)

    def _finish_emcee(self, emcee_results: object, abort: bool) -> None:
        ''' redirects the emcee worker signal (results: lmfit.Results, abort: True if finished earlier) to the view '''
        if emcee_results is None:
            self.emcee_finished.emit(emcee_results, abort)
            return

        self.emcee_finished.emit(emcee_results, abort)

    def initialize_fitting_cache(self) -> None:
        ''' clears the cached data '''
        self.current_wavelength = None
        self.current_delay = None
        self.current_delA = None
        self.current_fit_results = None

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

    def verify_abs_data(self) -> bool:
        ''' checks if absdata is set in the model '''
        if self.abs_model.rawdata_before is not None:
            return True
        else:
            return False

    def verify_regular_grid(self, vector: NDArray) -> bool:
        ''' checks if vector is evenly spaced / forms a regular grid '''
        if vector.ndim != 1 or vector.size < 2:
            return False
        d = np.diff(vector)

        return np.allclose(d, d[0], rtol=0.05, atol=0.05e-9)

    def get_abs_data(self) -> NDArray | bool:
        ''' returns the absorbance data array or false if not present '''
        if self.verify_abs_data():
            return self.abs_model.rawdata_before
        else:
            False

    def save_current_fit(self, ds: str) -> None:
        ''' triggered by the view when pb is pressed. saves the fit to the model '''
        ds_model = self._get_ds_model(ds)

        if not self.verify_rawdata():
            self.call_statusbar("error", msg.Error.e05)
            return
        if self.current_fit_results is None:
            self.call_statusbar('error', msg.Error.e21)
            return

        # create unique key:
        base = 'global'
        ukey = base
        suffix = 1
        while ukey in ds_model.global_fit:
            ukey = f"{base}_{suffix}"
            suffix += 1

        # self.current_fit_results.pop('residuals', None)
        self.current_fit_results.pop('unweighted_residuals', None)

        ds_model.update_global_fit(ukey, self.current_fit_results)
        self.call_statusbar('info', msg.Status.s24)

    def get_fitting_meta(self, fit_results: dict) -> str:
        ''' called by the view to retrive fitting metadata '''
        timestamp = datetime.datetime.now().strftime("%y%m%d")
        exp_meta = self.ta_model.metadata
        title = ''
        if exp_meta['sample'] != '':
            title += exp_meta['sample'] + '  |  '
        if exp_meta['excitation wavelength'] != '':
            title += exp_meta['excitation wavelength'] + '  |  '
        if exp_meta['excitation intensity'] != '':
            title += exp_meta['excitation intensity'] + '  |  '
        if exp_meta['solvent'] != '':
            title += exp_meta['solvent'] + '  |  '  # meta = ''
        fit_meta = ''

        return f'Date: {timestamp}\nExperimental conditions: {title}\n{fit_meta}'

    def get_SVD(self, residuals: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        ''' Retrieve and return the three leading SVD components of the residuals
        left singular vectors (U), singular values (s), right singular vectors (Vh)'''
        components = 3

        U, s, Vh = svd(residuals, full_matrices=False)

        s_norm = s/s.max()
        s_norm = s_norm[:components]
        U = U[:, :components]
        Vh = Vh[:components, :]

        return U, s_norm, Vh

    def get_fit(self, ds: str, selected_fit: int) -> tuple[dict, str]:
        ''' gets and returns the fit dict and the name of the selected fit and dataset in the gui'''
        ds_model = self._get_ds_model(ds)

        ukey = list(ds_model.global_fit)[selected_fit]

        return ds_model.global_fit[ukey], ukey

    def delete_fit(self, ds: str, selected_fit: int) -> None:
        ''' delets selected fit from the model '''
        ds_model = self._get_ds_model(ds)
        ukey = list(ds_model.global_fit)[selected_fit]

        ds_model.del_global_fit_key(ukey)

    def get_global_fit_list(self, ds: str) -> dict:
        ''' returns complete global fit nested dict '''
        ds_model = self._get_ds_model(ds)
        return ds_model.global_fit

    def create_weight_vector(self, wavelength: NDArray, intervals: tuple[float, float, float]) -> NDArray:
        """
        Create a weight vector for a given wavelength array and a list of intervals.

        Parameters:
            wavelength (np.ndarray): 1D array of wavelength values in SI.
            intervals (list of tuples): Each tuple is (low_val, high_val, weight) in nm


        Returns:
            np.ndarray: A weight vector of the same length as the wavelength array.
        """
        wavelength = wavelength*1e9  # intervals uses nanometer values (int)

        # Initialize the weight vector with the default weight.
        weights = np.full(wavelength.shape, 1, dtype=float)
        current_max = -np.inf  # Track the highest wavelength already covered.
        for low_val, high_val, w in intervals:

            if high_val <= current_max:  # ignores empty intervals from spinboxes
                continue

            idx_low = np.abs(wavelength - low_val).argmin()
            idx_high = np.abs(wavelength - high_val).argmin()

            # Ensure the lower index comes first, should be always the case though
            if idx_low > idx_high:
                idx_low, idx_high = idx_high, idx_low

            weights[idx_low: idx_high + 1] = w
            current_max = high_val
        return weights

    def create_ds(self, ds: str) -> None:
        ''' fill ds with rawdata if empty '''
        ds_model = self._get_ds_model(ds)
        if ds_model.dsZ is None:
            ds_model.dsX = self.ta_model.rawdata['wavelength']
            ds_model.dsY = self.ta_model.rawdata['delay']
            ds_model.dsZ = self.ta_model.rawdata['delA']

    def get_params(self, QtTable: QWidget) -> Parameters:
        '''
        dynamically reads the parameter table inputs, dumps them into a  nested dict and writes it to a lm paramter object

        Returns
        -------
        p : lm Parameters Object
            Parameters object used for fitting with init values, bounds and vary bools.

        '''
        params = {}
        for row in range(QtTable.rowCount()):
            params[str(QtTable.verticalHeaderItem(row).text())] = {}
            for column in range(4):
                # dict["key"] = value
                try:
                    params[str(QtTable.verticalHeaderItem(row).text())][str(QtTable.horizontalHeaderItem(
                        column).text())] = utils.Converter.convert_str_input2float(QtTable.item(row, column).text())
                except (AttributeError, ValueError):
                    params[str(QtTable.verticalHeaderItem(row).text())][str(
                        QtTable.horizontalHeaderItem(column).text())] = None

        # sets the default vary parameter to be true:
        for i in list(params):
            if params[i]['vary'] is None:
                params[i]['vary'] = True
        if params['IRF']['min'] is None:
            params['IRF']['min'] = 1e-15

        p = Parameters()
        for param in params:
            key = params[param]
            try:
                p.add(param, value=key["value"], min=key["min"],
                      max=key["max"], vary=key["vary"])
            except ValueError:
                self.call_statusbar('error', msg.Error.e29)
                raise
        return p

    @staticmethod
    def _get_model_function(model: str) -> Callable:
        ''' Returns the ModelFunctions.ModelFunctions.model_<model> function. '''
        cls = ModelFunctions.ModelFunctions
        fn_name = f"model_{model}"
        return getattr(cls, fn_name)

    def get_data(self, ds: str) -> tuple[NDArray, NDArray, NDArray]:
        ''' returns the data stored in a given dataset ds, or returns the rawdata if ds is empty '''
        if not self.verify_rawdata():
            self.call_statusbar("error", msg.Error.e05)
            raise ValueError

        ds_model = self._get_ds_model(ds)
        if ds_model.dsZ is None:
            buffer_dataX = self.ta_model.rawdata['wavelength']
            buffer_dataY = self.ta_model.rawdata['delay']
            buffer_dataZ = self.ta_model.rawdata['delA']
        else:
            buffer_dataX = ds_model.dsX
            buffer_dataY = ds_model.dsY
            buffer_dataZ = ds_model.dsZ

        return buffer_dataX, buffer_dataY, buffer_dataZ

    def get_gs_spec(self, wavelength: np.ndarray) -> bool | np.ndarray:
        '''
        Interpolate and normalize steady-state absorption data at requested wavelengths.

        Parameters
        ----------
        wavelength : np.ndarray
            1D array of wavelengths at which to evaluate the steady-state absorption spectrum.

        Returns
        -------
        False or np.ndarray
            - Returns False if no absorption data is available.
            - Otherwise returns an (N, 2) float32 array where the first column is the
              input wavelengths and the second column is the normalized absorbance:
                1. Linearly interpolated (and extrapolated) from the original data.
                2. Clamped to the endpoint values outside the original range.
                3. Baseline-subtracted and scaled to a maximum of 1.

        Notes
        -----
        - Original absorbance data is fetched via `self.get_abs_data()` and expected
          as an (M, 2) array of [wavelength, absorbance].
        - If there is no overlap between `wavelength` and the original data range,
          the returned absorbance column will be all zeros.
        '''

        ss_abs = self.get_abs_data()
        if ss_abs is False or ss_abs is None:
            return False

        orig_wl, orig_abs = ss_abs[:, 0], ss_abs[:, 1]
        idx = np.argsort(orig_wl)
        sorted_wl = orig_wl[idx]
        sorted_abs = orig_abs[idx]

        spline = make_interp_spline(sorted_wl, sorted_abs, k=1)

        lam = wavelength.astype(float)
        y = spline(lam)  # may extrapolate linearly outside [min,max]

        lo, hi = sorted_wl[0], sorted_wl[-1]
        inside = (lam >= lo) & (lam <= hi)  # inside mask
        if not inside.any():
            return np.column_stack((lam.astype(np.float32),
                                    np.zeros_like(lam, np.float32)))

        # endpoint values from the *inside* region
        left_val = y[inside][0]
        right_val = y[inside][-1]

        # clamp tails only where they exist
        left_mask = lam < lo
        right_mask = lam > hi

        if left_mask.any():
            y[left_mask] = left_val
        if right_mask.any():
            y[right_mask] = right_val

        # baseline‐subtract & unit‐peak normalize
        y_min = y.min()
        y -= y_min
        y_max = y.max()
        if y_max > 0:
            y /= y_max

        return np.column_stack((lam.astype(np.float32),
                                y  .astype(np.float32)))

    @staticmethod
    def _params_to_theta(params: Parameters, use_threshold_t0: bool, gs_spec: bool | NDArray = False) -> NDArray:
        '''
        converts Parameters object to jax array for the XLA model fitting and returns it as "theta"

        Parameters
        ----------
        params : Parameters
            lmfit Parameters object.
        use_threshold_t0 : bool
            if ture, the t0 parameter input referres to the 5% rising of the signal.
            else the mean of the IRF centroid. Internally, the centroid is used for fitting
        gs_spec : TYPE, optional
            Whether to use an external spectra to model the ground-state. The default is False.

        Returns
        -------
        NDArray
            theta: jax compartiple parameters array [t0, IRF, t1, ...tn].

        '''
        # -------- init parameters -----------------------------------------------------------------
        t0_raw = params['t0'].value
        fwhm = params['IRF'].value
        if gs_spec is not False:
            gs_sigma = params['gs_sigma'].value
            gs_shift = params['gs_shift'].value
        if use_threshold_t0:
            sigma = fwhm / 2.35482
            # t(5%) = t0(center) + sqrt(2) sigma erf^-1(2*5%-1)
            k05 = -1.6448536269514729  # = sqrt(2) sigma erf^-1(2*5%-1)
            t0_centroid = t0_raw - sigma * k05
        else:
            t0_centroid = t0_raw

        # -------- return sorted theta vector ------------------------------------------------------
        if '__lnsigma' not in params:
            # Define the ordering: t1, t2, ..., t_components, then t0 and IRF.
            if gs_spec is False:
                return jnp.array([t0_centroid, fwhm] +
                                 [params[f'τ{i}'].value for i in range(1, len(params)-1)])
            else:
                return jnp.array([t0_centroid, fwhm] +
                                 [params[f'τ{i}'].value for i in range(1, len(params)-3)] +
                                 [gs_shift, gs_sigma])

        else:
            if gs_spec is False:
                return jnp.array([t0_centroid, fwhm] +
                                 [params[f'τ{i}'].value for i in range(1, len(params)-2)])
            else:
                return jnp.array([t0_centroid, fwhm] +
                                 [params[f'τ{i}'].value for i in range(1, len(params)-4)] +
                                 [gs_shift, gs_sigma])

    @staticmethod
    def _scale_from_guess(params: Parameters) -> NDArray:
        """return one scale factor for every varying parameter. Default is 1e-12.
        Helps the solvers to converge over large time domains"""
        scale = []
        for p in params.values():
            if not p.vary:
                continue

            v = abs(p.value)

            if v == 0 or not np.isfinite(v):
                check_values = []
                if np.isfinite(p.min):
                    check_values.append(abs(p.min))
                if np.isfinite(p.max):
                    check_values.append(abs(p.max))
                v = max(check_values) if check_values else 1e-12

            v_rounded = 10.0 ** round(np.log10(v))
            v_rounded = np.clip(v_rounded, 1e-15, 1e-3)

            scale.append(v_rounded)

        return np.asarray(scale, dtype=float)

    @staticmethod
    def model_theta_wrapper(params: Parameters, delay: NDArray, delA: NDArray, Ainf: bool,
                            model: str, weights: NDArray, use_threshold_t0: bool,
                            substeps: int = 10,  gs=False, gs_spec=False, ca_order: int = 0,
                            output: bool = False,) -> NDArray | tuple[NDArray, NDArray, NDArray]:
        '''
        wrapps lmfit parameters object and model function to XLA compartiple code.

        Parameters
        ----------
        params : Parameters
            lmfit Parameters object.
        delay : NDArray
            1D delay time vector.
        delA : NDArray
            2D delA values.
        Ainf : bool
            if true, a non-decaying component is added.
        model : str
            used fitting model.
        weights : NDArray
            weights vector, length wavelength, values between 0 and 1.
        use_threshold_t0 : bool
            if ture, the t0 parameter input referres to the 5% rising of the signal.
            else the mean of the IRF centroid. Internally, the centroid is used for fitting
        substeps : int, optional
            each experimental delay bin is subdivided into n microsteps for the RK-4 method.
            The default is 10. Not used for the parallel model
        output : bool, optional
            if true, delA_cal, c, eps.T  will be returned, else only the residuals are returned.
            The default is False.

        Returns
        -------
        residuals : 1D NDArray
            residuals vector used by the external solver to be minimized (output = False).
        delA calculated, concentration matrix, Amplitude : tuple[NDArray, NDArray, NDArray]
            resulting simulated matrixes  (output = True).

        '''

        # Define the ordering: t1, t2, ..., t_components, then t0 and IRF.
        theta = GlobalFitController._params_to_theta(params, use_threshold_t0, gs_spec)

        model_function = GlobalFitController._get_model_function(model=model)
        use_bleach = True if gs_spec is not False else False
        if model == 'parallel':
            return model_function(
                theta, delay, delA, Ainf, weights, gs, use_bleach, gs_spec, ca_order, output)

        else:
            return model_function(
                theta, delay, delA, Ainf, weights, substeps, gs, use_bleach, gs_spec, ca_order, output)

    def optimize_params(
            self, params: Parameters, ds: str, Ainf: bool,  model: str, method: str,
            weights: NDArray, use_threshold_t0: bool, substeps: int = 10,
            gs: bool = False, gs_spec: bool | NDArray = False, ca_order: int = 0) -> dict:
        '''
        performs the minimization of the objective model function and returns a dict with the
        optimized parameters and metadata

        uses lmfits Minimizer class. Since the objective function uses jax and jit for XLA, the
        lmfit Parameters object is transformed to an array "theta" using the model_theta_wrapper

        Parameters
        ----------
        params : Parameters
            holds the init values, bounds, vary bools.
        ds : str
            name of the current dataset.
        Ainf : bool
            if true, an infinite non-decaying component is added.
        model : str
            name of the model function: parallel, sequential or target model nC_mk_p.
        method : str
            name of the fitting method used by lmfit and scipy.
        weights : NDArray
            weightsvector weightening the wavelengths: resid = (delA - delA_cal) * jnp.sqrt(weights)
        use_threshold_t0 : bool
            if true t0 parameter represents 5% of the rising IRF, if False uses the Guassian centroid.
            internally t0 always represents the maximum of the IRF.
        substeps : int, optional
            Number of RK4 sub-steps to take within each interval `delay[i] → delay[i+1]`.
            Higher values improve integration accuracy (smaller local error per interval)
            at the cost of proportional extra compute. Defaults to 10. Not used for parallel model
        gs : bool, optional
            Whether to include an explicit ground-state component in the fit.
        gs_spec : TYPE, optional
            Whether to use an external spectra to model the ground-state. The default is False.

        Raises
        ------
        exc.FittingError()
            raised if fit fails

        Returns
        -------
        fit_results : dict
            results dict containing optimized parameters and additional meta data.

        '''
        # -------- create Minimizer instance -------------------------------------------------------
        wavelength, delay, delA = self.get_data(ds)
        if gs_spec is not False:
            params.add('gs_shift', value=1e-9, min=-5e-9, max=5e-9)
            params.add('gs_sigma', value=1e-9, min=0, max=5e-9)

        minner = Minimizer(self.model_theta_wrapper, params,  fcn_kws={
                           'delay': delay, 'delA': delA,  'Ainf': Ainf,  'model': model,
                           'weights': jnp.array(weights), 'use_threshold_t0': use_threshold_t0,
                           'substeps': substeps, 'gs': gs, 'gs_spec': gs_spec, 'ca_order': ca_order, 'output': False, })

        # -------- perform the minimization --------------------------------------------------------
        try:
            t_start = time.time()
            if method == "diff-evol":
                lm_fit_results = minner.minimize(
                    method="differential_evolution", workers=1)
            elif method == "leastsq":
                lm_fit_results = minner.minimize(
                    method="leastsq", diag=self._scale_from_guess(params))
            else:
                lm_fit_results = minner.minimize(method=method)
            t_end = time.time()
            fit_time = t_end - t_start
        except Exception as err:
            err_msg = str(err)
            if "NaN values detected" in err_msg:
                err_msg = msg.Error.e44
            self.call_statusbar("error", (f"Optimization failed: {err_msg}"))
            raise exc.FittingError()

        # -------- create results dict  ------------------------------------------------------------
        fit_results = {}
        fit_results['meta'] = {}
        fit_results['opt_params'] = lm_fit_results.params
        fit_results['theta'] = self._params_to_theta(
            fit_results['opt_params'], use_threshold_t0, gs_spec)
        fit_results['meta']['Ainf'] = Ainf
        fit_results['meta']['model'] = model
        fit_results['meta']['fit_time'] = fit_time
        fit_results['meta']['#eval'] = lm_fit_results.nfev
        fit_results['residuals'] = lm_fit_results.residual
        fit_results['meta']['method'] = lm_fit_results.method
        fit_results['weight_vector'] = weights
        fit_results['meta']['substeps'] = substeps
        fit_results['meta']['gs'] = gs
        fit_results['meta']['gs_spec'] = gs_spec
        fit_results['meta']['ca_order'] = ca_order
        fit_results['meta']['time_zero_convention'] = '5% threshold' if use_threshold_t0 else 'Gaussian centroid'
        fit_results['meta']['use_threshold_t0'] = use_threshold_t0
        num_comp = (len(fit_results['theta']) -
                    2) if gs_spec is False else (len(fit_results['theta'])-4)
        labels = self.get_component_labels(
            model=fit_results['meta']['model'], Ainf=fit_results['meta']['Ainf'], num=num_comp, gs=gs, ca_order=fit_results['meta']['ca_order'])
        fit_results['meta']['components'] = labels

        # -------- cache data for subsequent fit statistics  ---------------------------------------
        self.current_wavelength = wavelength
        self.current_delay = delay
        self.current_delA = delA

        return fit_results

    def get_component_labels(self, model: str, Ainf: bool, num: int, gs: bool = False, ca_order: int = 0, local : bool = False) -> list[str]:
        '''
        returns a list of component names depending on the input

        Parameters
        ----------
        model : str
            name of the model function: parallel, sequential or target model nC_mk_p.
        Ainf : bool
            if true, an infinite non-decaying component is added.
        num : int
            number of components.
        gs : bool, optional
            Whether to include an explicit ground-state component in the fit.
        local : bool, optional
            true, when called by local fit, uses different naming convention

        Returns
        -------
        list
            list of component names.

        '''
        labels = []
        if ca_order:
            labels += [f'CA_{i}' for i in range(ca_order)]
        if local:
            labels += [f'C_{i + 1}' for i in range(num)]
            if Ainf:
                labels.append('C_{inf}')
            return labels
        
        if model == 'sequential':
            labels += [f'EAS_{i + 1}' for i in range(num)]
            if Ainf:
                labels.append('EAS_{inf}')
        elif model == 'parallel':
            labels += [f'DAS_{i + 1}' for i in range(num)]
            if Ainf:
                labels.append('DAS_{inf}')
        elif model == '2C_3k_1':
            labels += ['A', 'B']
            if Ainf:
                labels.append('C_{inf}')
        elif model == '3C_5k_1':
            labels += ['A', 'B', 'C']
            if Ainf:
                labels.append('D_{inf}')
        elif model == '3C_4k_1':
            labels += ['A', 'B', 'C']
            if Ainf:
                labels.append('D_{inf}')     
        elif model == '4C_6k_1':
            labels += ['A', 'B', 'C', 'D']
            if Ainf:
                labels.append('E_{inf}')   
        else:
            labels += [f'C_{i + 1}' for i in range(num)]
            if Ainf:
                labels.append('C_{inf}')
        if gs:
            labels.append('GS')

        return labels

    def run_emcee(self, ds: str, results: dict, burn: int, init: int,
                  thin: int, target_ratio: int) -> None:
        '''
        initializes the emcee thread, sets noise parameter and connects the signals

        uses lmfits implementation of the emcee model,
        see https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee
        for more information

        Parameters
        ----------
        ds : str
            current dataset.
        results : dict
            contains fitted parameters and metadata, will be reused by the emcee algorithm.
        burn : int
            number of samples to discard at the beginning of the sampling regime.
        init : int
            number of initial samples drawn from the distribution (see steps in Minimizer.emcee).
        thin : int
            only accept one in every thin sample.
        target_ratio : int
            the chain should be longer than x times the integrated autocorr time.

        Returns
        -------
        None.

        '''

        _, delay, delA = self.get_data(ds=ds)
        emcee_results = copy.deepcopy(results)
        emcee_results['opt_params'].add('__lnsigma', value=np.log(
            0.1), min=np.log(0.001), max=np.log(1*np.nanmax(abs(delA))))

        self.worker = EmceeWorker(results=emcee_results, delay=delay, delA=delA, burn=burn,
                                  init=init, thin=thin, target_ratio=target_ratio)
        self.worker.finished.connect(self._finish_emcee)
        self.worker.progress.connect(self._redirect_worker_info)
        self.worker.results_signal.connect(self._redirect_worker_results)
        self.call_statusbar("info", msg.Status.s34)
        self.worker.start()

    def save_emcee_result_2model(self, results: dict) -> None:
        ''' saves emcee results dict to model '''
        if not results:
            self.call_statusbar('error', msg.Error.e21)
            return
        if 'ukey' in results:
            ukey = results['ukey']
        else:
            self.call_statusbar('error', msg.Error.e21)
            return

        ds_model = self._get_ds_model(results['ds'])
        results.pop('ds', None)
        results.pop('ukey', None)
        ds_model.update_global_fit_emcee(ukey, results)
        self.call_statusbar("info", msg.Status.s36)

    @staticmethod
    def _effective_dof(residuals: NDArray, num_params: int) -> tuple[NDArray, NDArray, NDArray]:
        '''
        Compute integrated autocorrelation estimates and effective degrees of freedom

        Parameters
        ----------
        residuals : NDArray
            1D array of shape (n_delay,) or 2D array of shape (n_delay, n_wave)
            containing your residuals.
        num_params : int
            number of fitted parameters.

        Returns
        -------
        rho_delay : float
            Mean lag-1 autocorrelation along the delay dimension.
        rho_wave : float or None
            Mean lag-1 autocorrelation along the wavelength dimension
            (None if residuals is 1D).
        dof : float
            Effective degrees of freedom = n_eff / autocorr length.

        '''
        # -------- 1D Matrix (Local Fit) -----------------------------------------------------------
        if residuals.ndim == 1:
            tau = emcee.autocorr.integrated_time(residuals[:, None],
                                                 c=5, quiet=True)[0]
            rho_delay = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            n_eff = residuals.size / tau

            return float(rho_delay), None, float(n_eff - num_params)

        # -------- 2D-Matrix -----------------------------------------------------------------------
        n_delay, n_wave = residuals.shape

        # delay axis
        taus_d, rhos_d = [], []
        for j in range(n_wave):
            tr = residuals[:, j]
            try:
                tau_j = emcee.autocorr.integrated_time(
                    tr[:, None],
                    c=5, quiet=True)[0]
            except AutocorrError:
                tau_j = len(tr)  # fall‑back: no correlation info
            taus_d.append(tau_j)
            rhos_d.append(np.corrcoef(tr[:-1], tr[1:])[0, 1])
        tau_delay = np.nanmean(taus_d)
        rho_delay = np.nanmean(rhos_d)
        n_eff_delay = n_delay / tau_delay

        # wavelength axis:
        taus_w, rhos_w = [], []
        for i in range(n_delay):
            tr = residuals[i, :]
            try:
                tau_i = emcee.autocorr.integrated_time(
                    tr[:, None],
                    c=5, quiet=True)[0]
            except AutocorrError:
                tau_i = len(tr)
            taus_w.append(tau_i)
            rhos_w.append(np.corrcoef(tr[:-1], tr[1:])[0, 1])
        tau_wave = np.nanmean(taus_w)
        rho_wave = np.nanmean(rhos_w)
        n_eff_wave = n_wave / tau_wave
        n_eff = n_eff_delay * n_eff_wave

        return float(rho_delay), float(rho_wave), float(n_eff - num_params)

    def calculate_fitting_output(self, fit_results: dict) -> dict:
        ''' calculates and estimates fit statistics and variability, caches and returns results dict '''

        # -------- calculate spectra from fitted parameters ----------------------------------------
        t_start = time.time()
        if fit_results['meta']['model'] == 'sequential':
            fit_results['delA_calc'], fit_results['conc'], fit_results['EAS'],  = self.model_theta_wrapper(
                params=fit_results['opt_params'], delay=self.current_delay, delA=self.current_delA,
                Ainf=fit_results['meta']['Ainf'],  model='sequential', weights=fit_results['weight_vector'],
                use_threshold_t0=fit_results['meta']['use_threshold_t0'],
                substeps=fit_results['meta']['substeps'], gs=fit_results['meta']['gs'],
                gs_spec=fit_results['meta']['gs_spec'], ca_order=fit_results['meta']['ca_order'], output=True)

            _, _, fit_results['DAS'] = self.model_theta_wrapper(
                params=fit_results['opt_params'], delay=self.current_delay, delA=self.current_delA,
                Ainf=fit_results['meta']['Ainf'],  model='parallel', weights=fit_results['weight_vector'],
                use_threshold_t0=fit_results['meta']['use_threshold_t0'],
                substeps=fit_results['meta']['substeps'], gs=fit_results['meta']['gs'],
                gs_spec=fit_results['meta']['gs_spec'], ca_order=fit_results['meta']['ca_order'], output=True)

        elif fit_results['meta']['model'] == 'parallel':
            fit_results['delA_calc'], fit_results['conc'], fit_results['DAS'],  = self.model_theta_wrapper(
                params=fit_results['opt_params'], delay=self.current_delay, delA=self.current_delA,
                Ainf=fit_results['meta']['Ainf'],  model='parallel', weights=fit_results['weight_vector'],
                use_threshold_t0=fit_results['meta']['use_threshold_t0'],
                substeps=fit_results['meta']['substeps'], gs=fit_results['meta']['gs'],
                gs_spec=fit_results['meta']['gs_spec'], ca_order=fit_results['meta']['ca_order'], output=True)

            _, _, fit_results['EAS'] = self.model_theta_wrapper(
                params=fit_results['opt_params'], delay=self.current_delay, delA=self.current_delA,
                Ainf=fit_results['meta']['Ainf'],  model='sequential', weights=fit_results['weight_vector'],
                use_threshold_t0=fit_results['meta']['use_threshold_t0'],
                substeps=fit_results['meta']['substeps'], gs=fit_results['meta']['gs'],
                gs_spec=fit_results['meta']['gs_spec'], ca_order=fit_results['meta']['ca_order'], output=True)

        else:
            fit_results['delA_calc'], fit_results['conc'], fit_results['SAS'],  = self.model_theta_wrapper(
                params=fit_results['opt_params'], delay=self.current_delay, delA=self.current_delA,
                Ainf=fit_results['meta']['Ainf'],  model=fit_results['meta']['model'],
                weights=fit_results['weight_vector'], use_threshold_t0=fit_results['meta']['use_threshold_t0'],
                substeps=fit_results['meta']['substeps'], gs=fit_results['meta']['gs'],
                gs_spec=fit_results['meta']['gs_spec'], ca_order=fit_results['meta']['ca_order'], output=True)

        t_end = time.time()
        fit_results['meta']['fit_time'] += (t_end-t_start)

        # -------- calculate basic fit statistics --------------------------------------------------
        t_start = time.time()
        fit_results['meta']['weights'] = False
        fit_results['meta']['grid_points'] = fit_results['residuals'].size
        fit_results['residuals'] = np.reshape(fit_results['residuals'],
                                              (len(self.current_delay), len(self.current_wavelength)))

        weights2D = np.tile(fit_results['weight_vector'], (self.current_delA.shape[0], 1))
        fit_results['meta']['SSR'] = jnp.sum(fit_results['residuals']**2)
        mean_delA = (self.current_delA*weights2D).sum() / weights2D.sum()
        SST_weighted = ((self.current_delA - mean_delA)**2 * weights2D).sum()
        fit_results['meta']['r2'] = np.round(1 - fit_results['meta']['SSR'] / SST_weighted, 8)
        fit_results['meta']['rmse'] = np.sqrt(np.mean(fit_results['residuals']**2))
        fit_results['meta']['mea'] = np.mean(np.abs(fit_results['residuals']))

        # -------- estimate autocorrelation (lag-1 + IAC) of residuals & effective sample size -----
        fit_results['meta']['rho_delay_est'], fit_results['meta']['rho_wave_est'], dof_eff_weighted = self._effective_dof(
            fit_results['residuals'], len(fit_results['theta']))
        fit_results['meta']['n_eff'] = dof_eff_weighted + len(fit_results['theta'])

        # -------- estimate variance ---------------------------------------------------------------
        fit_results['meta']['var_resid_eff'] = fit_results['meta']['SSR'] / dof_eff_weighted
        var_est_weighted = fit_results['meta']['var_resid_eff']
        fit_results['meta']['sigma_eff'] = jnp.sqrt(var_est_weighted)

        # -------- calculate the scaled jacobian of the varied parameters --------------------------

        for nm in ('gs_shift', 'gs_sigma'):
            if nm in fit_results['opt_params']:
                fit_results['opt_params'][nm].vary = False

        param_items = list(fit_results['opt_params'].items())

        vary_mask = jnp.array([par.vary for name, par in param_items])

        use_bleach = True if fit_results['meta']['gs_spec'] is not False else False
        jacobian = self.jacobian_func(
            fit_results["theta"],
            self.current_delay,
            self.current_delA,
            Ainf=fit_results["meta"]["Ainf"],
            model=fit_results["meta"]["model"],
            weights=fit_results["weight_vector"],
            use_threshold_t0=fit_results["meta"]["use_threshold_t0"],
            substeps=fit_results["meta"]["substeps"],
            gs=fit_results["meta"]["gs"],
            use_bleach=use_bleach,
            gs_spec=fit_results["meta"]["gs_spec"],
            ca_order=fit_results["meta"]["ca_order"],
            output=False)

        jacobian_free = jacobian[:, vary_mask]
        scale_free = self._scale_from_guess(fit_results['opt_params'])

        J_scaled = jacobian_free * scale_free[None, :]
        JTJ_scaled = J_scaled.T @ J_scaled
        eigs = jnp.linalg.eigvalsh(JTJ_scaled)
        fit_results['meta']['jac_condition_num'] = float(jnp.sqrt(eigs[-1] / eigs[0]))

        # -------- calculate the covariance, uncertainty & correlation -----------------------------
        fit_results['meta']['error_success'] = False
        if not jnp.isnan(jacobian_free).any():
            covar_scaled = jnp.linalg.pinv(JTJ_scaled, rcond=1e-12) * var_est_weighted
            covar = covar_scaled * jnp.outer(scale_free, scale_free)
            fit_results['covariance_matrix'] = covar
            if not jnp.isnan(covar).any():
                stderr = jnp.sqrt(jnp.diag(covar))
                if not jnp.isnan(stderr).any():
                    for name, par in param_items:
                        par.stderr = None
                    fit_results['corr_matrix'] = np.asarray(
                        covar / jnp.outer(stderr, stderr))
                    free_names = [name for name, par in param_items if par.vary]
                    for name, err in zip(free_names, stderr):
                        fit_results['opt_params'][name].stderr = float(err)
                    fit_results['meta']['error_success'] = True

        # -------- calculate basic fit statistics (unweighted) -------------------------------------
        if not np.all(fit_results['weight_vector'] == 1):  # custom weightening applied
            fit_results['meta']['weights'] = True
            weight_dummy = jnp.array(
                np.full(self.current_wavelength.shape, 1, dtype=float))
            fit_results['unweighted_residuals'] = self.current_delA - \
                fit_results['delA_calc']
            fit_results['meta']['unweighted_SSR'] = jnp.sum(
                fit_results['unweighted_residuals']**2)
            SST = ((self.current_delA - self.current_delA.mean())**2).sum()
            fit_results['meta']['unweighted_r2'] = 1 - \
                fit_results['meta']['unweighted_SSR'] / SST
            fit_results['meta']['unweighted_rmse'] = np.sqrt(
                np.mean(fit_results['unweighted_residuals']**2))
            fit_results['meta']['unweighted_mea'] = np.mean(
                np.abs(fit_results['unweighted_residuals']))

        # -------- estimate autocorrelation (lag-1 + IAC) of residuals & effective sample size -----
            fit_results['meta']['unweighted_rho_delay_est'], fit_results['meta']['unweighted_rho_wave_est'], dof_eff = self._effective_dof(
                fit_results['unweighted_residuals'], len(fit_results['theta']))
            fit_results['meta']['unweighted_n_eff'] = dof_eff + len(fit_results['theta'])

        # -------- estimate variance (unweighted)---------------------------------------------------
            fit_results['meta']['unweighted_var_resid_eff'] = fit_results['meta']['unweighted_SSR'] / dof_eff
            var_est = fit_results['meta']['unweighted_var_resid_eff']
            fit_results['meta']['unweighted_sigma_eff'] = jnp.sqrt(var_est)

        # -------- calculate the scaled jacobian of the varied parameters (unweighted)--------------
            jacobian = self.jacobian_func(
                fit_results["theta"],
                self.current_delay,
                self.current_delA,
                Ainf=fit_results["meta"]["Ainf"],
                model=fit_results["meta"]["model"],
                weights=weight_dummy,
                use_threshold_t0=fit_results["meta"]["use_threshold_t0"],
                substeps=fit_results["meta"]["substeps"],
                gs=fit_results["meta"]["gs"],
                use_bleach=use_bleach,
                gs_spec=fit_results["meta"]["gs_spec"],
                ca_order=fit_results["meta"]["ca_order"],
                output=False)
            jacobian_free = jacobian[:, vary_mask]
            J_scaled = jacobian_free * scale_free[None, :]
            JTJ_scaled = J_scaled.T @ J_scaled
            eigs = jnp.linalg.eigvalsh(JTJ_scaled)
            fit_results['meta']['jac_condition_num_unweighted'] = float(
                jnp.sqrt(eigs[-1] / eigs[0]))

        # -------- calculate the covariance, uncertainty & correlation -----------------------------
            fit_results['meta']['unweighted_error_success'] = False
            if not jnp.isnan(jacobian_free).any():
                covar_scaled = jnp.linalg.pinv(JTJ_scaled, rcond=1e-12) * var_est
                covar = covar_scaled * jnp.outer(scale_free, scale_free)
                fit_results['unweighted_covariance_matrix'] = covar
                if not jnp.isnan(covar).any():
                    stderr = jnp.sqrt(jnp.diag(covar))
                    if not jnp.isnan(stderr).any():
                        for name, par in param_items:
                            par.stderr = None
                        fit_results['unweighted_corr_matrix'] = np.asarray(
                            covar / jnp.outer(stderr, stderr))
                        free_names = [name for name, par in param_items if par.vary]
                        fit_results['unweighted_opt_params'] = fit_results['opt_params']
                        for name, err in zip(free_names, stderr):
                            fit_results['unweighted_opt_params'][name].stderr = float(err)
                        fit_results['meta']['unweighted_error_success'] = True

        t_end = time.time()

        fit_results['meta']['diagnostics_time'] = t_end - t_start
        fit_results['output'] = self.get_fitting_print(fit_results)
        self.current_fit_results = fit_results

        return fit_results

    def get_emcee_print(self, result: object) -> str:
        ''' formats and returns a string with the emcee fitting result '''
        # formatters
        value_formatter = tk.EngFormatter(places=1, sep="\N{THIN SPACE}")
        error_formatter = tk.EngFormatter(places=0, sep="\N{THIN SPACE}")
        label_width = 8

        # 1) build lists of original vs display names
        orig_names = [n for n, p in result.params.items() if p.vary]
        disp_names = ['ln(σ)' if n == '__lnsigma' else n for n in orig_names]
        fitting_print = '--- Results from the MCMC Posterior Analysis ---\n\n'
        fitting_print += f'Number of samples: {result.chain.shape[0]}\n'

        fitting_print += '\n--- PARAMETERS: ---\n'
        hdr = f"{'Param':>8s} {'–2σ':>11s} {'–1σ':>11s} {'Median':>11s} {'+1σ':>11s} {'+2σ':>11s}"
        sep = "-" * len(hdr)
        fitting_print += hdr + "\n" + sep + "\n"

        # 3) fill in each parameter’s quantiles
        for orig, disp in zip(orig_names, disp_names):
            chain = result.flatchain[orig]
            q_lo2, q_lo1, q_med, q_hi1, q_hi2 = np.percentile(
                chain, [2.275, 15.865, 50, 84.135, 97.725]
            )
            err_lo2, err_lo1 = q_lo2 - q_med, q_lo1 - q_med
            err_hi1, err_hi2 = q_hi1 - q_med, q_hi2 - q_med

            if orig == "__lnsigma":
                # plain 2-decimal rounding
                lo2 = f"{err_lo2:.2f}"
                lo1 = f"{err_lo1:.2f}"
                med = f"{q_med:.2f}"
                hi1 = f"{err_hi1:.2f}"
                hi2 = f"{err_hi2:.2f}"
            else:
                lo2 = error_formatter(err_lo2) + "s"
                lo1 = error_formatter(err_lo1) + "s"
                med = value_formatter(q_med) + "s"
                hi1 = error_formatter(err_hi1) + "s"
                hi2 = error_formatter(err_hi2) + "s"

            fitting_print += (
                f"{disp:>8s} "
                f"{lo2:>11s} {lo1:>11s} {med:>11s} {hi1:>11s} {hi2:>11s}\n"
            )

        # 4) lower-triangle correlation matrix
        if orig_names:
            # compute correlations from samples
            samples = np.vstack([result.flatchain[n] for n in orig_names]).T
            cov = np.cov(samples, rowvar=False)
            sigma = np.sqrt(np.diag(cov))
            C = cov / np.outer(sigma, sigma)
            n = len(orig_names)

            fitting_print += "\n--- CORRELATIONS: ---\n"
            # column headers (all but last)
            fitting_print += " " * label_width
            for disp in disp_names[:-1]:
                fitting_print += f"{disp:>{label_width}}"
            fitting_print += "\n"

            # each row
            for i in range(1, n):
                fitting_print += f"{disp_names[i]:<{label_width}}"
                for j in range(n - 1):
                    if j < i:
                        fitting_print += f"{C[i, j]:>{label_width}.2f}"
                    else:
                        fitting_print += " " * label_width
                fitting_print += "\n"
        else:
            fitting_print += "--- No varying parameters: correlation matrix not available ---\n"

        return fitting_print

    def get_fitting_print(self, fit_results: dict) -> str:
        ''' formats and returns a string with the fitting result '''
        # -------- unpack results ------------------------------------------------------------------
        value_formatter = tk.EngFormatter(places=1, sep="\N{THIN SPACE}")
        error_formatter = tk.EngFormatter(places=0, sep="\N{THIN SPACE}")
        jac_condition_number = fit_results['meta']['jac_condition_num']

        # -------- print reliability ---------------------------------------------------------------
        fitting_print = ''
        if fit_results['meta']['error_success']:
            if jac_condition_number < 100:
                fitting_print += 'problem seems well conditioned\n'
            elif jac_condition_number < 5000:
                fitting_print += 'caution: problem seems poorly conditioned\n'
            else:
                fitting_print += 'caution: problem is unstable: likely redundant parameters or bad scaling!\n'

            # -------- print parameters ------------------------------------------------------------
            fitting_print += '\n--- PARAMETERS: ---\n'
            for k, v in fit_results['opt_params'].valuesdict().items():
                par = fit_results['opt_params'][k]
                if not par.vary or par.stderr is None:
                    # Fixed parameter
                    value_str = value_formatter(v)
                    fitting_print += f"{k}: {value_str} (fixed)\n"
                else:
                    # Fitted parameter
                    error = par.stderr
                    percent_error = abs(100 * error / v) if v != 0 else 0
                    value_str = value_formatter(v)
                    error_str = error_formatter(error)
                    fitting_print += (
                        f"{k}: {value_str}s ± {error_str}s "
                        f"({percent_error:.2f}%)\n"
                    )
            if fit_results['meta']['Ainf']:
                fitting_print += 'Ainf:  True\n'
            else:
                fitting_print += 'Ainf:  False\n'

            # -------- print correlations ----------------------------------------------------------
            fitting_print += "\n--- CORRELATIONS: ---\n"
            free_names = [name for name, par in fit_results['opt_params'].items() if par.vary]
            n = len(free_names)
            C = fit_results['corr_matrix']
            row_idx = list(range(1, n))
            col_idx = list(range(0, n-1))
            label_width = 8

            fitting_print += " " * label_width
            for j in col_idx:
                fitting_print += f"{free_names[j]:>{label_width}}"
            fitting_print += "\n"

            # each row
            for i in row_idx:
                row = f"{free_names[i]:<{label_width}}"
                for j in col_idx:
                    if j < i:
                        row += f"{C[i, j]:>{label_width}.2f}"
                    else:
                        row += " " * label_width
                fitting_print += row + "\n"

            # -------- print fit metrices ----------------------------------------------------------
            fitting_print += '\n--- FIT METRICS: ---\n'
            fitting_print += f"Model: {fit_results['meta']['model']}\n"
            fitting_print += f"r²: {fit_results['meta']['r2']:.3f}\n"
            fitting_print += f"SSR: {fit_results['meta']['SSR']:.2f}\n"
            fitting_print += f"RMSE: {fit_results['meta']['rmse']:.3f} mOD\n"
            fitting_print += f"MEA: {fit_results['meta']['mea']:.3f} mOD\n"
            fitting_print += f"data points: {fit_results['meta']['grid_points']} (total), {fit_results['meta']['n_eff']:.0f} (effective)\n"
            fitting_print += f"eff residual var: {fit_results['meta']['var_resid_eff']:.2f}\n"
            fitting_print += f"eff residual σ: {fit_results['meta']['sigma_eff']:.2f} mOD\n"
            fitting_print += f'Jacobian cond num: {jac_condition_number:.0f}\n'
            fitting_print += f"Avg lag-1 autocorr x axis: {fit_results['meta']['rho_wave_est']:.3f}\n"
            fitting_print += f"Avg lag-1 autocor  y axis: {fit_results['meta']['rho_delay_est']:.3f}\n"
            fitting_print += f"Method: {fit_results['meta']['method']}\n"
            fitting_print += f"time zero: {fit_results['meta']['time_zero_convention']}\n"
            fitting_print += f"# evaluations: {fit_results['meta']['#eval']}\n"
            fitting_print += f"fit + diagnostics time: {fit_results['meta']['fit_time']:.2f} s + {fit_results['meta']['diagnostics_time']:.2f} s\n"

        # -------- if error calculation fails, print reduced results -------------------------------
        else:
            fitting_print += "errors could not be calculated:\nnon-stable derivatives\n"
            fitting_print += '\n--- PARAMETERS: ---\n'
            for k, v in fit_results['opt_params'].valuesdict().items():
                par = fit_results['opt_params'][k]
                if not par.vary:
                    # Fixed parameter
                    value_str = value_formatter(v)
                    fitting_print += f"{k}: {value_str} (fixed)\n"
                else:
                    # Fitted parameter
                    value_str = value_formatter(v)
                    fitting_print += (f"{k}: {value_str}s\n")
            if fit_results['meta']['Ainf']:
                fitting_print += 'Ainf:  True\n'
            else:
                fitting_print += 'Ainf:  False\n'

            fitting_print += '\n--- FIT METRICS: ---\n'
            fitting_print += f"Model: {fit_results['meta']['model']}\n"
            fitting_print += f"r²: {fit_results['meta']['r2']:.3f}\n"
            fitting_print += f"SSR: {fit_results['meta']['SSR']:.2f}\n"
            fitting_print += f"RMSE: {fit_results['meta']['rmse']:.3f} mOD\n"
            fitting_print += f"MEA: {fit_results['meta']['mea']:.3f} mOD\n"
            fitting_print += f"data points: {fit_results['meta']['grid_points']} (total), {fit_results['meta']['n_eff']:.0f} (effective)\n"
            fitting_print += f"eff residual var: {fit_results['meta']['var_resid_eff']:.2f}\n"
            fitting_print += f"eff residual σ: {fit_results['meta']['sigma_eff']:.2f} mOD\n"
            fitting_print += f"Avg lag-1 autocorr x axis: {fit_results['meta']['rho_wave_est']:.3f}\n"
            fitting_print += f"Avg lag-1 autocor  y axis: {fit_results['meta']['rho_delay_est']:.3f}\n"
            fitting_print += f"Method: {fit_results['meta']['method']}\n"
            fitting_print += f"time zero: {fit_results['meta']['time_zero_convention']}\n"
            fitting_print += f"# evaluations: {fit_results['meta']['#eval']}\n"
            fitting_print += f"fit + diagnostics time: {fit_results['meta']['fit_time']:.2f} s + {fit_results['meta']['diagnostics_time']:.2f} s\n"

        # -------- if custom weightening applied, print unweighted results -------------------------
        if fit_results['meta']['weights']:
            fitting_print += "\n-------------------\n--- UNWEIGHTED RESULTS: ---\n"
            if fit_results['meta']['unweighted_error_success']:
                # -------- print parameters --------------------------------------------------------
                fitting_print += '\n--- PARAMETERS: ---\n'
                for k, v in fit_results['opt_params'].valuesdict().items():
                    par = fit_results['unweighted_opt_params'][k]
                    if not par.vary or par.stderr is None:
                        # Fixed parameter
                        value_str = value_formatter(v)
                        fitting_print += f"{k}: {value_str} (fixed)\n"
                    else:
                        # Fitted parameter
                        error = par.stderr
                        percent_error = abs(100 * error / v) if v != 0 else 0
                        value_str = value_formatter(v)
                        error_str = error_formatter(error)
                        fitting_print += (
                            f"{k}: {value_str}s ± {error_str}s "
                            f"({percent_error:.2f}%)\n"
                        )
                if fit_results['meta']['Ainf']:
                    fitting_print += 'Ainf:  True\n'
                else:
                    fitting_print += 'Ainf:  False\n'

                # -------- print correlations ------------------------------------------------------
                fitting_print += "\n--- CORRELATIONS: ---\n"
                free_names = [name for name, par in fit_results['opt_params'].items() if par.vary]
                n = len(free_names)
                C = fit_results['unweighted_corr_matrix']
                row_idx = list(range(1, n))
                col_idx = list(range(0, n-1))
                label_width = 8

                fitting_print += " " * label_width
                for j in col_idx:
                    fitting_print += f"{free_names[j]:>{label_width}}"
                fitting_print += "\n"

                # each row
                for i in row_idx:
                    row = f"{free_names[i]:<{label_width}}"
                    for j in col_idx:
                        if j < i:
                            row += f"{C[i, j]:>{label_width}.2f}"
                        else:
                            row += " " * label_width
                    fitting_print += row + "\n"
                # -------- print fit metrices ------------------------------------------------------
                fitting_print += '\n--- FIT METRICS: ---\n'
                fitting_print += f"r²: {fit_results['meta']['unweighted_r2']:.3f}\n"
                fitting_print += f"SSR: {fit_results['meta']['unweighted_SSR']:.2f}\n"
                fitting_print += f"RMSE: {fit_results['meta']['unweighted_rmse']:.3f} mOD\n"
                fitting_print += f"MEA: {fit_results['meta']['unweighted_mea']:.3f} mOD\n"
                fitting_print += f"data points: {fit_results['meta']['grid_points']} (total), {fit_results['meta']['unweighted_n_eff']:.0f} (effective)\n"
                fitting_print += f"eff residual var: {fit_results['meta']['unweighted_var_resid_eff']:.2f}\n"
                fitting_print += f"eff residual σ: {fit_results['meta']['unweighted_sigma_eff']:.2f} mOD\n"
                fitting_print += f"Jacobian cond num: {fit_results['meta']['jac_condition_num_unweighted']:.0f}\n"
                fitting_print += f"\nAvg lag-1 autocorr x axis: {fit_results['meta']['unweighted_rho_wave_est']:.3f}\n"
                fitting_print += f"Avg lag-1 autocor  y axis: {fit_results['meta']['unweighted_rho_delay_est']:.3f}\n"

            # -------- if error calculation fails, print reduced results ---------------------------
            else:
                fitting_print += "errors could not be calculated:\nnon-stable derivatives\n"
                fitting_print += '\n--- FIT METRICS: ---\n'
                fitting_print += f"r²: {fit_results['meta']['unweighted_r2']:.3f}\n"
                fitting_print += f"SSR: {fit_results['meta']['unweighted_SSR']:.2f}\n"
                fitting_print += f"RMSE: {fit_results['meta']['unweighted_rmse']:.3f} mOD\n"
                fitting_print += f"MEA: {fit_results['meta']['unweighted_mea']:.3f} mOD\n"
                fitting_print += f"data points: {fit_results['meta']['grid_points']} (total), {fit_results['meta']['unweighted_n_eff']:.0f} (effective)\n"
                fitting_print += f"eff residual var: {fit_results['meta']['unweighted_var_resid_eff']:.2f} mOD\n"
                fitting_print += f"eff residual σ: {fit_results['meta']['unweighted_sigma_eff']:.2f}\n"
                fitting_print += f"\nAvg lag-1 autocorr x axis: {fit_results['meta']['unweighted_rho_wave_est']:.3f}\n"
                fitting_print += f"Avg lag-1 autocor  y axis: {fit_results['meta']['unweighted_rho_delay_est']:.3f}\n"

        return fitting_print

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)
