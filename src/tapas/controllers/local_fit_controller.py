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
import logging
import logging.config
import time
import copy

# Third‑Party Imports
from PyQt6.QtCore import pyqtSignal
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from lmfit import Minimizer, Parameters
import matplotlib.ticker as tk

# Local Application Imports
from .global_fit_controller import GlobalFitController, EmceeWorker
from ..configurations import messages as msg, exceptions as exc
from ..utils import utils
logger = logging.getLogger(__name__)


class LocalFitController(GlobalFitController):
    status_signal = pyqtSignal(str, str)

    def __init__(self,  ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3):
        super().__init__(None, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3)
        self.initialize_fitting_cache()

    def run_emcee(self, ds: str, results: dict, burn: int, init: int, thin: int, target_ratio: int) -> None:
        '''
        initializes the emcee thread, sets noise parameter and connects the signals

        uses lmfits implementation of the emcee model, see https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee
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

        area = int(0.5 * results['meta']['band average (nm)'])
        _, delay, delA = self.get_data(wavelength_input=str(
            results['meta']['wavelength']), wavelength_area=area, ds=ds)

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
        ds_model.update_local_fit_emcee(ukey, results)
        self.call_statusbar("info", msg.Status.s36)

    def save_current_fit(self, ds: str) -> None:
        ''' saves cached fit results dict to model '''
        ds_model = self._get_ds_model(ds)
        if not self.verify_rawdata():
            self.call_statusbar("error", msg.Error.e05)
            return

        if self.current_fit_results is None:
            self.call_statusbar('error', msg.Error.e21)
            return

        i = 1

        ukey = utils.Converter.convert_nm_float2string(self.current_wavelength)
        # create unique key:
        if ukey in ds_model.local_fit:
            while ukey + '_' + str(i) in ds_model.local_fit:
                i += 1
            ukey = ukey + '_' + str(i)

        self.current_fit_results['wavelength'] = self.current_wavelength
        self.current_fit_results['delay'] = self.current_delay
        self.current_fit_results['delA'] = self.current_delA
        self.current_fit_results.pop('residuals', None)

        ds_model.update_local_fit(ukey, self.current_fit_results)
        self.call_statusbar("info", msg.Status.s24)

    def delete_fit(self, ds: str, selected_fit: int) -> None:
        ''' delets selected fit from the model '''
        ds_model = self._get_ds_model(ds)
        ukey = list(ds_model.local_fit)[selected_fit]

        ds_model.del_local_fit_key(ukey)

    def get_fit(self, ds: str, selected_fit: int) -> tuple[dict, str]:
        ''' gets and returns the fit dict and the name of the selected fit and dataset in the gui'''
        ds_model = self._get_ds_model(ds)
        ukey = list(ds_model.local_fit)[selected_fit]

        return ds_model.local_fit[ukey], ukey

    def get_local_fit_list(self, ds: str) -> dict:
        ''' returns local fit dict of current dataset ds '''
        ds_model = self._get_ds_model(ds)
        return ds_model.local_fit

    def optimize_params(
            self, params: Parameters, ds: str, input_wavelength: str, wavelength_area: int,
            Ainf: bool,  model: str, method: str, use_threshold_t0: bool, ca_order:int) -> dict:
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
        input_wavelength : str
            wavlength where fit will be evaluated.
        wavelength_area : int
            fit will be evaluated at input_wavelength ± wavelength_area.
        Ainf : bool
            if true, an infinite non-decaying component is added.
        model : str
            name of the model function: parallel, sequential or target model nC_mk_p.
        method : str
            name of the fitting method used by lmfit and scipy.
        use_threshold_t0 : bool
            if true t0 parameter represents 5% of the rising IRF, if False uses the Guassian centroid.
            internally t0 always represents the maximum of the IRF.

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
        try:
            wavelength, delay, delA = self.get_data(
                wavelength_input=input_wavelength, wavelength_area=wavelength_area, ds=ds)
        except ValueError:
            raise exc.FittingError

        minner = Minimizer(self.model_theta_wrapper, params,  fcn_kws={
                           'delay': delay, 'delA': delA,  'Ainf': Ainf,  'model': model,
                           'weights': jnp.array([1]), 'use_threshold_t0': use_threshold_t0,
                           'ca_order': ca_order,'output': False, })

        if method == 'diff-evol':
            method = 'differential_evolution'

        # -------- perform the minimization --------------------------------------------------------
        try:
            t_start = time.time()
            lm_fit_results = minner.minimize(method=method)
            t_end = time.time()
            fit_time = t_end-t_start

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
            fit_results['opt_params'], use_threshold_t0)
        fit_results['meta']['Ainf'] = Ainf
        fit_results['meta']['model'] = model
        fit_results['meta']['fit_time'] = fit_time
        fit_results['meta']['#eval'] = lm_fit_results.nfev
        fit_results['residuals'] = lm_fit_results.residual
        fit_results['meta']['method'] = lm_fit_results.method
        fit_results['meta']['band average (nm)'] = 2 * wavelength_area
        fit_results['meta']['wavelength'] = wavelength
        fit_results['meta']['ca_order'] = ca_order
        fit_results['meta']['time_zero_convention'] = '5% threshold' if use_threshold_t0 else 'Gaussian centroid'
        fit_results['meta']['use_threshold_t0'] = use_threshold_t0
        labels = self.get_component_labels(
            model=fit_results['meta']['model'], Ainf=fit_results['meta']['Ainf'], num=(len(fit_results['theta'])-2), ca_order=ca_order, local = True)
        fit_results['meta']['components'] = labels

        # -------- cache data for subsequent fit statistics  ---------------------------------------
        self.current_wavelength = wavelength
        self.current_delay = delay
        self.current_delA = delA

        return fit_results

    def calculate_fitting_output(self, fit_results: dict) -> dict:
        ''' calculates and estimates fit statistics and variability, caches and returns results dict '''
        # -------- calculate spectra from fitted parameters ----------------------------------------
        t_start = time.time()
        fit_results['delA_calc'], fit_results['conc'], fit_results['Amp'],  = self.model_theta_wrapper(
            params=fit_results['opt_params'], delay=self.current_delay, delA=self.current_delA,
            Ainf=fit_results['meta']['Ainf'],  model=fit_results['meta']['model'],
            weights=jnp.array([1]), use_threshold_t0=fit_results['meta']['use_threshold_t0'],
            substeps=10, ca_order = fit_results['meta']['ca_order'], output=True)
        t_end = time.time()
        fit_results['meta']['fit_time'] += (t_end-t_start)

        # -------- calculate basic fit statistics --------------------------------------------------
        t_start = time.time()
        fit_results['meta']['grid_points'] = fit_results['residuals'].size
        fit_results['meta']['SSR'] = jnp.sum(fit_results['residuals']**2)
        mean_delA = np.mean(self.current_delA)
        SST = np.sum((self.current_delA - mean_delA)**2)
        fit_results['meta']['r2'] = np.round(1 - fit_results['meta']['SSR'] / SST, 4)
        fit_results['meta']['rmse'] = np.sqrt(np.mean(fit_results['residuals']**2))
        fit_results['meta']['mea'] = np.mean(np.abs(fit_results['residuals']))

        # -------- estimate autocorrelation (lag-1) of residuals & effective sample size -----------
        fit_results['meta']['rho_delay_est'], _, dof_eff = self._effective_dof(
            fit_results['residuals'], len(fit_results['theta']))

        # -------- estimate variance ---------------------------------------------------------------
        fit_results['meta']['var_resid_eff'] = fit_results['meta']['SSR'] / dof_eff
        var_est = fit_results['meta']['var_resid_eff']
        fit_results['meta']['sigma_eff'] = jnp.sqrt(var_est)
        fit_results['meta']['n_eff'] = dof_eff + len(fit_results['theta'])

        # -------- calculate the scaled jacobian of the varied parameters --------------------------
        param_items = list(fit_results['opt_params'].items())
        vary_mask = jnp.array([par.vary for name, par in param_items])
        jacobian = self.jacobian_func(
            fit_results['theta'],
            self.current_delay,
            self.current_delA,
            Ainf=fit_results['meta']['Ainf'],
            model=fit_results['meta']['model'],
            weights=jnp.array([1]),
            use_threshold_t0=fit_results['meta']['use_threshold_t0'],
            substeps=6,
            gs=False,
            use_bleach=False,
            gs_spec=False,
            ca_order=fit_results['meta']['ca_order'],
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
            covar_scaled = jnp.linalg.pinv(JTJ_scaled, rcond=1e-12) * var_est
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
        t_end = time.time()

        fit_results['meta']['diagnostics_time'] = t_end - t_start

        fit_results['output'] = self.get_fitting_print(fit_results)
        self.current_fit_results = fit_results

        return fit_results

    def get_data(self, wavelength_input: str, wavelength_area: int,  ds: str) -> tuple[float, NDArray, NDArray]:
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

        try:
            wavelength = utils.Converter.convert_str_input2float(
                wavelength_input)

        except ValueError:
            self.call_statusbar("error", msg.Error.e02)
            raise ValueError
        if wavelength is None:
            self.call_statusbar("error", msg.Error.e18)
            raise ValueError

        idx_wavelength = (abs(buffer_dataX - wavelength)).argmin()
        idx_wavelength_min = (
            abs(buffer_dataX - (wavelength-wavelength_area*1e-9))).argmin()
        idx_wavelength_max = (
            abs(buffer_dataX - (wavelength+wavelength_area*1e-9))).argmin()
        wavelength = round(buffer_dataX[idx_wavelength], 9)

        delay = buffer_dataY
        delA = np.mean(
            buffer_dataZ[:, idx_wavelength_min:idx_wavelength_max+1], axis=1)

        return wavelength, delay, delA

    def get_fitting_print(self, fit_results:dict) -> str:
        ''' formats and returns a string with the fitting result '''
        # -------- unpack results ------------------------------------------------------------------
        value_formatter = tk.EngFormatter(places=1, sep="\N{THIN SPACE}")
        error_formatter = tk.EngFormatter(places=0, sep="\N{THIN SPACE}")
        decays = [f'τ{i}' for i in range(1, len(fit_results['opt_params'])-1)]
        decays.append('Ainf')
        amp_list = fit_results['Amp']
        amp_list = amp_list[fit_results['meta']['ca_order']:]  # exclude irf amps from relative amp calc
        amp_norm = (sum(abs(amp_list)))
        amp_dict = dict(zip(decays, amp_list))
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
                    amp = (f"{round(100 * amp_dict[k] / amp_norm)} %"
                           if k not in ('t0', 'IRF') else '')

                    value_str = value_formatter(v)
                    error_str = error_formatter(error)
                    fitting_print += (
                        f"{k}: {amp} {value_str}s ± {error_str}s "
                        f"({percent_error:.2f}%)\n"
                    )
            if fit_results['meta']['Ainf']:
                fitting_print += 'Ainf: ' + \
                    str(round(100 * amp_dict['Ainf'] / amp_norm)) + ' %\n'
            else:
                fitting_print += 'Ainf: False\n'

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
                    amp = (f"{round(100 * amp_dict[k] / amp_norm)} %"
                           if k not in ('t0', 'IRF') else '')

                    value_str = value_formatter(v)
                    fitting_print += (f"{k}: {amp} {value_str}s\n")
            if fit_results['meta']['Ainf']:
                fitting_print += 'Ainf: ' + str(round(100 * amp_dict['Ainf'] / amp_norm)) + ' %\n'
            else:
                fitting_print += 'Ainf: False\n'

            fitting_print += '\n--- FIT METRICS: ---\n'
            fitting_print += f"Model: {fit_results['meta']['model']}\n"
            fitting_print += f"r²: {fit_results['meta']['r2']:.3f}\n"
            fitting_print += f"SSR: {fit_results['meta']['SSR']:.2f}\n"
            fitting_print += f"RMSE: {fit_results['meta']['rmse']:.3f} mOD\n"
            fitting_print += f"MEA: {fit_results['meta']['mea']:.3f} mOD\n"
            fitting_print += f"data points: {fit_results['meta']['grid_points']} (total), {fit_results['meta']['n_eff']:.0f} (effective)\n"
            fitting_print += f"eff residual var: {fit_results['meta']['var_resid_eff']:.2f}\n"
            fitting_print += f"eff residual σ: {fit_results['meta']['sigma_eff']:.2f} mOD\n"
            fitting_print += f"Avg lag-1 autocor  y axis: {fit_results['meta']['rho_delay_est']:.3f}\n"
            fitting_print += f"Method: {fit_results['meta']['method']}\n"
            fitting_print += f"time zero: {fit_results['meta']['time_zero_convention']}\n"
            fitting_print += f"# evaluations: {fit_results['meta']['#eval']}\n"
            fitting_print += f"fit + diagnostics time: {fit_results['meta']['fit_time']:.2f} s + {fit_results['meta']['diagnostics_time']:.2f} s\n"

        return fitting_print

    def call_statusbar(self, level: str, message: str) -> None:
        self.status_signal.emit(level, message)
