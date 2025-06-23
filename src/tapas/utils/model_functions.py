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

from functools import partial
import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from jax import lax, vmap, jit
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)


class ModelFunctions:

    @staticmethod
    def _sigma(fwhm: float) -> float:
        ''' converts a gaussian fwhm to its standard deviation '''
        return fwhm / 2.35482

    @staticmethod
    def _gaussian_irf(t: jnp.ndarray, t0: float, fwhm: float) -> jnp.ndarray:
        σ = ModelFunctions._sigma(fwhm)
        return jnp.exp(-0.5 * ((t - t0) / σ) ** 2)          # peak ≈ 1

    @staticmethod
    def _irf_derivative(t: jnp.ndarray, t0: float, fwhm: float) -> jnp.ndarray:
        g = ModelFunctions._gaussian_irf(t, t0, fwhm)
        σ = ModelFunctions._sigma(fwhm)
        return -(t - t0) / σ**2 * g                         # peak ≈ 1/e

    @staticmethod
    def _irf_second_derivative(t: jnp.ndarray, t0: float, fwhm: float) -> jnp.ndarray:
        g = ModelFunctions._gaussian_irf(t, t0, fwhm)
        σ = ModelFunctions._sigma(fwhm)
        return ((t - t0)**2 / σ**4 - 1/σ**2) * g

    @staticmethod
    def _gaussian_pdf(t: jnp.ndarray, t0: float, fwhm: float) -> jnp.ndarray:
        '''
        Gaussian probability density function (PDF).

        Computes the value of a normalized Gaussian (normal) distribution
        with center `t0` and width defined by its full-width at half-maximum (FWHM)
        at each time point in `t`.

        Parameters
        ----------
        t : array_like
            Time points (or general support) at which to evaluate the PDF.
        t0 : float
            Mean (center) of the Gaussian distribution.
        fwhm : float
            Full-width at half-maximum of the Gaussian. Internally converted
            to standard deviation σ via σ = fwhm / (2√(2 ln 2)).

        Returns
        -------
        pdf : jnp.ndarray
            Gaussian PDF values at each element of `t`, normalized so that
            the total area under the curve is 1:

        '''
        sigma = ModelFunctions._sigma(fwhm)
        return jnp.exp(-0.5*((t-t0)/sigma)**2) / (sigma*jnp.sqrt(2*jnp.pi))

    @staticmethod
    def _gaussian_cdf(t: jnp.ndarray, t0: float, fwhm: float) -> jnp.ndarray:
        '''
        Gaussian (normal) cumulative distribution function (CDF).

        Computes the cumulative probability up to each time point in `t`
        for a Gaussian distribution centered at `t0` with width defined
        by its full-width at half-maximum (FWHM).

        Parameters
        ----------
        t : array_like
            Time points at which to evaluate the CDF.
        t0 : float
            Mean (center) of the Gaussian distribution.
        fwhm : float
            Full-width at half-maximum of the Gaussian. Internally converted
            to standard deviation σ via σ = fwhm / (2√(2 ln 2)).

        Returns
        -------
        cdf : jnp.ndarray
            Gaussian CDF values at each element of `t`

        '''
        sigma = ModelFunctions._sigma(fwhm)
        return 0.5 * (1 + erf((t - t0)/(jnp.sqrt(2)*sigma)))

    @staticmethod
    def _emg(t: jnp.ndarray, tau: float, t0: float, fwhm: float) -> jnp.ndarray:
        '''
        Exponentially Modified Gaussian (EMG) response function.

        This is the analytical convolution of an exponential decay
        with rate 1/τ and a Gaussian instrument-response function (IRF)
        of width defined by its full-width at half-maximum (FWHM).

        Parameters
        ----------
        t : array_like
            Time points at which to evaluate the EMG.
        tau : float
            Exponential decay constant (time constant τ).  The decay rate is 1/τ.
        t0 : float
            Time-zero offset (center of the Gaussian IRF).
        fwhm : float
            Full-width at half-maximum of the Gaussian IRF.  Internally
            converted to standard deviation σ via σ = fwhm / (2√(2 ln 2)).

        Returns
        -------
        emg : jnp.ndarray
            EMG function values at each `t`.

        Notes
        -----
        - If τ → ∞, this reduces to a pure Gaussian centered at t0.
        - If fwhm → 0, this reduces to a pure exponential decay starting at t0.

        '''
        sigma = ModelFunctions._sigma(fwhm)
        r = 1.0 / tau
        dt = t - t0
        arg = (sigma**2 * r - dt)/(jnp.sqrt(2)*sigma)
        return 0.5 * jnp.exp(0.5*(sigma*r)**2 - r*dt) * erfc(arg)

    @staticmethod
    def _make_bleach(shift_nm: float, sigma_nm: float, ssa: jnp.ndarray) -> jnp.ndarray:
        '''
        Generate a negative “bleach” template by shifting and optionally broadening
        a steady-state absorption spectrum.

        Parameters
        ----------
        shift_nm : float
            Wavelength shift Δλ in meters. Positive values shift the spectrum
            to longer wavelengths, negative to shorter.
        sigma_nm : float
            Standard deviation σ of Gaussian broadening in meters. If zero
            (or very small), no broadening is applied.
        ssa : jnp.ndarray of shape (λ, 2)
            Steady-state absorbance data on the TA grid. Column 0 is wavelength (m)
            and column 1 is positive, unit-peak absorbance.

        Returns
        -------
        bleach : jnp.ndarray of shape (λ,)
            Negative bleach template on the same wavelength grid:
            1. The absorbance spectrum is linearly interpolated at (λ + shift_nm).
            2. If `sigma_nm > 0`, the shifted spectrum is Gaussian-filtered via FFT.
            3. The result is negated to represent a bleach (ΔA < 0).

        '''
        lam = ssa[:, 0]
        amp = ssa[:, 1]

        # --- shift --------------------------------------------------------------------------------
        shifted = jnp.interp(lam + shift_nm, lam, amp, left=0.0, right=0.0)

        # --- broadening ---------------------------------------------------------------------------
        def with_broadening(_):
            dλ = lam[1] - lam[0]
            k = jnp.fft.rfftfreq(lam.size, dλ)
            gauss_tf = jnp.exp(-0.5 * (2 * jnp.pi * k * sigma_nm) ** 2)
            return jnp.fft.irfft(jnp.fft.rfft(shifted) * gauss_tf, lam.size)

        broaden = lax.cond(sigma_nm > 1e-6,
                           with_broadening,
                           lambda _: shifted,
                           operand=None)

        return -broaden

    @staticmethod
    def interval_scan(
            c_init: jnp.ndarray, t_pair: tuple[float, float], substeps: int, t0: float, irf: float,
            kinetics_fn: callable) -> tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Propagate the system state over a single experimental interval using RK4 substeps.

        This function divides the interval [t_start, t_end] into `substeps` equal segments,
        evaluates the Gaussian-shaped excitation profile (IRF) at each midpoint, and applies
        a classic 4th-order Runge–Kutta update for the kinetics at each micro-step.

        Parameters
        ----------
        c_init : jnp.ndarray of shape (npool,)
            Initial concentrations (or populations) of each kinetic pool.
        t_pair : tuple of float
            (t_start, t_end) defining the bounds of the current experimental interval.
        substeps : int
            Number of micro–RK4 steps to perform within the interval; more steps
            increase integration accuracy at the cost of extra computation.
        t0 : float
            Center (mean) of the Gaussian instrument response function (IRF).
        irf : float
            Full-width at half-maximum (FWHM) of the Gaussian IRF.
        kinetics_fn : callable
            Function `f(c, g) → dc/dt` defining the kinetics, where `c` is the
            current state and `g` is the instantaneous Gaussian excitation.

        Returns
        -------
        c_final : jnp.ndarray of shape (npool,)
            The state vector after integrating across [t_start, t_end].

        '''
        t_start, t_end = t_pair
        dt = (t_end - t_start) / substeps
        tmid = t_start + (jnp.arange(substeps) + 0.5) * dt  # midpoints for Gaussian input
        gmid = ModelFunctions._gaussian_pdf(tmid, t0, irf)

        def micro_step(c, g):
            # One RK4 micro-step
            k1 = kinetics_fn(c, g)
            k2 = kinetics_fn(c + 0.5 * dt * k1, g)
            k3 = kinetics_fn(c + 0.5 * dt * k2, g)
            k4 = kinetics_fn(c + dt * k3, g)
            c_next = c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return c_next, None

        c_final, _ = lax.scan(micro_step, c_init, gmid)
        return c_final, c_final

    @staticmethod
    @partial(jit, static_argnames=('Ainf', 'gs', 'use_bleach', 'ca_order', 'output'))
    def model_parallel(theta: jnp.ndarray, delay: jnp.ndarray, delA: jnp.ndarray, Ainf: bool,
                       weights: jnp.ndarray, gs: bool, use_bleach: bool,  gs_spec: jnp.ndarray, ca_order: int,
                       output: bool) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Objective function to be minimized: Parallel convolution and optional ground-state/bleach fitting.

        constructs the model matrix by independently convolving each
        exponential decay component with the instrument response (EMG), optionally
        adds an infinite-time offset pool (Ainf), and fits amplitudes (and
        optionally a bleach component) to the observed ΔA data in one linear solve.

        Parameters
        ----------
        theta : jnp.ndarray of shape (P,)
            Model parameters packed as:
              - θ[0] = t₀ (time-zero offset)
              - θ[1] = IRF full-width at half-maximum (FWHM)
              - θ[2:2+K] = decay time constants τᵢ for each excited pool
              - if `use_bleach`, final two elements are (bleach_shift_nm, bleach_broadening_nm)
        delay : jnp.ndarray of shape (N,)
            Time points of the transient measurement.
        delA : jnp.ndarray of shape (N, λ)
            Measured absorbance change at each time and wavelength.
        Ainf : bool
            If True, include an infinite-time ground-state pool whose response is
            the Gaussian CDF of the IRF.
        weights : jnp.ndarray of shape (N, λ)
            Weighting matrix for residuals.
        gs : bool
            If True, include an explicit ground-state component in the fit.
        use_bleach : bool
            If True (and `gs`), model an additional bleach spectrum:
            shift and broaden `gs_spec` by the last two θ elements.
        gs_spec : jnp.ndarray of shape (λ, 2)
            Steady-state absorption spectrum grid (wavelengths and amplitudes)
            used to build the bleach template.
        output : bool
            If True, return `(delA_cal, c_matrix, eps.T)` instead of residuals.

        Returns
        -------
        resid_flat : jnp.ndarray
            Flattened weighted residuals (size N·λ), unless `output=True`.
        OR
        (delA_cal, c_matrix, amplitudes) : tuple
            - delA_cal : (N, λ) fitted model signal
            - c_matrix : (N, npool) convolution basis (EMG [+ Ainf + GS if used])
            - amplitudes : (λ, npool) fitted amplitudes and (optional) bleach weights

        Notes
        -----
        1. Convolutions: Each decay component is convolved in parallel via
           EMG = exponential ⨂ Gaussian(IRF).
        2. Infinite-time pool: If `Ainf`, the CDF of the Gaussian IRF is appended.
        3. Ground state & bleach: When `gs=True`, a ‘GS’ vector ensures total
           population conservation. If `use_bleach`, a bleach spectrum is built
           from `gs_spec`, shifted by Δλ and optionally Gaussian-broadened.
        4. Linear solve: All columns are concatenated into `c_matrix` and amplitudes
           are found by least-squares; residuals are then weighted by √weights.

        '''
        # -------- parameters ----------------------------------------------------------------------
        decay_comps = theta[2:-2] if use_bleach else theta[2:]
        t0 = theta[0]
        irf = theta[1]

        # -------- kinetics ------------------------------------------------------------------------
        c = vmap(lambda tau: ModelFunctions._emg(delay, tau, t0, fwhm=irf),
                 out_axes=1)(decay_comps)
        if Ainf:
            cinf = ModelFunctions._gaussian_cdf(delay, t0=t0, fwhm=irf)
            c = jnp.concatenate([c, cinf[:, None]], axis=1)

        σ = ModelFunctions._sigma(irf)

        if ca_order == 0:
            ca_cols = jnp.empty((delay.shape[0], 0), delay.dtype)   # (N,0)

        elif ca_order == 1:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)   # (N,)
            ca_cols = ca_0[:, None]                                 # (N,1)

        elif ca_order == 2:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)
            ca_1 = σ * ModelFunctions._irf_derivative(delay, t0, irf)
            ca_cols = jnp.stack([ca_0, ca_1], axis=1)

        c = jnp.concatenate([ca_cols, c], axis=1)
        # -------- model explicit ground state -----------------------------------------------------
        if gs:
            GS = jnp.sum(c[:, ca_order:], axis=1, keepdims=True)
            if use_bleach:
                lambda_shift, sigma = theta[-2], theta[-1]
                bleach_vec = ModelFunctions._make_bleach(lambda_shift, sigma, gs_spec)  # (λ,)

                gsb_full = GS * bleach_vec         # shape (N,λ)
                alpha = jnp.sum(gsb_full * delA) / jnp.sum(gsb_full * gsb_full)

                delA_minus = delA - alpha * gsb_full
                eps_exc, *_ = jnp.linalg.lstsq(c, delA_minus, rcond=1e-6)

                delA_cal = c @ eps_exc + alpha * gsb_full
                eps = jnp.concatenate([eps_exc, (alpha * bleach_vec)[None, :]], axis=0)
                c = jnp.concatenate([c, GS], axis=1)

            else:
                c = jnp.concatenate([c, GS], axis=1)
                eps, *_ = jnp.linalg.lstsq(c, delA, rcond=None)
                delA_cal = c @ eps

        # -------- fit amplitudes & build residuals ------------------------------------------------
        else:
            eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-3)
            delA_cal = c @ eps

        resid = (delA - delA_cal) * jnp.sqrt(weights)

        if output:
            return delA_cal, c, eps.T          # (N,λ), (N,npool), (λ,npool)
        return resid.ravel()

    @staticmethod
    @partial(jit, static_argnames=('Ainf', 'substeps', 'gs', 'use_bleach', 'ca_order', 'output'))
    def model_sequential(theta: jnp.ndarray, delay: jnp.ndarray, delA: jnp.ndarray, Ainf: bool,
                         weights: jnp.ndarray, substeps: int, gs: bool, use_bleach: bool,
                         gs_spec: jnp.ndarray, ca_order: int, output: bool) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Objective function to be minimized: sequential kinetic pools via micro–RK4 integration over each time interval

        treats the decay components as a chain of first‐order pools (A→B→…),
        integrates the coupled ODEs over each experimental interval using a fixed number of
        RK4 micro‐steps, and then performs a single linear least‐squares fit of the model
        to the observed ΔA data. Optionally includes an infinite‐time offset pool (Ainf),
        an explicit ground‐state component (gs), and a bleach spectrum (use_bleach).

        Parameters
        ----------
        theta : jnp.ndarray of shape (P,)
            Model parameters packed as:
              - θ[0] = t₀ (time-zero offset)
              - θ[1] = IRF full-width at half-maximum (FWHM)
              - θ[2:2+K] = decay time constants τᵢ for each excited pool
              - if `use_bleach`, final two elements are (bleach_shift_nm, bleach_broadening_nm)
        delay : jnp.ndarray of shape (N,)
            Time points of the transient measurement.
        delA : jnp.ndarray of shape (N, λ)
            Measured absorbance change at each time and wavelength.
        Ainf : bool
            If True, include an infinite-time ground-state pool whose response is
            the Gaussian CDF of the IRF.
        weights : jnp.ndarray of shape (N, λ)
            Weighting matrix for residuals.
        substeps : int
            Number of RK4 micro‐steps per interval [delay[i], delay[i+1]].
        gs : bool
            If True, include an explicit ground-state component in the fit.
        use_bleach : bool
            If True (and `gs`), model an additional bleach spectrum:
            shift and broaden `gs_spec` by the last two θ elements.
        gs_spec : jnp.ndarray of shape (λ, 2)
            Steady-state absorption spectrum grid (wavelengths and amplitudes)
            used to build the bleach template.
        output : bool
            If True, return `(delA_cal, c_matrix, eps.T)` instead of residuals.

        Returns
        -------
        resid_flat : jnp.ndarray
            Flattened weighted residuals (size N·λ), unless `output=True`.
        OR
        (delA_cal, c_matrix, amplitudes) : tuple
            - delA_cal : (N, λ) fitted model signal
            - c_matrix : (N, npool) convolution basis (EMG [+ Ainf + GS if used])
            - amplitudes : (λ, npool) fitted amplitudes and (optional) bleach weights

        Notes
        -----
        1. Pools are represented by decay rates kᵢ = 1/τᵢ in a linear chain.
        2. ODE: dc₀/dt = g(t) – k₀ c₀;  dcᵢ/dt = k_{i−1} c_{i−1} – kᵢ cᵢ.
        3. g(t) is the Gaussian IRF evaluated at the midpoint of each micro‐step.
        4. RK4 micro‐step size = (t_end – t_start)/substeps.
        5. After simulation, ground‐state (negative sum of excited pools) and bleach
           are appended as extra columns before the final fit.

        '''

        # -------- parameters ----------------------------------------------------------------------
        t0, irf = theta[0], theta[1]
        decay_comps = theta[2:-2] if use_bleach else theta[2:]
        ks = 1.0 / decay_comps
        ks = jnp.concatenate([ks, jnp.zeros(1)]) if Ainf else ks
        npool = ks.size
        c0 = jnp.zeros(npool)

        # -------- kinetics ------------------------------------------------------------------------
        def kinetics(c, g):
            d0 = g - ks[0] * c[0]             # pumping & decay of pool 0
            flux = ks[:-1]*c[:-1] - ks[1:]*c[1:]  # A→B, B→C, …
            return jnp.concatenate([d0[None], flux])

        # -------- scan over all experimental intervals --------------------------------------------
        intervals = jnp.stack([delay[:-1], delay[1:]], axis=1)      # (N-1,2)
        scan_fn = partial(ModelFunctions.interval_scan, substeps=substeps,
                          t0=t0, irf=irf, kinetics_fn=kinetics)

        _, c_hist = lax.scan(scan_fn, c0, intervals)          # (N-1,npool)
        c = jnp.vstack([c0, c_hist])

        σ = ModelFunctions._sigma(irf)
        if ca_order == 0:
            ca_cols = jnp.empty((delay.shape[0], 0), delay.dtype)   # (N,0)

        elif ca_order == 1:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)   # (N,)
            ca_cols = ca_0[:, None]                                 # (N,1)

        elif ca_order == 2:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)
            ca_1 = σ * ModelFunctions._irf_derivative(delay, t0, irf)
            ca_cols = jnp.stack([ca_0, ca_1], axis=1)
        c = jnp.concatenate([ca_cols, c], axis=1)

        # -------- model explicit ground state -----------------------------------------------------
        if gs:
            GS = jnp.sum(c[:, ca_order:], axis=1, keepdims=True)
            if use_bleach:
                lambda_shift, sigma = theta[-2], theta[-1]
                bleach_vec = ModelFunctions._make_bleach(lambda_shift, sigma, gs_spec)  # (λ,)

                gsb_full = GS * bleach_vec         # shape (N,λ)
                alpha = jnp.sum(gsb_full * delA) / jnp.sum(gsb_full * gsb_full)

                delA_minus = delA - alpha * gsb_full
                eps_exc, *_ = jnp.linalg.lstsq(c, delA_minus, rcond=1e-6)

                delA_cal = c @ eps_exc + alpha * gsb_full
                eps = jnp.concatenate([eps_exc, (alpha * bleach_vec)[None, :]], axis=0)
                c = jnp.concatenate([c, GS], axis=1)

            else:
                c = jnp.concatenate([c, GS], axis=1)
                eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
                delA_cal = c @ eps

        # -------- fit amplitudes & build residuals ------------------------------------------------
        else:
            eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
            delA_cal = c @ eps

        resid = (delA - delA_cal) * jnp.sqrt(weights)

        if output:
            return delA_cal, c, eps.T          # (N,λ), (N,npool), (λ,npool)
        return resid.ravel()

    @staticmethod
    @partial(jit, static_argnames=('Ainf',  'substeps', 'gs', 'use_bleach', 'ca_order', 'output'))
    def model_2C_3k_1(theta: jnp.ndarray, delay: jnp.ndarray, delA: jnp.ndarray, Ainf: bool,
                      weights: jnp.ndarray, substeps: int, gs: bool, use_bleach: bool,
                      gs_spec: jnp.ndarray, ca_order: int, output: bool) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Objective function to be minimized: target kinetic pools via micro–RK4 integration over each time interval

        A --k1--> B --k2--> Cinf/GS; A --k3--> GS

        Parameters
        ----------
        theta : jnp.ndarray of shape (P,)
            Model parameters packed as:
              - θ[0] = t0 (time-zero offset)
              - θ[1] = IRF full-width at half-maximum (FWHM)
              - θ[2] = τ1
              - θ[3] = τ2
              - θ[4] = τ3
              - if `use_bleach`, final two elements are (bleach_shift_nm, bleach_broadening_nm)
        delay : jnp.ndarray of shape (N,)
            Time points of the transient measurement.
        delA : jnp.ndarray of shape (N, λ)
            Measured absorbance change at each time and wavelength.
        Ainf : bool
            If True, include an infinite-time ground-state pool whose response is
            the Gaussian CDF of the IRF.
        weights : jnp.ndarray of shape (N, λ)
            Weighting matrix for residuals.
        substeps : int
            Number of RK4 micro‐steps per interval [delay[i], delay[i+1]].
        gs : bool
            If True, include an explicit ground-state component in the fit.
        use_bleach : bool
            If True (and `gs`), model an additional bleach spectrum:
            shift and broaden `gs_spec` by the last two θ elements.
        gs_spec : jnp.ndarray of shape (λ, 2)
            Steady-state absorption spectrum grid (wavelengths and amplitudes)
            used to build the bleach template.
        output : bool
            If True, return `(delA_cal, c_matrix, eps.T)` instead of residuals.

        Returns
        -------
        resid_flat : jnp.ndarray
            Flattened weighted residuals (size N·λ), unless `output=True`.
        OR
        (delA_cal, c_matrix, amplitudes) : tuple
            - delA_cal : (N, λ) fitted model signal
            - c_matrix : (N, npool) convolution basis (EMG [+ Ainf + GS if used])
            - amplitudes : (λ, npool) fitted amplitudes and (optional) bleach weights
        '''

        # ---------- parameters --------------------------------------------------------------------
        t0, irf = theta[0],  theta[1]
        k1, k2, k3 = 1.0 / theta[2:-2] if use_bleach else 1 / theta[2:]

        n_pool = 3 if Ainf else 2
        c0 = jnp.zeros(n_pool)

        # ---------- kinetics ----------------------------------------------------------------------
        def kinetics(c, g):
            A = c[0]
            B = c[1]
            dA = g - (k1 + k3) * A
            dB = k1 * A - k2 * B
            if Ainf:
                C = c[2]
                dC = k2 * B
                return jnp.array([dA, dB, dC])
            else:
                return jnp.array([dA, dB])

        # -------- scan over all experimental intervals --------------------------------------------
        intervals = jnp.stack([delay[:-1], delay[1:]], axis=1)      # (N-1,2)
        scan_fn = partial(ModelFunctions.interval_scan, substeps=substeps,
                          t0=t0, irf=irf, kinetics_fn=kinetics)

        _, c_hist = lax.scan(scan_fn, c0, intervals)          # (N-1,npool)
        c = jnp.vstack([c0, c_hist])                                # (N,npool)

        σ = ModelFunctions._sigma(irf)
        if ca_order == 0:
            ca_cols = jnp.empty((delay.shape[0], 0), delay.dtype)   # (N,0)

        elif ca_order == 1:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)   # (N,)
            ca_cols = ca_0[:, None]                                 # (N,1)

        elif ca_order == 2:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)
            ca_1 = σ * ModelFunctions._irf_derivative(delay, t0, irf)
            ca_cols = jnp.stack([ca_0, ca_1], axis=1)
        c = jnp.concatenate([ca_cols, c], axis=1)

        # -------- model explicit ground state -----------------------------------------------------
        if gs:
            GS = jnp.sum(c[:, ca_order:], axis=1, keepdims=True)
            if use_bleach:
                lambda_shift, sigma = theta[-2], theta[-1]
                bleach_vec = ModelFunctions._make_bleach(lambda_shift, sigma, gs_spec)  # (λ,)

                gsb_full = GS * bleach_vec         # shape (N,λ)
                alpha = jnp.sum(gsb_full * delA) / jnp.sum(gsb_full * gsb_full)

                delA_minus = delA - alpha * gsb_full
                eps_exc, *_ = jnp.linalg.lstsq(c, delA_minus, rcond=1e-6)

                delA_cal = c @ eps_exc + alpha * gsb_full
                eps = jnp.concatenate([eps_exc, (alpha * bleach_vec)[None, :]], axis=0)
                c = jnp.concatenate([c, GS], axis=1)

            else:
                c = jnp.concatenate([c, GS], axis=1)
                eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
                delA_cal = c @ eps

        # -------- fit amplitudes & build residuals ------------------------------------------------
        else:
            eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
            delA_cal = c @ eps

        resid = (delA - delA_cal) * jnp.sqrt(weights)

        if output:
            return delA_cal, c, eps.T          # (N,λ), (N,npool), (λ,npool)
        return resid.ravel()

    @staticmethod
    @partial(jit, static_argnames=('Ainf', 'substeps', 'gs', 'use_bleach', 'ca_order', 'output'))
    def model_3C_5k_1(theta: jnp.ndarray, delay: jnp.ndarray, delA: jnp.ndarray, Ainf: bool,
                      weights: jnp.ndarray, substeps: int, gs: bool, use_bleach: bool,
                      gs_spec: jnp.ndarray, ca_order: int, output: bool) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Objective function to be minimized: target kinetic pools via micro–RK4 integration over each time interval

        A --k1--> B --k2--> Dinf/GS; A --k3--> C --k4--> Dinf/GS; A --k5--> GS

        Parameters
        ----------
        theta : jnp.ndarray of shape (P,)
            Model parameters packed as:
              - θ[0] = t0 (time-zero offset)
              - θ[1] = IRF full-width at half-maximum (FWHM)
              - θ[2] = τ1
              - θ[3] = τ2
              - θ[4] = τ3
              - θ[5] = τ4
              - θ[6] = τ5
              - if `use_bleach`, final two elements are (bleach_shift_nm, bleach_broadening_nm)
        delay : jnp.ndarray of shape (N,)
            Time points of the transient measurement.
        delA : jnp.ndarray of shape (N, λ)
            Measured absorbance change at each time and wavelength.
        Ainf : bool
            If True, include an infinite-time ground-state pool whose response is
            the Gaussian CDF of the IRF.
        weights : jnp.ndarray of shape (N, λ)
            Weighting matrix for residuals.
        substeps : int
            Number of RK4 micro‐steps per interval [delay[i], delay[i+1]].
        gs : bool
            If True, include an explicit ground-state component in the fit.
        use_bleach : bool
            If True (and `gs`), model an additional bleach spectrum:
            shift and broaden `gs_spec` by the last two θ elements.
        gs_spec : jnp.ndarray of shape (λ, 2)
            Steady-state absorption spectrum grid (wavelengths and amplitudes)
            used to build the bleach template.
        output : bool
            If True, return `(delA_cal, c_matrix, eps.T)` instead of residuals.

        Returns
        -------
        resid_flat : jnp.ndarray
            Flattened weighted residuals (size N·λ), unless `output=True`.
        OR
        (delA_cal, c_matrix, amplitudes) : tuple
            - delA_cal : (N, λ) fitted model signal
            - c_matrix : (N, npool) convolution basis (EMG [+ Ainf + GS if used])
            - amplitudes : (λ, npool) fitted amplitudes and (optional) bleach weights
        '''

        # ---------- parameters --------------------------------------------------------------------
        t0, irf = theta[0],  theta[1]
        k1, k2, k3, k4, k5 = 1.0 / theta[2:-2] if use_bleach else 1 / theta[2:]

        n_pool = 4 if Ainf else 3
        c0 = jnp.zeros(n_pool)

        # ---------- kinetics ----------------------------------------------------------------------
        def kinetics(c, g):
            A = c[0]
            B = c[1]
            C = c[2]

            dA = g - (k1 + k3 + k5) * A
            dB = k1 * A - k2 * B
            dC = k3 * A - k4 * C

            if Ainf:
                D = c[3]
                dD = k2 * B + k4 * C
                return jnp.array([dA, dB, dC, dD])
            else:
                return jnp.array([dA, dB, dC])

        # -------- scan over all experimental intervals --------------------------------------------
        intervals = jnp.stack([delay[:-1], delay[1:]], axis=1)      # (N-1,2)
        scan_fn = partial(ModelFunctions.interval_scan, substeps=substeps,
                          t0=t0, irf=irf, kinetics_fn=kinetics)

        _, c_hist = lax.scan(scan_fn, c0, intervals)          # (N-1,npool)
        c = jnp.vstack([c0, c_hist])                                # (N,npool)

        σ = ModelFunctions._sigma(irf)
        if ca_order == 0:
            ca_cols = jnp.empty((delay.shape[0], 0), delay.dtype)   # (N,0)

        elif ca_order == 1:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)   # (N,)
            ca_cols = ca_0[:, None]                                 # (N,1)

        elif ca_order == 2:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)
            ca_1 = σ * ModelFunctions._irf_derivative(delay, t0, irf)
            ca_cols = jnp.stack([ca_0, ca_1], axis=1)
        c = jnp.concatenate([ca_cols, c], axis=1)

        # -------- model explicit ground state -----------------------------------------------------
        if gs:
            GS = jnp.sum(c[:, ca_order:], axis=1, keepdims=True)
            if use_bleach:
                lambda_shift, sigma = theta[-2], theta[-1]
                bleach_vec = ModelFunctions._make_bleach(lambda_shift, sigma, gs_spec)  # (λ,)

                gsb_full = GS * bleach_vec         # shape (N,λ)
                alpha = jnp.sum(gsb_full * delA) / jnp.sum(gsb_full * gsb_full)

                delA_minus = delA - alpha * gsb_full
                eps_exc, *_ = jnp.linalg.lstsq(c, delA_minus, rcond=1e-6)

                delA_cal = c @ eps_exc + alpha * gsb_full
                eps = jnp.concatenate([eps_exc, (alpha * bleach_vec)[None, :]], axis=0)
                c = jnp.concatenate([c, GS], axis=1)

            else:
                c = jnp.concatenate([c, GS], axis=1)
                eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
                delA_cal = c @ eps

        # -------- fit amplitudes & build residuals ------------------------------------------------
        else:
            eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
            delA_cal = c @ eps

        resid = (delA - delA_cal) * jnp.sqrt(weights)

        if output:
            return delA_cal, c, eps.T          # (N,λ), (N,npool), (λ,npool)
        return resid.ravel()

    @staticmethod
    @partial(jit, static_argnames=('Ainf', 'substeps', 'gs', 'use_bleach', 'ca_order', 'output'))
    def model_3C_4k_1(theta: jnp.ndarray, delay: jnp.ndarray, delA: jnp.ndarray, Ainf: bool,
                      weights: jnp.ndarray, substeps: int, gs: bool, use_bleach: bool,
                      gs_spec: jnp.ndarray, ca_order: int, output: bool) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Objective function to be minimized: target kinetic pools via micro–RK4 integration over each time interval

        A --k1--> B --k2--> C --k3--> Dinf/GS; A --k4--> GS

        Parameters
        ----------
        theta : jnp.ndarray of shape (P,)
            Model parameters packed as:
              - θ[0] = t0 (time-zero offset)
              - θ[1] = IRF full-width at half-maximum (FWHM)
              - θ[2] = τ1
              - θ[3] = τ2
              - θ[4] = τ3
              - θ[5] = τ4
              - if `use_bleach`, final two elements are (bleach_shift_nm, bleach_broadening_nm)
        delay : jnp.ndarray of shape (N,)
            Time points of the transient measurement.
        delA : jnp.ndarray of shape (N, λ)
            Measured absorbance change at each time and wavelength.
        Ainf : bool
            If True, include an infinite-time ground-state pool whose response is
            the Gaussian CDF of the IRF.
        weights : jnp.ndarray of shape (N, λ)
            Weighting matrix for residuals.
        substeps : int
            Number of RK4 micro‐steps per interval [delay[i], delay[i+1]].
        gs : bool
            If True, include an explicit ground-state component in the fit.
        use_bleach : bool
            If True (and `gs`), model an additional bleach spectrum:
            shift and broaden `gs_spec` by the last two θ elements.
        gs_spec : jnp.ndarray of shape (λ, 2)
            Steady-state absorption spectrum grid (wavelengths and amplitudes)
            used to build the bleach template.
        output : bool
            If True, return `(delA_cal, c_matrix, eps.T)` instead of residuals.

        Returns
        -------
        resid_flat : jnp.ndarray
            Flattened weighted residuals (size N·λ), unless `output=True`.
        OR
        (delA_cal, c_matrix, amplitudes) : tuple
            - delA_cal : (N, λ) fitted model signal
            - c_matrix : (N, npool) convolution basis (EMG [+ Ainf + GS if used])
            - amplitudes : (λ, npool) fitted amplitudes and (optional) bleach weights
        '''

        # ---------- parameters --------------------------------------------------------------------
        t0, irf = theta[0],  theta[1]
        k1, k2, k3, k4, = 1.0 / theta[2:-2] if use_bleach else 1 / theta[2:]

        n_pool = 4 if Ainf else 3
        c0 = jnp.zeros(n_pool)

        # ---------- kinetics ----------------------------------------------------------------------
        def kinetics(c, g):
            A = c[0]
            B = c[1]
            C = c[2]

            dA = g - (k1 + k4) * A
            dB = k1 * A - k2 * B
            dC = k2 * B - k3 * C

            if Ainf:
                D = c[3]
                dD = k3 * C
                return jnp.array([dA, dB, dC, dD])
            else:
                return jnp.array([dA, dB, dC])

        # -------- scan over all experimental intervals --------------------------------------------
        intervals = jnp.stack([delay[:-1], delay[1:]], axis=1)      # (N-1,2)
        scan_fn = partial(ModelFunctions.interval_scan, substeps=substeps,
                          t0=t0, irf=irf, kinetics_fn=kinetics)

        _, c_hist = lax.scan(scan_fn, c0, intervals)          # (N-1,npool)
        c = jnp.vstack([c0, c_hist])                                # (N,npool)

        σ = ModelFunctions._sigma(irf)
        if ca_order == 0:
            ca_cols = jnp.empty((delay.shape[0], 0), delay.dtype)   # (N,0)

        elif ca_order == 1:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)   # (N,)
            ca_cols = ca_0[:, None]                                 # (N,1)

        elif ca_order == 2:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)
            ca_1 = σ * ModelFunctions._irf_derivative(delay, t0, irf)
            ca_cols = jnp.stack([ca_0, ca_1], axis=1)
        c = jnp.concatenate([ca_cols, c], axis=1)

        # -------- model explicit ground state -----------------------------------------------------
        if gs:
            GS = jnp.sum(c[:, ca_order:], axis=1, keepdims=True)
            if use_bleach:
                lambda_shift, sigma = theta[-2], theta[-1]
                bleach_vec = ModelFunctions._make_bleach(lambda_shift, sigma, gs_spec)  # (λ,)

                gsb_full = GS * bleach_vec         # shape (N,λ)
                alpha = jnp.sum(gsb_full * delA) / jnp.sum(gsb_full * gsb_full)

                delA_minus = delA - alpha * gsb_full
                eps_exc, *_ = jnp.linalg.lstsq(c, delA_minus, rcond=1e-6)

                delA_cal = c @ eps_exc + alpha * gsb_full
                eps = jnp.concatenate([eps_exc, (alpha * bleach_vec)[None, :]], axis=0)
                c = jnp.concatenate([c, GS], axis=1)

            else:
                c = jnp.concatenate([c, GS], axis=1)
                eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
                delA_cal = c @ eps

        # -------- fit amplitudes & build residuals ------------------------------------------------
        else:
            eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
            delA_cal = c @ eps

        resid = (delA - delA_cal) * jnp.sqrt(weights)

        if output:
            return delA_cal, c, eps.T          # (N,λ), (N,npool), (λ,npool)
        return resid.ravel()

    @staticmethod
    @partial(jit, static_argnames=('Ainf', 'substeps', 'gs', 'use_bleach', 'ca_order', 'output'))
    def model_4C_6k_1(theta: jnp.ndarray, delay: jnp.ndarray, delA: jnp.ndarray, Ainf: bool,
                      weights: jnp.ndarray, substeps: int, gs: bool, use_bleach: bool,
                      gs_spec: jnp.ndarray, ca_order: int, output: bool) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        '''
        Objective function to be minimized: target kinetic pools via micro–RK4 integration over each time interval

        A --k1--> B --k2--> C --k3--> D --k4--> GS; B --k6--> GS; D --k5--> Einf/GS

        Parameters
        ----------
        theta : jnp.ndarray of shape (P,)
            Model parameters packed as:
              - θ[0] = t0 (time-zero offset)
              - θ[1] = IRF full-width at half-maximum (FWHM)
              - θ[2] = τ1
              - θ[3] = τ2
              - θ[4] = τ3
              - θ[5] = τ4
              - θ[6] = τ5
              - θ[7] = τ6
              - if `use_bleach`, final two elements are (bleach_shift_nm, bleach_broadening_nm)
        delay : jnp.ndarray of shape (N,)
            Time points of the transient measurement.
        delA : jnp.ndarray of shape (N, λ)
            Measured absorbance change at each time and wavelength.
        Ainf : bool
            If True, include an infinite-time ground-state pool whose response is
            the Gaussian CDF of the IRF.
        weights : jnp.ndarray of shape (N, λ)
            Weighting matrix for residuals.
        substeps : int
            Number of RK4 micro‐steps per interval [delay[i], delay[i+1]].
        gs : bool
            If True, include an explicit ground-state component in the fit.
        use_bleach : bool
            If True (and `gs`), model an additional bleach spectrum:
            shift and broaden `gs_spec` by the last two θ elements.
        gs_spec : jnp.ndarray of shape (λ, 2)
            Steady-state absorption spectrum grid (wavelengths and amplitudes)
            used to build the bleach template.
        output : bool
            If True, return `(delA_cal, c_matrix, eps.T)` instead of residuals.

        Returns
        -------
        resid_flat : jnp.ndarray
            Flattened weighted residuals (size N·λ), unless `output=True`.
        OR
        (delA_cal, c_matrix, amplitudes) : tuple
            - delA_cal : (N, λ) fitted model signal
            - c_matrix : (N, npool) convolution basis (EMG [+ Ainf + GS if used])
            - amplitudes : (λ, npool) fitted amplitudes and (optional) bleach weights
        '''

        # ---------- parameters --------------------------------------------------------------------
        t0, irf = theta[0],  theta[1]
        k1, k2, k3, k4, k5, k6 = 1.0 / theta[2:-2] if use_bleach else 1 / theta[2:]

        n_pool = 5 if Ainf else 4
        c0 = jnp.zeros(n_pool)

        # ---------- kinetics ----------------------------------------------------------------------
        def kinetics(c, g):
            A = c[0]
            B = c[1]
            C = c[2]
            D = c[3]

            dA = g - k1 * A
            dB = k1 * A - (k2 + k6) * B
            dC = k2 * B - k3 * C
            dD = k3 * C - (k4 + k5) * D

            if Ainf:
                E = c[4]
                dE = k5 * D
                return jnp.array([dA, dB, dC, dD, dE])
            else:
                return jnp.array([dA, dB, dC, dD])

        # -------- scan over all experimental intervals --------------------------------------------
        intervals = jnp.stack([delay[:-1], delay[1:]], axis=1)      # (N-1,2)
        scan_fn = partial(ModelFunctions.interval_scan, substeps=substeps,
                          t0=t0, irf=irf, kinetics_fn=kinetics)

        _, c_hist = lax.scan(scan_fn, c0, intervals)          # (N-1,npool)
        c = jnp.vstack([c0, c_hist])                                # (N,npool)
        σ = ModelFunctions._sigma(irf)

        if ca_order == 0:
            ca_cols = jnp.empty((delay.shape[0], 0), delay.dtype)   # (N,0)

        elif ca_order == 1:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)   # (N,)
            ca_cols = ca_0[:, None]                                 # (N,1)

        elif ca_order == 2:
            ca_0 = ModelFunctions._gaussian_irf(delay, t0, irf)
            ca_1 = σ * ModelFunctions._irf_derivative(delay, t0, irf)
            ca_cols = jnp.stack([ca_0, ca_1], axis=1)

        c = jnp.concatenate([ca_cols, c], axis=1)
        # -------- model explicit ground state -----------------------------------------------------
        if gs:
            GS = jnp.sum(c[:, ca_order:], axis=1, keepdims=True)
            if use_bleach:
                lambda_shift, sigma = theta[-2], theta[-1]
                bleach_vec = ModelFunctions._make_bleach(lambda_shift, sigma, gs_spec)  # (λ,)

                gsb_full = GS * bleach_vec         # shape (N,λ)
                alpha = jnp.sum(gsb_full * delA) / jnp.sum(gsb_full * gsb_full)

                delA_minus = delA - alpha * gsb_full
                eps_exc, *_ = jnp.linalg.lstsq(c, delA_minus, rcond=1e-6)

                delA_cal = c @ eps_exc + alpha * gsb_full
                eps = jnp.concatenate([eps_exc, (alpha * bleach_vec)[None, :]], axis=0)
                c = jnp.concatenate([c, GS], axis=1)

            else:
                c = jnp.concatenate([c, GS], axis=1)
                eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
                delA_cal = c @ eps

        # -------- fit amplitudes & build residuals ------------------------------------------------
        else:
            eps, *_ = jnp.linalg.lstsq(c, delA, rcond=1e-6)
            delA_cal = c @ eps

        resid = (delA - delA_cal) * jnp.sqrt(weights)

        if output:
            return delA_cal, c, eps.T          # (N,λ), (N,npool), (λ,npool)
        return resid.ravel()
