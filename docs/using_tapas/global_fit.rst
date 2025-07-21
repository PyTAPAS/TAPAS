Global Fit
----------

The **Global Fit** tab extends local fitting across the full time–wavelength matrix.  
In addition to the Local Fit options, users may include a ground-state recovery term (optionally constrained by an external steady-state spectrum), apply wavelength-dependent weighting, and preview the kinetic scheme diagram.  
A global fit produces decay-associated spectra (DAS) and evolution-associated spectra (EAS), or species-associated spectra (SAS) when using a target model, along with concentration profiles, a residual map, and the SVD of residuals.  
Fit parameters are reported with absolute and relative standard errors, plus a summary of fit quality, parameter correlations, and stability diagnostics.  Optional MCMC-based Bayesian exploration reveals parameter asymmetries or multimodality beyond first-order error estimates.
Results can be saved back to the dataset and forwarded to the Visualization tab.


- **Dataset Widget**

  select the dataset to which the processing should be applied. 

- **View Manipulations**

  - ``Lin/Log Transition``

    set the time where the delay time axis switches from a linear to a logarithmic scale. 

  - ``Normalize DAS/EAS/SAS``

    if checked, the associated spectra are all normalized to 1. 

  - ``Percent Residuals``

    if checked, the fit residuals are normalized to their amplitude to give a relative deviation in percentage.

- **Fitting Parameters**

  - ``# Components``

    set the number of components used to fit the data.

  - ``Infinite Component?``

    add an optional additional component with a nondecaying lifetime.

  - ``Explicit GSB``

    if checked, an additional component will be modeled whose concentration profile is the sum of all excited species. This can be used to model the ground-state recovery. If steady-state data is given and the grid is regular along the wavelength axis (can be regularized in the ``Preprocessing`` Tab), the steady-state data can be used as a template for the spectral profile of this component when ``Use Steady-State`` is checked. During optimization, a small wavelength shift (``gs_shift``) and a boradening (``gs_sigma``) of the template is accepted.

  - ``Fit CA``

    either ``false``, ``zero-order`` or ``zero + 1st order``. If ``zero-order`` is selected, coherent artifacts (CAs) are modeled with a Gaussian shaped temporal profile using the fitted t0 and IRF (FWHM of the Gaussian) parameters. If ``zero + 1st order`` is selected, in addition to the Gaussin, CAs are additionally fitted with the derivative of the Gaussian. While this can improve the simulation of the early timepoints, first-order error estimation might not succeed due to the highly correlated CAs in the Jacobian. A future version aims to stabilize the error estimation under 1st order CA modeling. 

  - ``Model``

    set the kinetic model used to fit the data 

  - ``Microsteps``

    Number of RK4 sub-steps to take within each interval `delay[i] → delay[i+1]`. Higher values improve integration accuracy (smaller local error per interval) at the cost of proportional extra compute. Defaults to 10.

  - ``Time Zero``

    set the time zero definition to either be the beginning of the signal rise (``5% Threshold``) or the center of the Gaussian IRF function (``Gaussian Center``). This affects the representation of the t0 parameter. Internally, the center IRF value will be used for fitting since it is analytically exact and the derived parameters (IRF=FWHM and t0) are uncorrelated. However, the ``5% Threshold`` value might be more intuitive and easier to estimate. 

  - ``Method``

    set the minimization algorithm. The method is used by the `lmfit <https://lmfit.github.io/lmfit-py/fitting.html>`_ package which internally calles `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_. ``nelder`` does not require derivatives and is a robust method for exponential problems, however it might be slower than ``leastsq`` which estimates gradients to find minima quicker. However it might fail if the problem is ill-conditioned. ``diff-evol`` calls the global optimizer differential-evolution which requires preset bounds of every parameter and is generally slow. However it aims to find the global minimum and is less succeptible to be trapped in local minimma due to initial guesses. 

  - ``Weights``

    wavelength areas can be downweighted to help the optimizer find the minimum and to get realistic error estimates. The weightening will be used for evaluating the residuals according to  resid = (delA - delA_cal) * sqrt(weights). This can be usefull to downweight areas of high scattered light, detector noise or interpolated areas between two datasets. If a custom weightening is applied, the unweighted error estimates are also calculated and printed in the ``Fitting Results`` widget. 

  - ``Paramter Table``

    set the initial guesses and bounds for the parameters. This table will automatically resize depending on the presset number of components. In the first column, the initial guessed values can be set, in the second and third column, the lower and upper bound can be set respectively. The ``vary`` flag is true by default but can be set to false. This will freeze the parameter to the inital value and exclude it from the optimization. 

  - ``Show Initial Guess``

    Plots the selected model with the initial guessed paramter values. Good to evaluate the quality of the selected model, paramters and estimates.  

  - ``Fit``

    Runt the optimization. If the optimization does not succeed, an error message and suggestion will be displayed in the status bar and the ``Fitting Results` widget will indicate that the fitting did not succeed.

- **Fitting Results**

  If the optimization succeeded, the parameters and error estimates will be displayed in this widget. 

  - ``export fitting results``

    saves the printed fitting results of a current or selected fit to TXT using the project directory

  - ``Save Fit to Dataset``

    saves the fitting results to the dataset and project, so that it can be reloaded, used for MCMC analysis or used in the ``Visualization`` tab. 

- **Fitting List**

  If the fit was saved using ``Save Fit to Dataset``, it will appear in this widget. Selecting a fit, will display its data in the ``Fitting Preview`` and ``Fitting Results`` widget. 

  - ``delete Selected Fit``

    removes the selected fit from the dataset and project. 

- **Explore Parameter Space**

  Performs an MCMC posterior analysis of a selected fit in the ``Fitting List`` widget using `lmfit <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee>`_ and `emcee <https://emcee.readthedocs.io/en/stable/>`_. 

  - ``discard the first``

    number of samples to discard at the beginning of the sampling regime. 

  - ``Initialize``

    Number of inital samples drwan from the distribution. 

  - ``Accept one per``

    Thins the samples to use one every x samples

  - ``Target Ratio``

    the resulting flatted chain should be longer than x time the integrated autocorrelation time. This is the termination criterion TAPAS will use to evaluate if the expolartion needs more runs or finished successfully. 

  - ``Perform Analysis``

    Start the MCMC posterior paramter exploration with the current settings. First, the number of steps set in ``discard the first`` will be performed to find a good starting point and to estimate the time needed per step. Second, a first round with the number of steps set in ``Initialize`` is performed to estimate the needed time and number of samples for reliable estimates depending on the set target ratio. Then the number of estimated samples will be run and added to the first round. If the ``Target Ratio`` criterion isn't fullfilled yet, another round will be performed.

  - ``Abort Analysis``

    If clicked, the analysis will be aborted after the next round and saved. This can already take a substatial amount of time depending on the number of steps evaluated in the relevant round. 

  - ``Save Analysis``

    saves the analysis to the fit, to make it accessible in the ``Visualization`` tab. 



