Local Fit
---------

The **Local Fit** tab optimizes a single kinetic trace either at a chosen wavelength or across an averaged spectral window.  
The user selects the number of exponential components (and optional non‐decaying offset), kinetic scheme, fitting algorithm, and coherent‐artifact model (none, zero-order Gaussian, or combined zero- and first-order Gaussian).  
A parameter table lists lifetimes, time-zero (t₀), and IRF (Gaussian FWHM), each with initial guess, bounds, and a “fixed” flag.  Time-zero may be defined as the signal’s 5 % onset (practical, but correlated with IRF width) or as the IRF center (analytically precise).
An “initial-guess” run helps expose model inconsistencies before full optimization.  After fitting, TAPAS displays the experimental trace, simulated curve, individual components, and residuals. Optional MCMC-based Bayesian exploration reveals parameter asymmetries or multimodality beyond first-order error estimates.
Results can be saved back to the dataset and forwarded to the Visualization tab.

- **Dataset Widget**

  select the dataset to which the processing should be applied. 

- **Plot Manipulations**

  - ``Delay``

    set the minimum (left) and maximum (right) delay value which will be plotted. The scale (linear, logarithmic or semi-logarithmic) can be set in the drop-down menu. If linlog is selected, a value where the scale switches from linear to logarithmic can be set. 

  - ``ΔA``

    set the minimum (left) and maximum (right) ΔA value in mOD which will be plotted. The scale (linear, logarithmic or semi-logarithmic) can be set in the drop-down menu. If linlog is selected, a value where the scale switches from linear to logarithmic can be set. 

- **Fitting Parameters**

  - ``Wavelength``

    set the wavelength which will be fitted. An additional area can be set over which all the values are averaged. Usefull to reduce noise and fluctuations, but can induce artifacts if multiple spectrally separable features are averaged out. 

  - ``# Components``

    set the number of components used to fit the data.

  - ``Infinite Component?``

    add an optional additional component with a nondecaying lifetime.

  - ``Fit CA``

    either ``false``, ``zero-order`` or ``zero + 1st order``. If ``zero-order`` is selected, coherent artifacts (CAs) are modeled with a Gaussian shaped temporal profile using the fitted t0 and IRF (FWHM of the Gaussian) parameters. If ``zero + 1st order`` is selected, in addition to the Gaussin, CAs are additionally fitted with the derivative of the Gaussian. While this can improve the simulation of the early timepoints, first-order error estimation might not succeed due to the highly correlated CAs in the Jacobian. A future version aims to stabilize the error estimation under 1st order CA modeling. 

  - ``Model``

    set the kinetic model used to fit the data 

  - ``Time Zero``

    set the time zero definition to either be the beginning of the signal rise (``5% Threshold``) or the center of the Gaussian IRF function (``Gaussian Center``). This affects the representation of the t0 parameter. Internally, the center IRF value will be used for fitting since it is analytically exact and the derived parameters (IRF=FWHM and t0) are uncorrelated. However, the ``5% Threshold`` value might be more intuitive and easier to estimate. 

  - ``Method``

    set the minimization algorithm. The method is used by the `lmfit <https://lmfit.github.io/lmfit-py/fitting.html>`_ package which internally calles `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_. ``nelder`` does not require derivatives and is a robust method for exponential problems, however it might be slower than ``leastsq`` which estimates gradients to find minima quicker. However it might fail if the problem is ill-conditioned. ``diff-evol`` calls the global optimizer differential-evolution which requires preset bounds of every parameter and is generally slow. However it aims to find the global minimum and is less succeptible to be trapped in local minimma due to initial guesses. 

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

