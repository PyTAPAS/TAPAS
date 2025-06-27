Local Fit
---------

The **Local Fit** tab optimizes a single kinetic trace—either at a chosen wavelength or across an averaged spectral window.  
The user selects the number of exponential components (and optional non‐decaying offset), kinetic scheme, fitting algorithm, and coherent‐artifact model (none, zero-order Gaussian, or combined zero- and first-order Gaussian).  
A parameter table lists lifetimes, time-zero (t₀), and IRF (Gaussian FWHM), each with initial guess, bounds, and a “fixed” flag.  Time-zero may be defined as the signal’s 5 % onset (practical, but correlated with IRF width) or as the IRF center (analytically precise).
An “initial-guess” run helps expose model inconsistencies before full optimization.  After fitting, TAPAS displays the experimental trace, simulated curve, individual components, and residuals. Results can be saved back to the dataset or forwarded to the Visualization tab.
