Visualization
-------------

In the **Visualization** tab, users create publication-quality figures, including 2D contour maps, ΔA spectral and kinetic slices, and specialized plots for local and global fits. 
Graphical attributes including colors, axis limits and scales, legends, normalizations, figure sizes, and fonts can be adjusted interactively or via Matplotlib’s ``rcParams``. Complete ``matplotlibrc`` files may be loaded to match journal styles.  
Figures export to vector formats (PDF, SVG) or raster images (JPG, PNG, TIFF), and underlying data can be saved as CSV for use in external graphic tools.  

- **Dataset Widget**

  select the dataset to which the processing should be applied. 

- **Plot Widget**

  - ``2D Plot``

    Plot the 2D hypersphere of the dataset

  - ``DeltaA Plot``

    Plot spectral slices at selected delay times

  - ``Kin Trace``

    Plot kinetic slices at selected wavelengths 

  - ``Local Fit``

    - ``Kin Trace``

      Plot fit results at selected local fit wavelength 

    - ``Posterior Dist``

      Plot the corner representation of the results of the MCMC parameter posterior analysis

  - ``Global Fit``

    - ``2D Plot``

      Plot the 2D hypersphere of the fitted data or the residuals of a selected global fit

    - ``EAS/DAS/SAS``

      Plot the Evolution-associated, decay-associated or species-associated spectra of a selected global fit

    - ``Concentration``

      Plot the time-dependendt evolution of the concentration of each component of a selected global fit

    - ``DeltaA Plot``
  
      Plot spectral slices at selected delay times of a selected global fit

  
    - ``Kin Trace``
  
      Plot kinetic slices at selected wavelengths of a selected global fit

    - ``Posterior Dist``

      Plot the corner representation of the results of the MCMC parameter posterior analysis

