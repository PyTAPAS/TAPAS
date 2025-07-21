Visualization
-------------

In the **Visualization** tab, users create publication-quality figures, including 2D contour maps, ΔA spectral and kinetic slices, and specialized plots for local and global fits. 
Graphical attributes including colors, axis limits and scales, legends, normalizations, figure sizes, and fonts can be adjusted interactively or via Matplotlib’s ``rcParams``. Complete ``matplotlibrc`` files may be loaded to match journal styles.  
Figures export to vector formats (PDF, SVG) or raster images (JPG, PNG, TIFF), and underlying data can be saved as CSV for use in external graphic tools.  

- **Dataset Widget**

  select the dataset to which the processing should be applied. 

- **Plot Widget**

  select the plot type. 


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

- **Figure Style Widget**

  Figure styling widget, applied to every plot type and used for exporting

  - ``Add rcParam``
  
    add a matplotlib rcParam key and value pair to temporarily overwrite the current rcParams style sheet. In the toolbar, a new or custom rcParams style sheet can be loaded in (see :ref:`main-window` for mor information). More infomrations on the various rcParams can be found in the matplotlib  `user guide <https://matplotlib.org/stable/users/explain/customizing.html>`_.

  - ``Figure Size``

    Set the figure dimensions: first entry width, second entry height. Accepted units are inch ("), mm or cm. 

  - ``Show Real Size?``

    If checked, the figure canvas will be adjusted to match the actual real-world figure dimensions.  

  - ``Figure Font``

    Select one preinstalled font on your local machine

  - ``Font Style``

    Select one available font style of the selected and installed font family

  - ``Label Size``

    Set the size of all axes labels, legends, and titles in point. 

  - ``Tick Size``

    Set the size of all tick labels in point.

  - ``Format``

    Select a vector graphic or rastergraphic format. If a rastergraphic (e.g. png, jpg, tiff) is selected, the DPI value can be set subsequently. If a vector graphics format is selected, the text information can be saved as text or as Path. Rendering as text gives the opportunity to change the text with a suitable vector graphics editing program like Inkscape. However, different image displaying programs might change the font style and family slightly if the font is not available on the target device. Saving as path burrys the text information as object information, so no text editing is possible afterwards. However, the text will always be displayed as expected and no preinstalled font is necessary at the target device. 

  - ``Result Dir``

    Set the path, where the figure will be saved to. If none is set, the project path will be used or a file dialog will open up. 

  - ``Save``

    Save the figure with the current settings to the current directory. Depending on the metadata, filenames and subfolders will be automatically created. 

  - ``Save as..``

    Opens a file dialog where the user can select a directory to which the figure will be saved to. 

  - ``export data``

    In the figure canvas, ``export data`` saves the data to produce the current figure to CSV using the results directory set in the ``Figure Style Widget``. 

