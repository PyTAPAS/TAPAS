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

  - ``ΔA Plot``

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

    - ``ΔA Plot``
  
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

- **Plot Properties**

  The Plot Properties change depending on the plot type selected in the ``Plot Widget``

  - ``2D Plot``

    - ``Wavelength``
      set the minimum (left) and maximum (right) wavelength value which will be plotted. Leaving this empty will use the minimum / maximum value available in the dataset. 

    - ``Delay``
      set the minimum (left) and maximum (right) delay value which will be plotted. Leaving this empty will use the minimum / maximum value available in the dataset. 

    - ``ΔA``
      set the minimum (left) and maximum (right) ΔA value in mOD which will be plotted. Leaving this empty will use the minimum / maximum value available in the dataset. The spinbox sets the center of the colorbar in mOD. 

    - ``Lin/Log Transition``
      set delay time from where on the delay axis changes from a linear to a logarithmic scale. 

    - ``Ratio (Lin|Log|ss)``
      set the relative height of the linear, logarithmic and steady-state (optional) areas of the plot

    - ``Hide Area``
      set the minimum (left) and maximum (right) x-axes value which will be left empty in the figure. Usefull for hiding scattered light area or areas of low detector sensitivity.

    - ``Colormap``
      ticking the checkbox will display the colorbar. Different colorschemes can be selected in the dropdown menu, ``right`` or ``bottom`` will locate the colorbar accordingly. 

    - ``Display ΔA Cuts ``
      if checked, the selected ΔA cuts in the ``ΔA Plot`` window are displayed with the related colors.

    - ``Display Kinetic Cuts``
       if checked, the selected kinetic cuts in the ``Kin Trace`` window are displayed with the related colors. 

    - ``Display 2nd Axis``
      plots a second energy axis on top in eV. 

    - ``Display Metadata``
      displays the metadata set in the ``Import Tab`` as the title.

    - ``Display Pump``
      displays the pump wavelength as a vertical line. the pump wavelength needs to be set first in the ``Import Tab``.


  - ``DeltaA Plot``

    - ``Wavelength``
      set the minimum (left) and maximum (right) wavelength value which will be plotted. Leaving this empty will use the minimum / maximum value available in the dataset. 

    - ``ΔA``
      set the minimum (left) and maximum (right) ΔA value in mOD which will be plotted. The scale (linear, logarithmic or semi-logarithmic) can be set in the drop-down menu. If linlog is selected, a value where the scale switches from linear to logarithmic can be set. 

    - ``Show Legend``
      if ticked, displays the legend at the current positional argument. ``Outside`` positions the legend on the right next to the figure, ``best`` lets matplotlib decide where to plot the legend and the other arguments display the legend accordingly. 

    - ``Delay Times``
      set the delay times which will be displayed in the figure, separated by commas. 

    - ``Colors``
      the field on the right side accepts named matplotlib colors (e.g. blue, r, coral, etc) or hex values (e.g. #123456). Each color value correspond to one slice and must be separated by a comma. Further, a gradient colorscheme or the default color cylcer of the current global style (-) can be used. If the number of custom colors set is lower than the slices, the current gradient colorschme is used.  

    - ``Hide Area``
      set the minimum (left) and maximum (right) x-axes value which will be left empty in the figure. Usefull for hiding scattered light area or areas of low detector sensitivity.

    - ``Display 2nd Axis``
      if checked, plots a second energy axis on top in eV. 

    - ``Display Metadata``
      if checked, displays the metadata set in the ``Import Tab`` as the title.

    - ``Display Pump``
      if checked, displays the pump wavelength as a vertical line. the pump wavelength needs to be set first in the ``Import Tab``.

  - ``Kin Trace``

    - ``Delay``
      set the minimum (left) and maximum (right) delay value which will be plotted. The scale (linear, logarithmic or semi-logarithmic) can be set in the drop-down menu. If linlog is selected, a value where the scale switches from linear to logarithmic can be set. 

    - ``ΔA``
      set the minimum (left) and maximum (right) ΔA value in mOD which will be plotted. The scale (linear, logarithmic or semi-logarithmic) can be set in the drop-down menu. If linlog is selected, a value where the scale switches from linear to logarithmic can be set. 

    - ``Show Legend``
      if ticked, displays the legend at the current positional argument. ``Outside`` positions the legend on the right next to the figure, ``best`` lets matplotlib decide where to plot the legend and the other arguments display the legend accordingly. 

    - ``Wavelengths``
      set the wavelengths which will be displayed in the figure, separated by commas. 

    - ``Colors``
      the field on the right side accepts named matplotlib colors (e.g. blue, r, coral, etc) or hex values (e.g. #123456). Each color value correspond to one slice and must be separated by a comma. Further, a gradient colorscheme or the default color cylcer of the current global style (-) can be used. If the number of custom colors set is lower than the slices, the current gradient colorschme is used.  

    - ``Normalize``
      if checked, the ΔA values are normalized so that the maximum absolute value is 1 within the given interval. 

    - ``Absolute Values``
      if checked, the absolute (only positive) values are displayed. Might be valuable when comparing a bleaching and an induced absorption signal. 

    - ``Display Metadata``
      if checked, displays the metadata set in the ``Import Tab`` as the title.


  - ``Kin Trace (Local Fit)``

    Plot fit results at selected local fit wavelength 

  - ``Posterior Dist (Local Fit)``

    Plot the corner representation of the results of the MCMC parameter posterior analysis

  - ``2D Plot (Global Fit)``

    Plot the 2D hypersphere of the fitted data or the residuals of a selected global fit

  - ``EAS/DAS/SAS (Global Fit)``

    Plot the Evolution-associated, decay-associated or species-associated spectra of a selected global fit

  - ``Concentration (Global Fit)``

    Plot the time-dependendt evolution of the concentration of each component of a selected global fit

  - ``DeltaA Plot (Global Fit)``

    Plot spectral slices at selected delay times of a selected global fit


  - ``Kin Trace (Global Fit)``

    Plot kinetic slices at selected wavelengths of a selected global fit

  - ``Posterior Dist (Global Fit)``

    Plot the corner representation of the results of the MCMC parameter posterior analysis
