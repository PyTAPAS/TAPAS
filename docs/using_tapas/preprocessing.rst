Preprocessing
=============

The **Preprocessing** tab presents an interactive 2D view of the transient-absorption surface, with a crosshair that updates both the ΔA vs. time trace and the ΔA vs. wavelength slice.  Users can trim, resample, subtract background/solvent, align time zero, filter, and correct for group-velocity dispersion (“chirp”). Each operation generates a new **dataset** linked to its immutable raw data, with structured metadata recorded automatically. Exporting to HDF5 preserves the complete raw–dataset hierarchy and metadata, ensuring full provenance and reproducibility.  



- **Dataset Widget**

  select the dataset to which the processing should be applied. 

- **Views Widget**

  - ``Full View``

    Plots the full range of the dataset

  - ``Time Zero View``

    Plots the temporal range close to zero. Good for inspection of the chirp.

  - ``Manual View``

    Internally used by TAPAS when data is trimmed. 

  - Toolbox above the Canvas

    Additionally, the displayed view can be adjusted by the arrow and magnifying glass tools above the canvas.

- **View Manipulations Widget**

  Set a delay-time (eg. 2ps, 2e-12, 2µs) where the delay-time axis should switch from a linear to a logarithmic time-scale.


- **Trim Data Widget**

  set a wavelength or delay-time interval. If no value is provided, the minimum/maximum value of the dataset is used. The Canvas will be updated immediately. Pressing ``Apply`` will save the new interval to the selected dataset. 

- **Resampling Widget**

  - ``Resample``: 

    Down or upsampe the data grid. 

    - ``Intervall``:

      Set the intervall where the resampling should be applied. 

    - ``Factor``:

      the sampling factor (eg. 0.1 means downsampling to 10% of datapoints, 3 means subsamling to 3 times the datapoints). 

    - ``Axis``:

      the axis (wavelength or delay time) over which the sampling will be applied

    - ``Method``: 

      interpolation method: 
      see `scipy.interpolate.make_interp_spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html>`_ for further information on the ``linear``, ``quadratic`` and ``cubic`` interpolation method, `scipy.interpolate.pchip_interpolate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html>`_ for ``pchip`` interpolation  and `scipy.interpolate.Akima1DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_ for ``akima`` and ``makima`` interpolation.


  - ``Regularize``: 

    Regularize the grid with equally spaced data points. 

    - ``Datapoints``: 

      Number of datapoints of the regularized grid

    - ``Axis``: 

      the axis (wavelength or delay time) over which the sampling will be applied. 

    - ``Method``: 

      interpolation method 
      see `scipy.interpolate.make_interp_spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html>`_ for further information on the ``linear``, ``quadratic`` and ``cubic`` interpolation method, `scipy.interpolate.pchip_interpolate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html>`_ for ``pchip`` interpolation  and `scipy.interpolate.Akima1DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_ for ``akima`` and ``makima`` interpolation.


- **Chirp Correction Widget**

  - ``Autocorrect``: 

    automatically find points in the dataset given a preset heuristic (eg. x % rise, x mOD threshold, maximum value).

  - ``Manually Correct``:

    when activated, chirppoints used for fitting can be added by left-click and deleted by right-click in the 2D data canvas. 

  - ``Correct from Project``:

    load the chirp fitting coefficients of another project and dataset with a file dialog. 

  - ``Fit``:

    fit a 4th order polynomial to the given chirp points. a minimum of 5 points must be provided

  - ``Show Correction``:

    after a successfull fit, the chirp correction can be displayed in the cavas with this button. 

  - ``Apply``:

    Apply the fitted chirp line to the dataset via interpolation. 

- **Background Correction Widget**

  - ``Subtract Area``:

    use an averaged spectrum to subtract from the dataset.

    - ``Intervall``:

      select the delay-time interval which will be used for the averaged background.

    - ``Subtract up to``: 

      set the maximum delay-time up to which the background will be subtracted. 

    - ``Method``: 

      averaging method

  - ``Subtract from Project``:

    load a 2D surface from another project and dataset. Only valid, if both datasets have the same dimensions. 

  - ``Subtract Solvent``:

    Subtract the solvent data imported in the ``Import`` tab. Only valid, if both datasets have the same dimensions. 

- **Time Zero Correction Widget**

  - ``Time-Point``:

    Delay time vector will be shifted so that ``Time-Point`` equals zero. 

- **Averaging and Filtering Widget**

  - ``Savitzky-Golay``:

    Apply a Savitzky-Golay filter `scipy.signal.savgol_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_

    - ``Window``:

      The length of the filter window. 

    - ``Order``:

      Order of the polynomial used to fit the data. 

    - ``Axis``:

      the axis (wavelength or delay time) over which the sampling will be applied. 
    

  - ``Moving Median``:

    Apply a median filter `scipy.ndimage.median_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html>`_

    - ``Size``:

      Filter subset length. 

    - ``Axis``:

      the axis (wavelength or delay time) over which the sampling will be applied. 

  - ``Moving Average``:

    Apply a uniform filter `scipy.ndimage.uniform_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html>`_

    - ``Size``:

      Filter subset length. 

    - ``Axis``:

      the axis (wavelength or delay time) over which the sampling will be applied. 

