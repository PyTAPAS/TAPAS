Preprocessing
=============

The **Preprocessing** tab presents an interactive 2D view of the transient-absorption surface, with a crosshair that updates both the ΔA vs. time trace and the ΔA vs. wavelength slice.  Users can trim, resample, subtract background/solvent, align time zero, filter, and correct for group-velocity dispersion (“chirp”). Each operation generates a new **dataset** linked to its immutable raw data, with structured metadata recorded automatically. Exporting to HDF5 preserves the complete raw–dataset hierarchy and metadata, ensuring full provenance and reproducibility.  



- **Dataset**

  select the dataset to which the processing should be applied. 

- **Views**

  - ``Full View``

    Plots the full range of the dataset

  - ``Time Zero View``

    Plots the temporal range close to zero. Good for inspection of the chirp.

  - ``Manual View``

    Internally used by TAPAS when data is trimmed. 

  - Toolbox above the Canvas

    Additionally, the displayed view can be adjusted by the arrow and magnifying glass tools above the canvas.

- **View Manipulations**

  Set a delay-time (eg. 2ps, 2e-12, 2µs) where the delay-time axis should switch from a linear to a logarithmic time-scale.


- **Trim Data**

  set a wavelength or delay-time interval. If no value is provided, the minimum/maximum value of the dataset is used. The Canvas will be updated immediately. Pressing ``Apply`` will save the new interval to the selected dataset. 

- **Resampling**

  - ``Resample``

    Down or upsampe the data grid. 
    Set the intervall where the resampling should be applied. 

    ``Factor``: the sampling factor (eg. 0.1 means downsampling to 10% of datapoints, 3 means subsamling to 3 times the datapoints). 

    ``Axis``: the axis (wavelength or delay time) over which the sampling will be applied

    ``Method``: interpolation method 
    see `scipy.interpolate.make_interp_spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html>`_ for further information on the ``linear``, ``quadratic`` and ``cubic`` interpolation method, `scipy.interpolate.pchip_interpolate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html>`_ for ``pchip`` interpolation  and `scipy.interpolate.Akima1DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_ for ``akima`` and ``makima`` interpolation.




