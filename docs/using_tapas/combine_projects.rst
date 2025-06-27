Combine Projects
----------------

The **Combine Projects** tab merges multiple datasets—each covering different spectral or temporal regions under similar experimental conditions—onto a single, unified grid. After selecting the source datasets, the user chooses overlap-handling and gap-extrapolation rules. TAPAS then builds the merged data matrix, saves it as a new root dataset in a separate HDF5 project, and records links to all parent projects. The resulting project offers a coherent data surface for downstream analysis while preserving full provenance of every original measurement.  


- **Load Projects Widget**

  - Insert datapath:
  
    Populate the writable line with a TAPAS HDF project by using the ``Browse`` button, pasting the file path or by drag-drop. 

  - ``Browse``:

    Opens a file-dialog where the path to the TAPAS HDF project path can be added.

  - ``use``:

    select which dataset should be used for merging. The valid dataset intervall will be displayed automatically. 

  - ``Load``:

    Load the dataset of the project. 

- **Combine Projects Widget**

  After loading both project files and datasets, a unified dataset can be created. 

  - ``Overlapping Range use``:

    If both datasets have an overlapping range, TAPAS will use either ``Project1``, ``Project2`` or average the values of both projects.

  - ``Interpolation Method``:

    the method used to create the unified grid. uses `scipy.interpolate.interpn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn>`_ for ``nearest``, ``linear``,  ``cubic`` and ``bicubic`` (``splinef2d``)

  - ``Extrapolation Method``:

    select how missing values in the unified grid are handled (eg. at grid edges). 

    - ``zero``:

      set every missing value to zero

    - ``linear``:

      linearly interpolate the missing values using `numpy.interp <https://numpy.org/doc/stable/reference/generated/numpy.interp.html>`_

  - ``Preview``:

    preview the newly created dataset in the canvas. 

  - ``Apply``:

    create the new dataset and set it as raw data of a new project. Caution: unsaved changes to the current project are not saved. The new project links the raw and used data for the merged data. 
