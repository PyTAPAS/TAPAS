Import Tab
==========

In the **Import** tab, users add raw TA, solvent TA, and steady‐state files by file dialog, path paste, or drag-and-drop.  TAPAS then:

- Parses input with the chosen delimiter  
- Builds wavelength and delay-time vectors and a Δ absorbance matrix  
- Extracts trailing metadata and converts all values to SI units  

The result is a standardized, machine-readable HDF5 dataset with separate metadata.  Multiple TA scans can be averaged automatically, and pre-/post-TA steady-state measurements are evaluated to flag degradation.  User-defined metadata (e.g., experiment name, sample ID, excitation wavelength) populates plot titles, output directories, and annotations such as scattered pump-light indicators.```


- **Imports Widget**

  - Insert datapath 
  
    Populate the writable line with one or multiple data paths by using the ``Browse`` button, pasting the file paths or by drag-drop. 

  - ``Browse`` 

    Opens a file-dialog where the path to one or multiple data files can be added. Accepted are all readable text files regardless of the extension (.csv, .txt, .dat etc).

  - ``Delimiter`` 

    Set the separation used in the data file (eg, comma-separated, semicolon, tab etc.)

  - ``Ignore Header``  

    Sometimes the first lines can contain textual information. These should be ignored during import by selecting the number of lines to exclude.

  - ``Time Unit`` 

    Unit of the delay time. TAPAS internally uses SI units. 

  - ``Energy Unit`` 

    Unit of the spectral axis. TAPAS internally uses SI units. 

  - ``ΔA Unit`` 

    Unit of the differential absorption. TAPAS internally uses OD. 

  - ``Data Orientation``

    Generally, TA data should be structured by displaying the spectral vector as the first row/column and the temporal vector as the first column/row. All the other elements are the ΔA values of the corresponding wavelength-delay tuple. This orientation can be set here. 

  - ``Load Data``

    Loads the data from the preset path with the corresponding settings and plots the data if everything was set correctly. 

  - ``Clear``

    Clears the raw data and all linked sub data. 

  - Multiple TA data files

    if multiple TA data files are provided, TAPAS asks wether to average all files or only use the last data file. This only works if all the files have the same dimensions.

  - Steady-State Data

    TAPAS can handle up to two absorbance or emission files or two datasets (containing a wavelength and a data vector) within one file. TAPAS will treat the first dataset as "data before the TA experiment" and the second dataset as "data after the TA experiment". This can be usefull for tracing degradation during the TA erxperiments.


- **Metadata Widget**

  Insert textual information which will be linked to the raw TA data. This can be used for automatic directory creation during export, or for visualization. TAPAS will try to extract possible metadata at the end of the imported TA text file. 
