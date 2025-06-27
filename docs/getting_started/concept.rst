Concept
=======

**TAPAS** (Transient Absorption Processing and Analysis Software) is implemented entirely in **Python** with a **PyQt** front-end, providing a native look-and-feel on every major platform. Pre-built application bundles are distributed for Windows, macOS and Linux; users can launch the graphical interface immediately. No Python interpreter, package manager or command-line setup is required.  Despite this convenience, the bundled directory keeps all source files readable. Advanced users may open a module, modify a kinetic model or plotting routine, and simply restart the programme to apply their changes. A detailed map of the code base showing where each layer (data I/O, preprocessing, fitting, visualisation) resides and what responsibility it carries is given later in the :ref:`software_architecture` section.

User-interface overview
-----------------------

The GUI is organised into **seven workflow tabs**, mirroring the standard TA analysis pipeline:

1. **Import** – drag-and-drop raw TA, solvent and steady-state files.  
2. **Preprocessing** – time-zero alignment, chirp correction, trimming and resampling.  
3. **Combine Projects** - fuse datasets covering different spectral / temporal regions.
4. **Visualization** – publication-ready spectra, kinetics, fit results and correlation matrices.  
5. **Components** – singular-value plots, kinetic traces and residual maps.  
6. **Local Fit** – parallel or sequential exponential models with statistical diagnostics at a given (averaged) wavelength.  
7. **Global Fit** – parallel, sequential or target model with statistical diagnostics over the full ΔA matrix


A tab-by-tab walk-through of every control, option and dialogue box appears in the :ref:`using-tapas` section.

