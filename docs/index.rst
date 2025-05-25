TAPAS: Transient Absorption Processing & Analysis Software
==========================================================

Welcome to TAPAS! 

TAPAS is a portable, user-friendly GUI for importing, processing, fitting, and visualizing time-resolved **transient absorption** data.   
Just download and run the cross-platform binaries — no Python environment or pip installs required.  
For power users, every binary ships with the raw Python source and configuration files, so you can tweak kinetic models, extend fitting routines, or build custom analyses for your needs.  
Built on Python with JAX (XLA + JIT), TAPAS performs global fitting in seconds and provides detailed fit statistics, such as standard deviations and parameter correlations. 
It organizes raw inputs (including solvent and steady-state scans), processed results, and analyses into compact HDF5 files with automatically generated metadata for easy and comprehensible management.

**Key features:**  

- **Zero-install portable binaries** for Windows, macOS, and Linux
- **Intuitive GUI** with clear workflow — no command-line or scripting required
- **Flexible processing tools** like resampling, filtering, chirp- & background correction and SVD analysis
- **Fast fitting and error estimation** powered by JAX (XLA + JIT)
- **MCMC-based posterior analysis** of fitted parameters
- **Extendable kinetic models & routines** via editable code and config files
- **Publication-ready plotting** with multiple plot types (e.g., contour maps, kinetics traces, spectral overlays)
- **Combine multiple projects** into unified workspaces
- **HDF5 data bundles** with auto-generated metadata for easy management


This documentation contains a number of usefull resources and is structured as follows: 

- :doc:`Getting Started <getting_started/index>` gives basic information about installation, the concept and architecture of the software.
- :doc:`Using TAPAS <using_tapas/index>` demonstrates and explains every functionality found in TAPAS
- :doc:`Tutorials <tutorial/index>` showcase the workflow routines for different tasks and problems


.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/index

.. toctree::
   :maxdepth: 1
   :hidden:

   using_tapas/index

.. toctree::
   :maxdepth: 1
   :hidden:  

   tutorial/index



