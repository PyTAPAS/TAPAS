.. figure:: https://raw.githubusercontent.com/PyTAPAS/TAPAS/main/src/tapas/assets/splash.png
   :alt: Project Icon
   :align: center
   :width: 200px

TAPAS
=====

**Transient Absorption Processing & Analysis Software**


.. image:: https://readthedocs.org/projects/tapas-docs/badge/?version=latest
   :target: https://tapas-docs.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-GPLv3-blue.svg
   :target: https://github.com/PyTAPAS/TAPAS/blob/main/LICENSE
   :alt: License (GPLv3)

.. image:: https://img.shields.io/badge/Code%20of%20Conduct-Covenant%20v2.1-4d88ff.svg
   :alt: Code of Conduct
   :target: https://github.com/PyTAPAS/TAPAS/blob/main/CODE_OF_CONDUCT.md

.. image:: https://github.com/PyTAPAS/TAPAS/actions/workflows/codeql-analysis.yml/badge.svg
   :target: https://github.com/PyTAPAS/TAPAS/actions/workflows/codeql-analysis.yml
   :alt: CodeQL Analysis Status

.. image:: https://img.shields.io/github/last-commit/PyTAPAS/TAPAS.svg
   :target: https://github.com/PyTAPAS/TAPAS/commits/main
   :alt: Last Commit


.. image:: https://img.shields.io/github/v/release/PyTAPAS/TAPAS?label=Latest%20Release
   :target: https://github.com/PyTAPAS/TAPAS/releases/latest
   :alt: Latest Release

.. image:: https://img.shields.io/static/v1?label=Download&message=Windows&color=blue&logo=windows&logoColor=white
   :target: https://github.com/PyTAPAS/TAPAS/releases/latest
   :alt: Download for Windows

.. image:: https://img.shields.io/static/v1?label=Download&message=Linux&color=orange&logo=linux&logoColor=white
   :target: https://github.com/PyTAPAS/TAPAS/releases/latest
   :alt: Download for Linux

.. image:: https://img.shields.io/static/v1?label=Download&message=macOS&color=black&logo=apple&logoColor=white
   :target: https://github.com/PyTAPAS/TAPAS/releases/latest
   :alt: Download for macOS

.. image:: https://img.shields.io/pypi/v/pytapas.svg
   :target: https://pypi.org/project/pytapas/
   :alt: PyPI version


What is TAPAS?
==============

TAPAS is a portable, user-friendly GUI for importing, processing, fitting, and visualizing time-resolved **transient absorption** data.   
Just download and run the cross-platform binaries — no Python environment or pip installs required.  
For power users, every binary ships with the raw Python source and configuration files, so you can tweak kinetic models, extend fitting routines, or build custom analyses for your needs.  
Built on Python with JAX (XLA + JIT), TAPAS performs global fitting in seconds and provides detailed fit statistics, such as standard deviations and parameter correlations. 
It organizes raw inputs (including solvent and steady-state scans), processed results, and analyses into compact HDF5 files with automatically generated metadata for easy and comprehensible management.

**Key features:**  

- **free and Open-Source** software
- **Zero-install portable binaries** for Windows, macOS, and Linux
- **Intuitive GUI** with clear workflow — no command-line or scripting required
- **Flexible processing tools** like resampling, filtering, chirp- & background correction and SVD analysis
- **Fast fitting and error estimation** powered by JAX (XLA + JIT)
- **MCMC-based posterior analysis** of fitted parameters
- **Extendable kinetic models & routines** via editable code and config files
- **Publication-ready plotting** with multiple plot types (e.g., contour maps, kinetics traces, spectral overlays)
- **Combine multiple projects** into unified workspaces
- **HDF5 data bundles** with auto-generated metadata for easy management


Installation Guide
==================

There are three ways to install and run TAPAS

1. Download & Run the Bundled App
2. Install from PyPI
3. Install from Source

Bundled App (GitHub Release)
----------------------------

This is the recommended way for non python users. 

#. Visit the `Releases page on GitHub <https://github.com/PyTAPAS/TAPAS/releases>`_  
#. Download the ZIP for your platform (e.g. ``TAPAS_vX.Y.Z.zip``).  
#. Extract the ZIP to a folder of your choice.
#. Your OS might flag the app since we don't purchase a commercial code-signing certificate. You can safetly ignore these warnings
#. Run the executable. The first cold start might take several seconds.


No Python installation or environment setup is required. Because this application is built and distributed directly from source and isn’t signed by a commercial certificate authority, you may see warnings from your OS when you first download or launch it. This is expected and does **not** indicate any malware.


The source code can be found under::

      _internal/tapas

and can be changed or extended to satify user specific needs. More information can be found `here <https://tapas-docs.readthedocs.io/en/latest/>`_.


Install from PyPI
-----------------

#. Check Interpreter Version:

   This package requires ``Python=3.11`` due to our PyQt6 dependency::

      python --version
      conda create -n myenv python=3.11
      conda activate myenv

#. Install via ``pip``::

      pip install pytapas

#. Launch the GUI:

   * Use the console script::

         tapas

   * Or invoke as a module::

         python -m tapas

   Both commands start the same TAPAS graphical interface.


From Source (Development Workflow)
----------------------------------

#. Obtain the source:

   * Clone the repo::

         git clone https://github.com/PyTAPAS/TAPAS.git
         cd TAPAS

   * **OR** download *Source code (zip)* from GitHub and extract it.

#. Create and activate a virtual environment:

   * **Windows (cmd.exe)**::

         python -m venv .venv
         .venv\Scripts\activate

   * **Windows (PowerShell)**::

         python -m venv .venv
         .venv\Scripts\Activate.ps1

   * **macOS / Linux**::

         python3 -m venv .venv
         source .venv/bin/activate

#. Install dependencies and the editable package::

      pip install --upgrade pip
      pip install -e .

   (This reads ``pyproject.toml`` and installs all required dependencies.)

#. Launch TAPAS::

      python launch_tapas.py


Documentation
=============

A detailed documentation can be found
`here <https://tapas-docs.readthedocs.io/en/latest/>`_.


License
=======

Copyright 2025 Philipp Frech

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


