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

.. image:: https://img.shields.io/github/last-commit/PyTAPAS/TAPAS.svg
   :target: https://github.com/PyTAPAS/TAPAS/commits/main
   :alt: Last Commit

.. image:: https://img.shields.io/badge/Code%20of%20Conduct-Contributor%20Covenant-4d88ff.svg
   :alt: Code of Conduct
   :target: https://github.com/PyTAPAS/TAPAS/blob/main/CODE_OF_CONDUCT.md

.. image:: https://github.com/PyTAPAS/TAPAS/actions/workflows/codeql-analysis.yml/badge.svg
   :target: https://github.com/PyTAPAS/TAPAS/actions/workflows/codeql-analysis.yml
   :alt: CodeQL Analysis Status

.. image:: https://img.shields.io/pypi/v/pytapas.svg
   :target: https://pypi.org/project/pytapas/
   :alt: PyPI version


What is TAPAS?
==============


Installation Guide
==================

There are three ways to install and run TAPAS

1. Download & Run the Bundled App
2. Install from PyPI
3. Install from Source

Bundled App (GitHub Release)
----------------------------

#. Visit the `Releases page on GitHub <https://github.com/PyTAPAS/TAPAS/releases>`_  
#. Download the ZIP for your platform (e.g. ``TAPAS_vX.Y.Z.zip``).  
#. Extract the ZIP to a folder of your choice.  
#. Run the executable:


No Python installation or environment setup is required.

Install from PyPI
-----------------

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

      python launch_TAPAS.py


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


