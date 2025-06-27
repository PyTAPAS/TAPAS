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

and can be changed or extended to satify user specific needs. More information can be found :ref:`here <software_architecture>`.


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


