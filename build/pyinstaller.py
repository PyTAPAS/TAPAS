# -*- coding: utf-8 -*-
"""
Copyright © 2025, Philipp Frech

This file is part of TAPAS.

    TAPAS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TAPAS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TAPAS.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
from pathlib import Path
import PyInstaller.__main__
import arviz  # imported to locate package data
import tomllib




script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_root = project_root / "src"
tapas_pkg = src_root / "tapas"
os.chdir(project_root)

with open(project_root / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
version = pyproject["project"]["version"]


if sys.platform.startswith('win'):
    icon_file = tapas_pkg / 'assets' / 'icon.ico'
    sep = ';'
elif sys.platform == 'darwin':
    icon_file = tapas_pkg / 'assets' / 'icon.icns'
    sep = ':'
else:
    icon_file = None
    sep = ':'


# data dirs inside your package
assets_data = f"{tapas_pkg/'assets'}{sep}tapas/assets"
configurations_data = f"{tapas_pkg/'configurations'}{sep}tapas/configurations"
log_data = f"{tapas_pkg/'log'}{sep}tapas/log"
controllers_data = f"{tapas_pkg / 'controllers'}{sep}tapas/controllers"
models_data = f"{tapas_pkg / 'models'}{sep}tapas/models"
views_data = f"{tapas_pkg / 'views'}{sep}tapas/views"
utils_data = f"{tapas_pkg / 'utils'}{sep}tapas/utils"

# root‑level license files
license_text = f"{project_root/'LICENSE'}{sep}tapas."
third_party_acknowledgements = f"{project_root/'THIRD-PARTY_ACKNOWLEDGEMENTS.txt'}{sep}tapas."
third_party_license_text = f"{project_root/'THIRD-PARTY_LICENSES.txt'}{sep}tapas."


arviz_path = Path(arviz.__file__).resolve().parent
arviz_static = arviz_path / "static"
arviz_data = arviz_path / "data"
arviz_static_data = f"{arviz_static}{sep}arviz/static"
arviz_data_data = f"{arviz_data}{sep}arviz/data"


entry_point = project_root / "launch_tapas.py"

pyinstaller_args = [
    str(entry_point),
    '--paths', str(project_root / "src"),
    "--clean",
    # Data inclusion
    '--add-data', assets_data,
    '--add-data', configurations_data,
    '--add-data', controllers_data,
    '--add-data', log_data,
    '--add-data', models_data,
    '--add-data', views_data,
    '--add-data', utils_data,
    '--add-data', license_text,
    '--add-data', third_party_acknowledgements,
    '--add-data', third_party_license_text,
    '--add-data', arviz_static_data,  # For static files (HTML templates, etc.)
    '--add-data', arviz_data_data,    # For data files (example_data, etc.)


    # Exclude modules so that they can be edited by user individually
    '--exclude-module', 'tapas.configurations.messages',
    '--exclude-module', 'tapas.controllers.import_controller',
    '--exclude-module', 'tapas.controllers.preproc_controller',
    '--exclude-module', 'tapas.controllers.combine_controller',
    '--exclude-module', 'tapas.controllers.visualize_controller',
    '--exclude-module', 'tapas.controllers.component_controller',
    '--exclude-module', 'tapas.controllers.local_fit_controller',
    '--exclude-module', 'tapas.controllers.global_fit_controller',
    '--exclude-module', 'tapas.controllers.main_controller',
    '--exclude-module', 'tapas.views.main_window',
    '--exclude-module', 'tapas.views.tabs.import_tab',
    '--exclude-module', 'tapas.views.tabs.preprocessing_tab',
    '--exclude-module', 'tapas.views.tabs.combine_data_tab',
    '--exclude-module', 'tapas.views.tabs.visualization_tab',
    '--exclude-module', 'tapas.views.tabs.component_tab',
    '--exclude-module', 'tapas.views.tabs.local_fit_tab',
    '--exclude-module', 'tapas.views.tabs.global_fit_tab',
    '--exclude-module', 'tapas.views.tabwidgets.import_tabwidgets',
    '--exclude-module', 'tapas.views.tabwidgets.preprocessing_tabwidgets',
    '--exclude-module', 'tapas.views.tabwidgets.combine_data_tabwidgets',
    '--exclude-module', 'tapas.views.tabwidgets.visualization_tabwidgets',
    '--exclude-module', 'tapas.views.tabwidgets.component_tabwidgets',
    '--exclude-module', 'tapas.views.tabwidgets.local_fit_tabwidgets',
    '--exclude-module', 'tapas.views.tabwidgets.global_fit_tabwidgets',
    '--exclude-module', 'tapas.utils.model_functions',
    '--exclude-module', 'tapas.utils.utils',


    # Hidden imports
    '--hidden-import', 'PyQt6.QtSvg',
    '--hidden-import', 'PyQt6.QtSvgWidgets',
    '--hidden-import', 'corner',
    '--hidden-import', 'matplotlib.backends.backend_qtagg',
    '--hidden-import', 'mpl_toolkits.axes_grid1',
    '--hidden-import', 'scipy.special',
    '--hidden-import', 'matplotlib.backends.backend_svg',
    '--hidden-import', 'mpl_toolkits.axes_grid1.inset_locator',
    '--hidden-import', 'lmfit',
    '--hidden-import', 'scipy.signal',
    '--hidden-import', 'pyqtconfig',
    '--hidden-import', 'h5py',
    '--hidden-import', 'jax.experimental.ode',
    

    # Application settings:
    '--name', f'tapas_{version}',
    '--noconfirm',
    '--debug=noarchive',
    '--noconsole',
]

if icon_file and icon_file.exists():
    pyinstaller_args.extend(['--icon', str(icon_file)])
if not sys.platform.startswith('win'):
    pyinstaller_args.append('--strip')


PyInstaller.__main__.run(pyinstaller_args)
