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

import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    # running in a PyInstaller bundle
    # sys._MEIPASS is where the bundle’s code & library.zip live
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    sys.path.insert(0, str(bundle_dir))
else:
    # normal dev
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "src"))

from tapas.app import run

if __name__ == "__main__":
    run()
