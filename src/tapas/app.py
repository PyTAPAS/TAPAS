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

# Standard Library Imports
import sys
import traceback
import logging
import logging.config  # helps import order
from pathlib import Path
import pkg_resources
import re

# Third‑Party Imports
from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen

# PyQt6 Imports
from PyQt6.QtCore import Qt, QTimer, QByteArray, QSize, QRunnable, QThreadPool, QObject, pyqtSignal
from PyQt6.QtGui import (
    QIcon,
    QPalette,
    QPixmap,
    QPainter
)
from PyQt6.QtSvg import QSvgRenderer
from .configurations import messages as msg


def resource_path(*parts: str) -> Path:
    ''' find where that resource lives at runtime—
    whether in your source tree, an installed wheel, or inside a PyInstaller bundle—and returns
    a `pathlib.Path` pointing to it on disk '''
    if getattr(sys, "frozen", False):
        # PyInstaller: files were collected into _MEIPASS under the same subfolders
        return Path(sys._MEIPASS, "tapas", *parts)
    resource_subpath = "/".join(parts)
    filename = pkg_resources.resource_filename("tapas", resource_subpath)
    return Path(filename)


log_file = resource_path("log", "error.log")
log_args = (str(log_file), "a")
config_ini = resource_path("configurations", "log_config.ini")

logging.config.fileConfig(
    fname=str(config_ini),
    disable_existing_loggers=False,
    defaults=dict(log_args=log_args))


def global_exception_handler(exctype, value, tb) -> None:
    '''
    Handle uncaught exceptions by logging and displaying an error dialog.

    Parameters
    ----------
    exctype : TYPE
        The exception class.
    value : BaseException
        The exception instance.
    tb : traceback
        A traceback object as returned by `sys.exc_info()[2]`..

    Returns
    -------
    None.

    '''
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    logger = logging.getLogger(__name__)
    logger.exception("Unhandled exception occurred",
                     exc_info=(exctype, value, tb))
    QMessageBox.critical(None, "Unhandled Exception", error_msg)

    # sys.exit(1)


def is_dark_theme() -> bool:
    '''
    Determine if the current system theme is dark by using the application's
    style hints if available, or falling back to a brightness heuristic.
    '''
    app = QApplication.instance()
    try:

        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:
        window_color = app.palette().color(QPalette.ColorRole.Window)
        return window_color.value() < 128


def load_svg_icon(path: str, color: str) -> QIcon:
    '''
    Load an SVG file, replace its 'currentColor' placeholders with a given color,
    and render it as a 32×32 QIcon with a transparent background.
    '''
    size = QSize(32, 32)
    with open(path, 'r') as file:
        svg_data = file.read()
    svg_data = svg_data.replace("currentColor", color)
    svg_data = re.sub(r'\s(width|height)="[^"]*"', '', svg_data)
    renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


class PreloadWorker(QRunnable):
    ''' preload the heavy modules '''
    class Signals(QObject):
        finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.signals = PreloadWorker.Signals()

    def run(self):
        from .models.model import AbsModel, EmModel, TAModel, TAModel_ds
        from .controllers.import_controller import ImportController
        from .controllers.preproc_controller import PreprocController
        from .controllers.main_controller import MainController
        from .controllers.visualize_controller import VisualizeController
        from .controllers.combine_controller import CombineController
        from .controllers.local_fit_controller import LocalFitController
        from .controllers.component_controller import ComponentController
        from .controllers.global_fit_controller import GlobalFitController
        from .views.main_window import MainWindow
        import jax.numpy
        import jaxlib.xla_client
        import scipy.signal
        import lmfit

        self.signals.finished.emit()


class App(QApplication):

    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)

    def load_main_window(self):
        ''' main GUI load after initial Splash screen '''

        # -------- lookup preloaded models, controllers and views --------------------------------------------
        from .models.model import AbsModel, EmModel, TAModel, TAModel_ds
        from .controllers.import_controller import ImportController
        from .controllers.preproc_controller import PreprocController
        from .controllers.main_controller import MainController
        from .controllers.visualize_controller import VisualizeController
        from .controllers.combine_controller import CombineController
        from .controllers.local_fit_controller import LocalFitController
        from .controllers.component_controller import ComponentController
        from .controllers.global_fit_controller import GlobalFitController
        from .views.main_window import MainWindow

        # -------- create Model instances ----------------------------------------------------------
        self.abs_model = AbsModel()
        self.em_model = EmModel()
        self.ta_model = TAModel()
        self.ta_model_ds1 = TAModel_ds()
        self.ta_model_ds2 = TAModel_ds()
        self.ta_model_ds3 = TAModel_ds()

        # --------- create Controller instances ----------------------------------------------------

        self.import_controller = ImportController(
            self.abs_model, self.em_model, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.preproc_controller = PreprocController(
            self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.visualize_controller = VisualizeController(
            self.abs_model, self.em_model, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.combine_controller = CombineController(
            self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.local_fit_controller = LocalFitController(
            self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.global_fit_controller = GlobalFitController(
            self.abs_model, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.component_controller = ComponentController(
            self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.main_controller = MainController(
            self.abs_model, self.em_model, self.ta_model, self.ta_model_ds1, self.ta_model_ds2,
            self.ta_model_ds3)

        # -------- create Main View and subsequent Tabs --------------------------------------------
        self.icon_color = "#ffffff" if is_dark_theme() else "#000000"
        self.main_window = MainWindow(
            self.abs_model, self.em_model, self.ta_model, self.ta_model_ds1, self.ta_model_ds2,
            self.ta_model_ds3, self.main_controller, self.import_controller,
            self.preproc_controller, self.visualize_controller, self.combine_controller,
            self.local_fit_controller, self.component_controller, self.global_fit_controller, self.icon_color)

        css_path = resource_path("configurations", "gui_styles.css")
        with open(css_path, "r", encoding="utf-8") as file:
            self.main_window.setStyleSheet(file.read())
        self.main_window.showMaximized()
        icon_path = resource_path("assets", "icon.svg")
        self.setWindowIcon(load_svg_icon(str(icon_path), self.icon_color))


def run():
    """Entry point for launching the GUI."""
    sys.excepthook = global_exception_handler
    app = App(sys.argv)

    splash_image_path = resource_path("assets", "splash.png")
    pix = QPixmap(str(splash_image_path)).scaled(400, 400,
                                                 Qt.AspectRatioMode.IgnoreAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)

    splash = QSplashScreen(pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.setFont(app.font())
    splash.show()

    messages = msg.Status.s_splash
    msg_idx = 0

    def advance_message():
        nonlocal msg_idx
        if msg_idx < len(messages):
            splash.showMessage(
                messages[msg_idx],
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                Qt.GlobalColor.black
            )
            msg_idx += 1
            QTimer.singleShot(2000, advance_message)

    advance_message()

    worker = PreloadWorker()
    worker.signals.finished.connect(lambda: on_preload_done())
    QThreadPool.globalInstance().start(worker)

    def on_preload_done():
        app.load_main_window()
        splash.finish(app.main_window)

    sys.exit(app.exec())


if __name__ == '__main__':
    run()
