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
import os
import sys
import pkg_resources
import json
import re
from pathlib import Path
import shutil

# Third‑Party Imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyqtconfig import ConfigManager

# PyQt6 Imports
from PyQt6.QtCore import Qt, pyqtSlot, QUrl, QByteArray, QSize, QProcess, QCoreApplication, QProcessEnvironment
from PyQt6.QtGui import (
    QIcon,
    QAction,
    QDesktopServices,
    QKeySequence,
    QPixmap,
    QPainter
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QToolBar,
    QFileDialog,
    QPushButton,
    QMenu,
    QLabel,
    QMessageBox,
    QDialog,
    QTextEdit,
    QVBoxLayout
)

# Local Application Imports
from ..configurations import messages as msg
from ..views.tabs.import_tab import ImportTab
from ..views.tabs.preprocessing_tab import PreprocTab
from ..views.tabs.visualization_tab import VisualizeTab
from ..views.tabs.combine_data_tab import CombineDataTab
from ..views.tabs.local_fit_tab import LocalFitTab
from ..views.tabs.component_tab import ComponentTab
from ..views.tabs.global_fit_tab import GlobalFitTab


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


pkg_root = resource_path()
gui_config = resource_path('configurations',  'default_gui_config.json')
with open(gui_config) as f:
    default_settings = json.load(f)


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


class MainWindow(QMainWindow):
    def __init__(self, abs_model, em_model, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3,
                 main_controller, import_controller, preproc_controller, visualize_controller,
                 combine_controller, local_fit_controller, component_controller,
                 global_fit_controller,  icon_color):
        super().__init__()

        self.project_path = None
        self.config_path = None
        self.config = ConfigManager()
        self.config.set_defaults(default_settings)
        self.abs_model = abs_model
        self.em_model = em_model
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.icon_color = icon_color
        self.inv_icon_color = "#000000" if self.icon_color == '#ffffff' else "#ffffff"
        self.models = (self.abs_model, self.em_model, self.ta_model,
                       self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        self.main_controller = main_controller
        self.import_controller = import_controller
        self.preproc_controller = preproc_controller
        self.visualize_controller = visualize_controller
        self.combine_controller = combine_controller
        self.local_fit_controller = local_fit_controller
        self.component_controller = component_controller
        self.global_fit_controller = global_fit_controller

        self.controllers = (
            self.main_controller, self.import_controller, self.preproc_controller,
            self.visualize_controller, self.combine_controller, self.local_fit_controller,
            self.component_controller, self.global_fit_controller)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Transient Absorption Processing & Analysis Software")
        try:
            plt.style.use(pkg_root / 'configurations' /
                          'default_style.mplstyle')
        except OSError:
            pass
        self._createActions()
        self._createMenuBar()
        self._createToolBar()
        self._createStatusBar()
        self._createTabWidget()
        self._createDefaultCmap()

    def _createMenuBar(self) -> None:
        ''' creates the menu bar and adds the actions '''
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('Project')

        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        config_menu = menu_bar.addMenu('Configurations')
        config_menu.addAction(self.open_config_action)
        config_menu.addAction(self.save_config_action)
        config_menu.addAction(self.save_config_as_action)

        help_menu = menu_bar.addMenu('Help')
        help_menu.addAction(self.documentation_action)
        help_menu.addAction(self.online_sourcecode_action)
        # help_menu.addAction(self.online_screencast_action)
        help_menu.addAction(self.local_sourcecode_action)
        help_menu.addSeparator()
        help_menu.addAction(self.acknowledgements_action)
        help_menu.addAction(self.third_party_license_action)
        help_menu.addAction(self.GPL_license_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)

    def _createStatusBar(self) -> None:
        ''' creates statusbar and listens to controller signals '''
        self.status_bar = self.statusBar()
        self.status_bar.setObjectName('default')
        self.status_bar.showMessage("Ready", 4000)
        self.project_status = QLabel("")
        self.status_bar.addPermanentWidget(self.project_status)
        # listen to info or error signals
        for i in self.models:
            i.status_signal.connect(self._showStatusMessage)
        for i in self.controllers:
            i.status_signal.connect(self._showStatusMessage)

    def _showStatusMessage(self, level: str, message: str) -> None:
        ''' sets font color depending on status level and shows the message '''
        self.status_bar.setObjectName(level)
        self.status_bar.setStyleSheet("")
        self.status_bar.showMessage(message, 4000)

    def _createToolBar(self) -> None:
        ''' creates toolbar and sets actions '''
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)
        size = int(tool_bar.iconSize().width()*1.4)

        tool_bar.setIconSize(QSize(size, size))
        tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tool_bar.addAction(self.new_action)
        tool_bar.addAction(self.open_action)
        tool_bar.addAction(self.save_action)
        tool_bar.addAction(self.save_as_action)
        tool_bar.addSeparator()
        tool_bar.addAction(self.open_config_action)
        tool_bar.addAction(self.save_config_action)
        tool_bar.addAction(self.save_config_as_action)
        tool_bar.addSeparator()
        tool_bar.setStyleSheet(f"""
                QToolBar::separator {{
                    background: {self.icon_color};
                    width: 1px;
                    margin: 2px 5px;
                }}
            """)

        self.pb_style = QPushButton("", objectName="style_button")
        self.pb_style.setToolTip(msg.ToolTips.t17)
        ico = load_svg_icon(
            str(pkg_root / 'assets'/'mpl_style.svg'), self.icon_color)
        self.pb_style.setIcon(ico)
        self.pb_style.setFlat(True)
        self.pb_style.setIconSize(tool_bar.iconSize())
        # self.pb_style.setStyleSheet("border: none")

        menu = QMenu()
        menu.addAction('Default', self._update_style)
        menu.addAction('MPL Default', self._update_style)
        menu.addAction('gg Plot', self._update_style)
        menu.addAction('Seaborn', self._update_style)
        menu.addAction('gray scale', self._update_style)
        menu.addAction('custom...', self._update_style)

        self.pb_style.setMenu(menu)
        tool_bar.addWidget(self.pb_style)
        tool_bar.addSeparator()

        tool_bar.addAction(self.exit_action)
        # tool_bar.setAllowedAreas(self.LeftToolBarArea)

    def _createDefaultCmap(self) -> None:
        ''' create the default mpl colormap used for contour plots '''
        self.colorlist = plt.cm.nipy_spectral(np.linspace(
            start=0, stop=1, num=7, endpoint=True))
        self.colorlist[int(3), :] = [0.5, .5, .5, 1]
        self.colorlist[0, :] = [0, 0, 0, 1]
        self.colorlist[6, :] = [1, 1, 1, 1]
        self.default_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'default_cmap', self.colorlist)
        try:
            mpl.colormaps.register(self.default_cmap)
        except ValueError:  # already registered
            pass
        plt.rcParams.update({'image.cmap': 'default_cmap'})

    def _update_style(self) -> None:
        ''' updates global mpl rcparams depending on user input '''
        try:
            if self.sender().text() == 'Default':
                plt.rcParams.update(mpl.rcParamsDefault)
                plt.rcParams.update({'image.cmap': 'default_cmap'})
                plt.style.use(pkg_root / 'configurations' /
                              'default_style.mplstyle')

            elif self.sender().text() == 'MPL Default':
                plt.rcParams.update(mpl.rcParamsDefault)
                plt.rcParams.update({'figure.autolayout': True})

            elif self.sender().text() == 'gg Plot':
                plt.rcParams.update(mpl.rcParamsDefault)
                plt.style.use('ggplot')
                plt.rcParams.update({'figure.autolayout': True})

            elif self.sender().text() == 'Seaborn':
                plt.rcParams.update(mpl.rcParamsDefault)
                plt.style.use('seaborn-v0_8')
                plt.rcParams.update({'image.cmap': 'viridis'})
                plt.rcParams.update({'figure.autolayout': True})
            elif self.sender().text() == 'gray scale':
                plt.rcParams.update(mpl.rcParamsDefault)
                plt.style.use('grayscale')
                plt.rcParams.update({'image.cmap': 'Greys'})
                plt.rcParams.update({'figure.autolayout': True})

            elif self.sender().text() == 'custom...':
                plt.rcParams.update(mpl.rcParamsDefault)
                default_path = pkg_root / 'configurations'
                filename, _ = QFileDialog.getOpenFileName(
                    self, 'Open Style Sheet', filter="*.mplstyle", directory=str(default_path))
                if not filename:
                    return

                plt.style.use(Path(filename))

            self.import_tab_widget.preplot_data("abs")
            self.import_tab_widget.preplot_data("em")
            self.import_tab_widget.preplot_data("ta")
            self.preproc_tab_widget.plot_data()
            self.visualize_tab_widget.tw_select_plot.le_style.setText(
                self.sender().text())

        except OSError:  # package style not found
            self._showStatusMessage("error", msg.Error.e03)
            plt.rcParams.update(mpl.rcParamsDefault)
            plt.rcParams.update({'figure.autolayout': True})

    def _createActions(self):
        ''' create the actions used by the tool- and menubar '''
        self.new_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'new_project.svg'), self.icon_color), 'New Project', self)
        self.new_action.setToolTip(msg.ToolTips.t01)
        self.new_action.setShortcut(
            QKeySequence(QKeySequence.StandardKey.New))
        self.new_action.triggered.connect(self.new_project)

        # open menu item
        self.open_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'open_project.svg'), self.icon_color), 'Open Project', self)
        self.open_action.setToolTip(msg.ToolTips.t02)
        self.open_action.setShortcut(
            QKeySequence(QKeySequence.StandardKey.Open))
        self.open_action.triggered.connect(self.open_project_gui)

        # save menu item
        self.save_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'save_project.svg'), self.icon_color), 'Save Project', self)
        self.save_action.setToolTip(msg.ToolTips.t03)
        self.save_action.setShortcut(
            QKeySequence(QKeySequence.StandardKey.Save))
        self.save_action.triggered.connect(self.save_project_gui)

        # save menu item
        self.save_as_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'save_as.svg'), self.icon_color), 'Save Project As', self)
        self.save_as_action.setToolTip(msg.ToolTips.t03)
        self.save_as_action.setShortcut(
            QKeySequence(QKeySequence.StandardKey.SaveAs))
        self.save_as_action.triggered.connect(self.save_project_as_gui)

        # exit menu item
        self.exit_action = QAction(load_svg_icon(
            str(pkg_root / 'assets'/'x.svg'), self.icon_color), 'Exit', self)
        self.exit_action.setToolTip(msg.ToolTips.t04)
        self.exit_action.setShortcut(
            QKeySequence(QKeySequence.StandardKey.Close))
        self.exit_action.triggered.connect(self.close)

        self.open_config_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'open_config.svg'), self.icon_color), 'Open Config', self)
        self.open_config_action.triggered.connect(self.open_config)

        # save current configuration
        self.save_config_as_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'save_config_as.svg'), self.icon_color), 'Save Config As', self)
        self.save_config_as_action.triggered.connect(self.save_config_as)

        self.save_config_action = QAction(load_svg_icon(str(
            pkg_root / 'assets'/'save_config.svg'), self.icon_color), 'Save Config', self)
        self.save_config_action.triggered.connect(self.save_config)

        self.documentation_action = QAction(load_svg_icon(str(
            pkg_root / "assets" / "doc.svg"), self.icon_color), 'Online Documentation', self)
        self.documentation_action.triggered.connect(self.online_docs)

        self.online_sourcecode_action = QAction(load_svg_icon(str(
            pkg_root / "assets" / "brand-github.svg"), self.icon_color), 'Online Source Code', self)
        self.online_sourcecode_action.triggered.connect(self.online_sourcecode)

        self.online_screencast_action = QAction(load_svg_icon(str(
            pkg_root / "assets" / "brand-youtube.svg"), self.icon_color), 'Online Screencast', self)

        self.local_sourcecode_action = QAction(load_svg_icon(str(
            pkg_root / "assets" / "code.svg"), self.icon_color), 'Local Source Code', self)
        self.local_sourcecode_action.triggered.connect(self.open_local_code)

        self.acknowledgements_action = QAction(load_svg_icon(str(
            pkg_root / "assets" / "acknowledgements.svg"), self.icon_color), 'Third-Party Acknowledgments', self)
        self.acknowledgements_action.triggered.connect(lambda: self.show_text_file(
            path=pkg_root / 'THIRD-PARTY_ACKNOWLEDGEMENTS.txt', title='Third-Party Acknowledgments'))

        self.third_party_license_action = QAction(load_svg_icon(str(
            pkg_root / "assets" / "third_party_license.svg"), self.icon_color), 'Third-Party Licenses', self)
        self.third_party_license_action.triggered.connect(lambda: self.show_text_file(
            path=pkg_root / 'THIRD-PARTY_LICENSES.txt', title='Third-Party Licenses'))

        self.GPL_license_action = QAction(load_svg_icon(
            str(pkg_root / "assets" / "license.svg"), self.icon_color), 'License', self)
        self.GPL_license_action.triggered.connect(lambda: self.show_text_file(
            path=pkg_root / 'LICENSE', title='Licensed under GPL-3.0-or-later'))

        self.about_action = QAction(load_svg_icon(
            str(pkg_root / 'assets'/'about.svg'), self.icon_color), 'About', self)
        self.about_action.setToolTip(msg.ToolTips.t05)
        self.about_action.setShortcut('F1')
        self.about_action.triggered.connect(self.show_version)

    def _createTabWidget(self):
        ''' create the Tabs, Icons and connect the controllers and models '''

        self.tabwidget = QTabWidget(self)
        self.setCentralWidget(self.tabwidget)
        self.import_tab_widget = ImportTab(
            self.tabwidget, self.abs_model, self.em_model, self.ta_model, self.import_controller,
            self.config)
        self.preproc_tab_widget = PreprocTab(
            self.tabwidget, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3,
            self.preproc_controller, self.config)
        self.combine_data_tab_widget = CombineDataTab(
            self.tabwidget, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3,
            self.combine_controller, self.config)
        self.visualize_tab_widget = VisualizeTab(
            self.tabwidget, self.abs_model, self.em_model, self.ta_model, self.ta_model_ds1,
            self.ta_model_ds2, self.ta_model_ds3, self.visualize_controller, self.config)
        self.local_fit_tab_widget = LocalFitTab(
            self.tabwidget, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3,
            self.local_fit_controller, self.config)
        self.component_tab_widget = ComponentTab(
            self.tabwidget, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3,
            self.component_controller, self.config)
        self.global_fit_tab_widget = GlobalFitTab(
            self.tabwidget, self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3,
            self.global_fit_controller, self.config)

        self.tabwidget.addTab(self.import_tab_widget, load_svg_icon(
            str(pkg_root / "assets" / "import.svg"), self.inv_icon_color), "Import")
        self.tabwidget.addTab(self.preproc_tab_widget, load_svg_icon(str(
            pkg_root / "assets" / "preprocess.svg"), self.inv_icon_color), "Preprocessing")
        self.tabwidget.addTab(self.combine_data_tab_widget, load_svg_icon(str(
            pkg_root / "assets" / "combine.svg"), self.inv_icon_color), "Combine Projects")
        self.tabwidget.addTab(self.visualize_tab_widget, load_svg_icon(str(
            pkg_root / "assets" / "visualization.svg"), self.inv_icon_color), "Visualization")
        self.tabwidget.addTab(self.component_tab_widget, load_svg_icon(
            str(pkg_root / "assets" / "SVD.svg"), self.inv_icon_color), "Component Analysis")
        self.tabwidget.addTab(self.local_fit_tab_widget, load_svg_icon(str(
            pkg_root / "assets" / "local_fit.svg"), self.inv_icon_color), "Local Fitting")
        self.tabwidget.addTab(self.global_fit_tab_widget, load_svg_icon(str(
            pkg_root / "assets" / "global_fit.svg"), self.inv_icon_color), "Global Fitting")

        self.tabwidget.currentChanged.connect(self._onTabSelected)

        self.combine_data_tab_widget.project_path_changed.connect(
            self._update_projet_path)

    @pyqtSlot(int)
    def _onTabSelected(self, index: int) -> None:
        ''' called when a tab is selected and selectively triggers the refresh of the current tab '''
        if index == 0:
            self.import_tab_widget.update_config()
            self.import_tab_widget.update_GUI()
        elif index == 1:
            self.preproc_tab_widget.update_config()
            self.preproc_tab_widget.update_GUI()
        elif index == 2:
            self.combine_data_tab_widget.update_config()
            self.combine_data_tab_widget.update_GUI()
        elif index == 3:
            self.visualize_tab_widget.update_config()
            self.visualize_tab_widget.update_GUI()
        elif index == 4:
            self.component_tab_widget.update_config()
            self.component_tab_widget.update_GUI()

        elif index == 5:
            self.local_fit_tab_widget.update_config()
            self.local_fit_tab_widget.update_GUI()
        elif index == 6:
            self.global_fit_tab_widget.update_config()
            self.global_fit_tab_widget.update_GUI()

    def _update_projet_path(self, path: bool | str = True) -> None:
        ''' resets the project_path. also needed if new project is created '''
        if path == 'reset':
            self.project_path = None
            self.project_status.setText('')
        else:
            self.project_status.setText(
                "Project Path: " + str(self.project_path))
        self.visualize_tab_widget.project_path = self.project_path
        self.local_fit_tab_widget.project_path = self.project_path
        self.global_fit_tab_widget.project_path = self.project_path

        if self.visualize_tab_widget.tw_select_plot.le_results_dir.text() == '':
            self.visualize_tab_widget.tw_select_plot.le_results_dir.setText(
                str(self.project_path.parent))

    def new_project(self) -> None:
        ''' action, that create a detached new TAPAS instance in all install modes.'''
        argv = QCoreApplication.arguments()            # e.g. ["app.py", …] or ["__main__.py", …] or ["tapas", …]
        prog = QCoreApplication.applicationFilePath()  # e.g. "/usr/bin/python3" or ".../tapas.exe"
        cwd  = QCoreApplication.applicationDirPath()
    
        name0 = Path(argv[0]).name
    
        prog_stem  = Path(prog).stem.lower()

        # 1) PyInstaller bundle or real EXE: prog is NOT python
        if not prog_stem.startswith("python"):
            # drop the first element (the exe path)
            args = argv[1:]
    
        # 2) Launched by python interpreter
        else:
            # a) module form: argv[0] == "__main__.py"
            if name0 == "__main__.py":
                args = ["-m", "tapas", *argv[1:]]
            # b) direct-script form: argv[0].endswith(".py")
            elif name0.lower().endswith(".py"):
                args = argv
            # c) console-script shim: python.exe [no .py in argv[0]]
            else:
                args = ["-m", "tapas", *argv[1:]]
    
        QProcess.startDetached(prog, args, cwd)

    def open_project_gui(self) -> None:
        ''' action, that gets project path via FileDialog, loads new project and configurations '''
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', filter="*.hdf5")
        if not filename:
            return

        self.project_path = Path(filename)
        self._update_projet_path()
        new_config = self.main_controller.open_project(self.project_path)
        self.import_tab_widget.update_metadata_GUI()
        if new_config:
            self._update_config(new_config=new_config)
        self._showStatusMessage("info", msg.Status.s07)

    def save_project_gui(self) -> None:
        ''' action, that informs main controller to save project to path '''
        if self.project_path is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Save Project', filter='*.hdf5'
            )
            if not filename:
                return
            self.project_path = Path(filename)
            self._update_projet_path()
        self.main_controller.save_project(
            project_path=self.project_path, gui_config=self.config.as_dict())

    def save_project_as_gui(self) -> None:
        ''' action, that sets project path via FileDialog and informs main controller to save project '''
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Project', filter='*.hdf5'
        )
        if not filename:
            return
        self.project_path = Path(filename)
        self._update_projet_path()

        self.main_controller.save_project(
            project_path=self.project_path, gui_config=self.config.as_dict())

    def save_config(self) -> None:
        ''' action, that saves the configuration file to .json '''

        if self.config_path is None:
            filename, _ = QFileDialog.getSaveFileName(self, 'Save Configuration', filter='*.json')
            if not filename:
                return
            self.config_path = Path(filename)

        try:
            os.remove(self.config_path)

        except OSError:
            pass

        config_buffer = ConfigManager(filename=self.config_path)
        config_buffer.set_defaults(self.config.as_dict())
        config_buffer.save()
        self._showStatusMessage("info", msg.Status.s17)

    def save_config_as(self) -> None:
        ''' action, that sets configuration path via FileDialog and saves the configuration file to .json '''
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Configuration', filter='*.json')
        if not filename:
            return

        self.config_path = Path(filename)
        try:
            os.remove(self.config_path)

        except OSError:
            pass

        config_buffer = ConfigManager(filename=self.config_path)
        config_buffer.set_defaults(self.config.as_dict())
        config_buffer.save()
        self._showStatusMessage("info", msg.Status.s17)

    def open_config(self) -> None:
        ''' action, that gets config path via FileDialog and loads new config file'''
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Configuration', filter="*.json")
        if not filename:
            return

        self.config_path = Path(filename)
        try:
            with open(self.config_path) as f:
                new_config = json.load(f)
        except ValueError:  # corrupted json file
            self._showStatusMessage("error", msg.Error.e02)
            return
        self._update_config(new_config=new_config)

    def _update_config(self, new_config: dict) -> None:
        ''' resets the GUI config and updates it on the current selected Tab '''
        self.config.reset()
        self.config.set_defaults(new_config)
        # update only the config of the selected tab, to speed up project load
        self._onTabSelected(index=self.tabwidget.currentIndex())

        self._showStatusMessage("info", msg.Status.s16)

    def show_version(self) -> None:
        ''' action, that shows a messagebox with the Version content '''
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("About")
        msgBox.setText(msg.Status.s00)
        msgBox.setInformativeText(msg.License.notes)
        icon = load_svg_icon(
            str(pkg_root / "assets" / "icon.svg"), self.icon_color)
        msgBox.setIconPixmap(icon.pixmap(75, 75))
        msgBox.exec()

    def online_sourcecode(self) -> None:
        ''' action, that opens the github project page on the default browser '''
        url = 'https://github.com/pytapas/tapas'
        QDesktopServices.openUrl(QUrl(url))

    def online_docs(self) -> None:
        ''' action, that opens the rtd project page on the default browser '''
        url = 'https://tapas-docs.readthedocs.io/en/latest/'
        QDesktopServices.openUrl(QUrl(url))

    def open_local_code(self):
        ''' action, that opens the explorer at the root source code directory '''
        url = QUrl.fromLocalFile(str(pkg_root))
        QDesktopServices.openUrl(url)

    def show_text_file(self, path: Path, title: str) -> None:
        ''' action, that shows the text file of a given Path object and sets the title.
        used by third_party_acknowledgments and licenses
        '''

        file_path = path
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return

        dialog = QDialog()
        dialog.setWindowTitle(title)
        dialog.resize(600, 400)

        text_edit = QTextEdit(dialog)
        text_edit.setPlainText(content)
        text_edit.setReadOnly(True)

        layout = QVBoxLayout(dialog)
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.exec()
