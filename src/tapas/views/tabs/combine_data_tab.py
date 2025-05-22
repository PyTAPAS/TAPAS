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
from pathlib import Path
import logging
import logging.config
# Third‑Party Imports

# PyQt6 Imports
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
    QFileDialog,
    QMessageBox,
)

# Matplotlib and Related Imports
from matplotlib import colors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import matplotlib.ticker as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Local Application Imports
from ...utils import utils
from ...configurations import exceptions as exc, messages as msg

from ...views.tabwidgets.combine_data_tabwidgets import (
    ImportProjectsWidget,
    CombineProjectsWidget,
    PreviewContainerWidget,
)

logger = logging.getLogger(__name__)


class CombineDataTab(QWidget):
    def __init__(self, tab,   ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3, controller, config):
        super().__init__()
        self.tab = tab
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.combine_controller = controller
        self.need_GUI_update = False
        self.config = config
        self.InitUI()
        
    project_path_changed = pyqtSignal(str)

    def InitUI(self):
        # -------- create Widgets ------------------------------------------------------------------
        self.tw_import_projects = ImportProjectsWidget()
        self.tw_combine_projects = CombineProjectsWidget()
        self.tw_preview_container = PreviewContainerWidget()
        self.update_config()

        # -------- add Widgets to layout -----------------------------------------------------------
        self.combine_layout = QGridLayout()
        self.setLayout(self.combine_layout)
        self.combine_layout.addWidget(
            self.tw_import_projects, 0, 0, 1, 4, alignment=Qt.AlignmentFlag.AlignTop)
        self.combine_layout.addWidget(
            self.tw_combine_projects, 0, 4, alignment=Qt.AlignmentFlag.AlignTop)
        self.combine_layout.addWidget(self.tw_preview_container, 1, 0, 1, 5)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.tw_import_projects.pb_browse_p1.pressed.connect(
            self.open_project_gui)
        self.tw_import_projects.pb_browse_p2.pressed.connect(
            self.open_project_gui)
        self.tw_import_projects.pb_load_p1.pressed.connect(lambda: self.display_loaded_project(
            self.tw_import_projects.le_path_p1.text(),
            self.tw_import_projects.cb_ds_p1.currentText()))
        self.tw_import_projects.pb_load_p2.pressed.connect(lambda: self.display_loaded_project(
            path=self.tw_import_projects.le_path_p2.text(),
            ds=self.tw_import_projects.cb_ds_p2.currentText()))
        self.tw_import_projects.pb_clear_p1.pressed.connect(self.clear_p1)
        self.tw_import_projects.pb_clear_p2.pressed.connect(self.clear_p2)
        self.tw_combine_projects.pb_preview_combine_projects.pressed.connect(
            self.preview_combination)
        self.tw_combine_projects.pb_apply_combine_projects.pressed.connect(
            self.ask_override_current_project)

        # -------- listen to model event signals ---------------------------------------------------
        self.ta_model.rawdata_changed.connect(self.queue_update_GUI)

    def queue_update_GUI(self) -> None:
        ''' called, if the raw data is changed. GUI update waits till tab is selected '''
        self.need_GUI_update = True
        if self.tab.currentIndex() == 2:
            self.update_GUI()

    def update_GUI(self) -> None:
        ''' function called directly by the main window everytime the Tab is clicked
        or if the Tab is active and data was changed (handled by queue_update_GUI).
        Tab is updated if needed (handled by the need_GUI_update boolean). '''
        if self.need_GUI_update:
            self._clear_all()
            self.need_GUI_update = False

    def update_config(self) -> None:
        '''updates configuration and standard values of QWidgets'''
        self.config.add_handler('combine_cb_overlap_use',
                                self.tw_combine_projects.cb_overlap_use)
        self.config.add_handler('combine_cb_interpolation_use',
                                self.tw_combine_projects.cb_interpolation_use)
        self.config.add_handler('combine_cb_extrapolation_use',
                                self.tw_combine_projects.cb_extrapolation_use)
        self.config.add_handler('combine_cb_ds_p1',
                                self.tw_import_projects.cb_ds_p1)
        self.config.add_handler('combine_cb_ds_p2',
                                self.tw_import_projects.cb_ds_p2)

    def open_project_gui(self) -> None:
        ''' opens file dialog and sets the path to the corresponding lineedit widget '''
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', filter="*.hdf5")
        if not filename:
            return
        if self.sender().objectName() == 'pb_browse_p1':

            self.tw_import_projects.le_path_p1.setText(str(Path(filename)))

        elif self.sender().objectName() == 'pb_browse_p2':

            self.tw_import_projects.le_path_p2.setText(str(Path(filename)))

    def preview_combination(self) -> None:
        ''' reads user input, asks controller to combine the datasets
            (cached by controller) and plots the data '''
        overlap_use = self.tw_combine_projects.cb_overlap_use.currentIndex()
        interpol_method = self.tw_combine_projects.cb_interpolation_use.currentText()
        extrapol_method = self.tw_combine_projects.cb_extrapolation_use.currentText()
        try:
            self.combine_controller.combine_ds(overlap_use, interpol_method, extrapol_method)
        except exc.NoDataError:
            return
        self.preplot_combined()

    def ask_override_current_project(self) -> None:
        '''
        triggered, when merged dataset will be applied to model.
        Combined dataset will be the new rawdata of a new project. User gets a
        warning, if there is already a project opend

        Returns
        -------
        None.

        '''
        overlap_use = self.tw_combine_projects.cb_overlap_use.currentIndex()
        interpol_method = self.tw_combine_projects.cb_interpolation_use.currentText()
        extrapol_method = self.tw_combine_projects.cb_extrapolation_use.currentText()
        if self.ta_model.rawdata:

            answer = QMessageBox.question(self,
                                          'Data in current project detected',
                                          msg.Widgets.i20,
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self.combine_controller.combine_ds(
                        overlap_use, interpol_method, extrapol_method)
                except exc.NoDataError:
                    return
                self.preplot_combined()
                self.project_path_changed.emit('reset')
            else:
                return
        else:
            try:
                self.combine_controller.combine_ds(overlap_use, interpol_method, extrapol_method)
            except exc.NoDataError:
                return
            self.preplot_combined()

    def display_loaded_project(self, path: str, ds: str) -> None:
        '''
        calls self.combine_controller.load_project which reads and caches the project data. 
        previews bounds and spectra and prepares everything for merging

        Parameters
        ----------
        path : str
            points to .hdf project.
        ds : str
            either raw or ds.

        Returns
        -------
        None.

        '''
        if path == '':
            self.combine_controller.call_statusbar("error", msg.Error.e03)
            return
        else:
            xmin, xmax, ymin, ymax = self.combine_controller.load_project(
                path, ds)

            if self.sender().objectName() == 'pb_load_p1':
                if xmin is None or xmax is None or ymin is None or ymax is None:
                    self.tw_import_projects.le_x_range_p1.setText('None')
                    self.tw_import_projects.le_y_range_p1.setText('None')
                    utils._remove_widget(self.tw_preview_container.p1_window)
                    self.tw_preview_container.p1_window = QLabel(
                        msg.Widgets.i15)
                    self.tw_preview_container.inner_layout.addWidget(
                        self.tw_preview_container.p1_window, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
                else:
                    self.tw_import_projects.le_x_range_p1.setText(str(tk.EngFormatter(places=0, sep="\N{THIN SPACE}")(
                        xmin) + 'm, ' + (str(tk.EngFormatter(places=0, sep="\N{THIN SPACE}")(xmax) + 'm'))))
                    self.tw_import_projects.le_y_range_p1.setText(str(tk.EngFormatter(places=2, sep="\N{THIN SPACE}")(
                        ymin) + 's, ' + (str(tk.EngFormatter(places=2, sep="\N{THIN SPACE}")(ymax) + 's'))))
                    self.preplot_p1()
            if self.sender().objectName() == 'pb_load_p2':
                if xmin is None or xmax is None or ymin is None or ymax is None:
                    self.tw_import_projects.le_x_range_p2.setText('None')
                    self.tw_import_projects.le_y_range_p2.setText('None')
                    utils._remove_widget(self.tw_preview_container.p2_window)
                    self.tw_preview_container.p2_window = QLabel(
                        msg.Widgets.i15)
                    self.tw_preview_container.inner_layout.addWidget(
                        self.tw_preview_container.p2_window, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
                else:
                    self.tw_import_projects.le_x_range_p2.setText(str(tk.EngFormatter(places=0, sep="\N{THIN SPACE}")(
                        xmin) + 'm, ' + (str(tk.EngFormatter(places=0, sep="\N{THIN SPACE}")(xmax) + 'm'))))
                    self.tw_import_projects.le_y_range_p2.setText(str(tk.EngFormatter(places=2, sep="\N{THIN SPACE}")(
                        ymin) + 's, ' + (str(tk.EngFormatter(places=2, sep="\N{THIN SPACE}")(ymax) + 's'))))
                    self.preplot_p2()

    def clear_p1(self) -> None:
        ''' cleans gui and requests the controller to delete the cached data '''
        self.tw_import_projects.le_path_p1.setText('')
        self.tw_import_projects.le_x_range_p1.setText('None')
        self.tw_import_projects.le_y_range_p1.setText('None')
        utils._remove_widget(self.tw_preview_container.p1_window)
        self.tw_preview_container.p1_window = QLabel(msg.Widgets.i15)
        self.tw_preview_container.inner_layout.addWidget(
            self.tw_preview_container.p1_window, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        self.combine_controller.delete_project('p1')

    def clear_p2(self) -> None:
        ''' cleans gui and requests the controller to delete the cached data '''
        self.tw_import_projects.le_path_p2.setText('')
        self.tw_import_projects.le_x_range_p2.setText('None')
        self.tw_import_projects.le_y_range_p2.setText('None')
        utils._remove_widget(self.tw_preview_container.p2_window)
        self.tw_preview_container.p2_window = QLabel(msg.Widgets.i15)
        self.tw_preview_container.inner_layout.addWidget(
            self.tw_preview_container.p2_window, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.combine_controller.delete_project('p2')

    def _clear_all(self) -> None:
        ''' cleans everything when new project is loaded '''
        self.clear_p1()
        self.clear_p2()
        utils._remove_widget(self.tw_preview_container.merge_window)
        self.tw_preview_container.merge_window = QLabel(msg.Widgets.i16)
        self.tw_preview_container.inner_layout.addWidget(
            self.tw_preview_container.merge_window, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)

    def _init_plot(self, project: str) -> None:
        """
        Initialize a PlotCanvas for the specified dataset and return a layout containing
        the canvas with its navigation toolbar.

        Parameters
        ----------
        project : str
            Key identifying which dataset to load and plot (e.g., 'p1', 'p2', or 'combined').

        Returns
        -------
        QLayout
            A QVBoxLayout containing the NavigationToolbar2QT and the PlotCanvas widget.
        """
        sc = utils.PlotCanvas(self, width=5, height=5, dpi=100)

        setattr(self, f"sc_{project}", sc)

        sc.axes_mapping = {}
        sc.mpl_connect(
            "scroll_event",
            lambda event, _sc=sc: _sc._zoom_TA(event, _sc.axes_mapping)
        )

        fig = sc.fig
        fig.clear()
        ax = fig.add_subplot(111)

        x, y, Z = self.combine_controller.return_ds(project)
        X, Y = np.meshgrid(x, y)
        mesh = ax.pcolormesh(
            X, Y, Z,
            shading="auto",
            norm=colors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)
        )
        ax.set_xlabel(msg.Labels.wavelength)
        ax.set_ylabel(msg.Labels.delay)
        ax.xaxis.set_major_formatter(sc.nm_formatter_ax)
        ax.yaxis.set_major_formatter(sc.delay_formatter0)
        ax.set_title("TA Data")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(mesh, cax=cax, shrink=0.6,
                          label=msg.Labels.delA).minorticks_on()

        sc.axes_mapping[ax] = (mesh, cb)

        toolbar = NavigationToolbar2QT(sc)
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)
        sc.draw()
        return layout

    def preplot_p1(self) -> None:
        ''' requesting the plot Canvas and updating the GUI with project 1 '''
        layout = self._init_plot('p1')

        utils._remove_widget(self.tw_preview_container.p1_window)
        self.tw_preview_container.p1_window = QWidget()
        self.tw_preview_container.p1_window.setLayout(layout)
        self.tw_preview_container.inner_layout.addWidget(
            self.tw_preview_container.p1_window, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)

    def preplot_p2(self) -> None:
        ''' requesting the plot Canvas and updating the GUI with project 2 '''
        layout = self._init_plot('p2')

        utils._remove_widget(self.tw_preview_container.p2_window)
        self.tw_preview_container.p2_window = QWidget()
        self.tw_preview_container.p2_window.setLayout(layout)
        self.tw_preview_container.inner_layout.addWidget(
            self.tw_preview_container.p2_window, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)

    def preplot_combined(self) -> None:
        ''' requesting the plot Canvas and updating the GUI with the merged project '''
        layout = self._init_plot('combined')

        utils._remove_widget(self.tw_preview_container.merge_window)
        self.tw_preview_container.merge_window = QWidget()
        self.tw_preview_container.merge_window.setLayout(layout)
        self.tw_preview_container.inner_layout.addWidget(
            self.tw_preview_container.merge_window, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
